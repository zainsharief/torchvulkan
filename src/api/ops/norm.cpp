#include <torch/extension.h>
#include <c10/util/accumulate.h>
#include <limits>
#include <optional>
#include <tuple>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "api/ops/helpers.h"
#include "api/ops/internal.h"

namespace {

enum class NormMode : uint32_t {
    BATCH_FORWARD = 0,
    BATCH_BACKWARD = 1,
    ROW_FORWARD = 2,
    ROW_BACKWARD_INPUT = 3,
    ROW_BACKWARD_PARAMS = 4
};

const uint32_t NORM_PARAM_COLUMNS = 32; // PARAM_COLUMNS in norm.slang

struct NormArgs {
    at::Tensor x;
    at::Tensor dy;
    at::Tensor y;
    at::Tensor weight;
    at::Tensor bias;
    at::Tensor mean;
    at::Tensor rstd;
    at::Tensor running_mean;
    at::Tensor running_var;
    at::Tensor grad_weight;
    at::Tensor grad_bias;
    uint32_t outer = 0;
    uint32_t channels = 0;
    uint32_t inner = 1;
    double eps = 0.0;
    double momentum = 0.0;
    bool training = false;
    bool update_running = false;
    bool rms = false;
    bool stats_accum = false;
    uint32_t outputs = 0; // bit 0 grad_input, bit 1 grad_weight, bit 2 grad_bias
};

uint64_t address_or_zero(const at::Tensor& t)
{
    return t.defined() ? get_tensor_address(t) : uint64_t(0);
}

uint32_t norm_workgroup_size(const DeviceContext* device)
{
    // a power of two, at most the 256 the shader's shared array holds
    uint32_t limit = std::min({256u, device->properties.limits.maxComputeWorkGroupInvocations, device->properties.limits.maxComputeWorkGroupSize[0]});
    uint32_t size = 1;
    while (size * 2 <= limit) size *= 2;
    return size;
}

void dispatch_norm_shader(NormMode mode, const NormArgs& a, uint32_t groups)
{
    DeviceContext* device = get_device_context(a.x);
    uint32_t workgroupSizeX = norm_workgroup_size(device);
    bool forward = mode == NormMode::BATCH_FORWARD || mode == NormMode::ROW_FORWARD;

    SpecializationBuilder spd{};
    spd.push(static_cast<uint32_t>(mode))
       .push(workgroupSizeX)
       .push(static_cast<uint32_t>(a.weight.defined()))
       .push(static_cast<uint32_t>(a.bias.defined()))
       .push(static_cast<uint32_t>(a.training))
       .push(static_cast<uint32_t>(a.update_running))
       .push(static_cast<uint32_t>(a.rms))
       .push(static_cast<uint32_t>(a.stats_accum))
       .push(a.outputs);

    PushConstantBuilder pcs{};
    pcs.push(address_or_zero(a.x))
       .push(address_or_zero(a.dy))
       .push(address_or_zero(a.y))
       .push(address_or_zero(a.weight))
       .push(address_or_zero(a.bias))
       .push(address_or_zero(a.mean))
       .push(address_or_zero(a.rstd))
       .push(address_or_zero(a.running_mean))
       .push(address_or_zero(a.running_var))
       .push(address_or_zero(a.grad_weight))
       .push(address_or_zero(a.grad_bias))
       .push(a.outer)
       .push(a.channels)
       .push(a.inner)
       .push(static_cast<float>(a.eps))
       .push(static_cast<float>(a.momentum));

    std::vector<at::Tensor> reads, writes;
    auto add = [](std::vector<at::Tensor>& list, const at::Tensor& t) { if (t.defined()) list.push_back(t); };
    add(reads, a.x);
    add(reads, a.dy);
    add(reads, a.weight);
    add(reads, a.bias);
    add(reads, a.running_mean);
    add(reads, a.running_var);
    add(writes, a.y);
    add(forward ? writes : reads, a.mean);
    add(forward ? writes : reads, a.rstd);
    if (a.update_running) {
        add(writes, a.running_mean);
        add(writes, a.running_var);
    }
    add(writes, a.grad_weight);
    add(writes, a.grad_bias);

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::NORM, a.x.scalar_type()},
        device->default_subgroup_size,
        spd.build(),
        pcs.build(),
        /* read = */ reads,
        /* write = */ writes,
        groups, 1, 1
    );
}

bool can_use_norm_shader(const at::Tensor& input)
{
    return input.device().type() == c10::DeviceType::PrivateUse1 && is_dtype_float(input.scalar_type()) &&
        is_dtype_supported(get_device_context(input), input.scalar_type()) && input.numel() > 0 &&
        input.numel() <= std::numeric_limits<int32_t>::max();
}

at::Tensor as_dtype(const at::Tensor& t, at::ScalarType dtype)
{
    return t.defined() ? t.to(dtype).contiguous() : t;
}

std::vector<int64_t> row_stat_shape(const at::Tensor& input, size_t normalized_dims)
{
    std::vector<int64_t> shape(input.sizes().begin(), input.sizes().end());
    for (size_t i = shape.size() - normalized_dims; i < shape.size(); i++) shape[i] = 1;
    return shape;
}

int64_t normalized_numel(const at::Tensor& input, size_t normalized_dims)
{
    return c10::multiply_integers(input.sizes().slice(input.dim() - normalized_dims));
}

std::tuple<at::Tensor, at::Tensor> row_backward_params(
    NormArgs args,
    const at::Tensor& weight,
    const at::Tensor& bias,
    bool want_weight,
    bool want_bias)
{
    args.y = at::Tensor();
    args.weight = at::Tensor();
    args.outputs = (want_weight ? 2u : 0u) | (want_bias ? 4u : 0u);
    if (want_weight) args.grad_weight = at::empty(weight.sizes(), args.x.options());
    if (want_bias) args.grad_bias = at::empty(bias.sizes(), args.x.options());
    dispatch_norm_shader(NormMode::ROW_BACKWARD_PARAMS, args, static_cast<uint32_t>(ceil_div(args.channels, NORM_PARAM_COLUMNS)));

    at::Tensor grad_weight = want_weight ? args.grad_weight.to(weight.scalar_type()) : at::Tensor();
    at::Tensor grad_bias = want_bias ? args.grad_bias.to(bias.scalar_type()) : at::Tensor();
    return {grad_weight, grad_bias};
}

std::tuple<at::Tensor, at::Tensor> fused_rms_norm_vulkan(
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    c10::optional<double> eps)
{
    at::ScalarType dtype = input.scalar_type();
    at::ScalarType compute = reduce_accumulate_dtype(dtype);
    double eps_value = eps.value_or(compute == at::kDouble ? std::numeric_limits<double>::epsilon() : std::numeric_limits<float>::epsilon());
    at::Tensor weight = weight_opt.has_value() && weight_opt->defined() ? *weight_opt : at::Tensor();
    size_t nnorm = normalized_shape.size();

    if (!can_use_norm_shader(input) || (weight.defined() && weight.device() != input.device())) {
        std::vector<int64_t> dims;
        for (size_t i = 0; i < nnorm; i++) dims.push_back(input.dim() - 1 - static_cast<int64_t>(i));
        at::Tensor x = input.to(compute);
        at::Tensor rstd = at::rsqrt(at::add(at::mean(at::pow(x, 2), dims, /*keepdim=*/true), eps_value));
        at::Tensor y = at::mul(x, rstd);
        if (weight.defined()) y = at::mul(y, weight);
        return {y.contiguous().to(dtype), rstd.contiguous()};
    }

    NormArgs args;
    args.x = input.contiguous();
    args.weight = as_dtype(weight, dtype);
    args.channels = static_cast<uint32_t>(normalized_numel(input, nnorm));
    args.outer = static_cast<uint32_t>(input.numel() / args.channels);
    args.eps = eps_value;
    args.rms = true;
    args.stats_accum = true;
    args.y = at::empty(input.sizes(), args.x.options());
    args.rstd = at::empty(row_stat_shape(input, nnorm), args.x.options().dtype(compute));
    dispatch_norm_shader(NormMode::ROW_FORWARD, args, args.outer);
    return {args.y, args.rstd};
}

std::tuple<at::Tensor, at::Tensor> fused_rms_norm_backward_vulkan(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const at::Tensor& rstd,
    const c10::optional<at::Tensor>& weight_opt,
    std::array<bool, 2> output_mask)
{
    at::ScalarType dtype = input.scalar_type();
    at::ScalarType compute = reduce_accumulate_dtype(dtype);
    at::Tensor weight = weight_opt.has_value() && weight_opt->defined() ? *weight_opt : at::Tensor();
    size_t nnorm = normalized_shape.size();
    bool want_weight = output_mask[1] && weight.defined();

    if (!can_use_norm_shader(input) || grad_out.sizes() != input.sizes()) {
        std::vector<int64_t> inner, outer;
        for (int64_t i = 0; i < input.dim(); i++) (i < input.dim() - static_cast<int64_t>(nnorm) ? outer : inner).push_back(i);
        at::Tensor x = input.to(compute);
        at::Tensor dy = grad_out.to(compute);
        at::Tensor r = rstd.to(compute);
        at::Tensor g = weight.defined() ? at::mul(dy, weight.to(compute)) : dy;
        at::Tensor xhat = at::mul(x, r);
        at::Tensor grad_input, grad_weight;
        if (output_mask[0]) {
            double n = static_cast<double>(normalized_numel(input, nnorm));
            at::Tensor proj = at::sum(at::mul(xhat, g), inner, /*keepdim=*/true);
            grad_input = at::mul(at::sub(g, at::mul(at::div(xhat, n), proj)), r).to(dtype);
        }
        if (want_weight) {
            at::Tensor gw = at::mul(dy, xhat);
            grad_weight = (outer.empty() ? gw : at::sum(gw, outer)).to(weight.scalar_type());
        }
        return {grad_input, grad_weight};
    }

    NormArgs args;
    args.x = input.contiguous();
    args.dy = grad_out.to(dtype).contiguous();
    args.rstd = as_dtype(rstd, compute);
    args.channels = static_cast<uint32_t>(normalized_numel(input, nnorm));
    args.outer = static_cast<uint32_t>(input.numel() / args.channels);
    args.rms = true;
    args.stats_accum = true;

    at::Tensor grad_input, grad_weight;
    if (output_mask[0]) {
        NormArgs in = args;
        in.weight = as_dtype(weight, dtype);
        in.outputs = 1;
        in.y = at::empty(input.sizes(), args.x.options());
        dispatch_norm_shader(NormMode::ROW_BACKWARD_INPUT, in, in.outer);
        grad_input = in.y;
    }
    if (want_weight) grad_weight = std::get<0>(row_backward_params(args, weight, at::Tensor(), true, false));
    return {grad_input, grad_weight};
}

} // namespace

std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> torchvulkan::layer_norm_shader(
    const at::Tensor& input,
    size_t normalized_dims,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double eps)
{
    if (!can_use_norm_shader(input)) return std::nullopt;
    if (normalized_numel(input, normalized_dims) == 0) return std::nullopt;

    at::ScalarType dtype = input.scalar_type();
    NormArgs args;
    args.x = input.contiguous();
    args.weight = as_dtype(weight, dtype);
    args.bias = as_dtype(bias, dtype);
    args.channels = static_cast<uint32_t>(normalized_numel(input, normalized_dims));
    args.outer = static_cast<uint32_t>(input.numel() / args.channels);
    args.eps = eps;
    args.y = at::empty(input.sizes(), args.x.options());
    std::vector<int64_t> stat_shape = row_stat_shape(input, normalized_dims);
    args.mean = at::empty(stat_shape, args.x.options());
    args.rstd = at::empty(stat_shape, args.x.options());
    dispatch_norm_shader(NormMode::ROW_FORWARD, args, args.outer);
    return std::make_tuple(args.y, args.mean, args.rstd);
}

std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> torchvulkan::layer_norm_backward_shader(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    size_t normalized_dims,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const at::Tensor& weight,
    const at::Tensor& bias,
    std::array<bool, 3> output_mask)
{
    if (!can_use_norm_shader(input) || grad_out.sizes() != input.sizes()) return std::nullopt;
    if (normalized_numel(input, normalized_dims) == 0) return std::nullopt;

    at::ScalarType dtype = input.scalar_type();
    NormArgs args;
    args.x = input.contiguous();
    args.dy = grad_out.to(dtype).contiguous();
    args.mean = as_dtype(mean, dtype);
    args.rstd = as_dtype(rstd, dtype);
    args.channels = static_cast<uint32_t>(normalized_numel(input, normalized_dims));
    args.outer = static_cast<uint32_t>(input.numel() / args.channels);

    at::Tensor grad_input, grad_weight, grad_bias;
    if (output_mask[0]) {
        NormArgs in = args;
        in.weight = as_dtype(weight, dtype);
        in.outputs = 1;
        in.y = at::empty(input.sizes(), args.x.options());
        dispatch_norm_shader(NormMode::ROW_BACKWARD_INPUT, in, in.outer);
        grad_input = in.y;
    }
    bool want_weight = output_mask[1] && weight.defined();
    bool want_bias = output_mask[2] && bias.defined();
    if (want_weight || want_bias) std::tie(grad_weight, grad_bias) = row_backward_params(args, weight, bias, want_weight, want_bias);
    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> torchvulkan::batch_norm_shader(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    bool training,
    double momentum,
    double eps)
{
    if (!can_use_norm_shader(input) || input.dim() < 2 || !input.is_contiguous()) return std::nullopt;
    if (running_mean.defined() != running_var.defined()) return std::nullopt;
    if (!training && !running_mean.defined()) return std::nullopt;

    at::ScalarType dtype = input.scalar_type();
    int64_t n = input.size(0);
    int64_t c = input.size(1);
    if (c == 0 || n == 0) return std::nullopt;
    int64_t inner = input.numel() / (n * c);
    if (training && n * inner <= 1) return std::nullopt; // the composite raises the error torch does

    NormArgs args;
    args.x = input;
    args.weight = as_dtype(weight, dtype);
    args.bias = as_dtype(bias, dtype);
    args.outer = static_cast<uint32_t>(n);
    args.channels = static_cast<uint32_t>(c);
    args.inner = static_cast<uint32_t>(inner);
    args.eps = eps;
    args.momentum = momentum;
    args.training = training;
    args.update_running = training && running_mean.defined();
    if (running_mean.defined()) {
        args.running_mean = running_mean.scalar_type() == dtype && running_mean.is_contiguous() ? running_mean : as_dtype(running_mean, dtype);
        args.running_var = running_var.scalar_type() == dtype && running_var.is_contiguous() ? running_var : as_dtype(running_var, dtype);
    }
    args.y = at::empty(input.sizes(), input.options());
    if (training) {
        args.mean = at::empty({c}, input.options());
        args.rstd = at::empty({c}, input.options());
    }
    dispatch_norm_shader(NormMode::BATCH_FORWARD, args, args.channels);

    if (args.update_running) {
        if (!args.running_mean.is_same(running_mean)) const_cast<at::Tensor&>(running_mean).copy_(args.running_mean);
        if (!args.running_var.is_same(running_var)) const_cast<at::Tensor&>(running_var).copy_(args.running_var);
    }
    if (!training) {
        at::Tensor empty = at::empty({0}, input.options());
        return std::make_tuple(args.y, empty, empty.clone());
    }
    return std::make_tuple(args.y, args.mean, args.rstd);
}

std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> torchvulkan::batch_norm_backward_shader(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    const at::Tensor& save_mean,
    const at::Tensor& save_invstd,
    bool train,
    double eps,
    std::array<bool, 3> output_mask)
{
    if (!can_use_norm_shader(input) || input.dim() < 2 || !input.is_contiguous() || grad_out.sizes() != input.sizes()) return std::nullopt;
    if (train ? (!save_mean.defined() || !save_invstd.defined()) : (!running_mean.defined() || !running_var.defined())) return std::nullopt;

    at::ScalarType dtype = input.scalar_type();
    int64_t n = input.size(0);
    int64_t c = input.size(1);
    if (c == 0 || n == 0) return std::nullopt;

    NormArgs args;
    args.x = input;
    args.dy = grad_out.to(dtype).contiguous();
    args.weight = as_dtype(weight, dtype);
    args.outer = static_cast<uint32_t>(n);
    args.channels = static_cast<uint32_t>(c);
    args.inner = static_cast<uint32_t>(input.numel() / (n * c));
    args.eps = eps;
    args.training = train;
    if (train) {
        args.mean = as_dtype(save_mean, dtype);
        args.rstd = as_dtype(save_invstd, dtype);
    } else {
        args.running_mean = as_dtype(running_mean, dtype);
        args.running_var = as_dtype(running_var, dtype);
    }
    args.outputs = (output_mask[0] ? 1u : 0u) | (output_mask[1] ? 2u : 0u) | (output_mask[2] ? 4u : 0u);
    if (output_mask[0]) args.y = at::empty(input.sizes(), input.options());
    if (output_mask[1]) args.grad_weight = at::empty({c}, input.options());
    if (output_mask[2]) args.grad_bias = at::empty({c}, input.options());
    if (args.outputs == 0) return std::make_tuple(at::Tensor(), at::Tensor(), at::Tensor());
    dispatch_norm_shader(NormMode::BATCH_BACKWARD, args, args.channels);

    at::ScalarType param_dtype = weight.defined() ? weight.scalar_type() : dtype;
    at::Tensor grad_weight = output_mask[1] ? args.grad_weight.to(param_dtype) : at::Tensor();
    at::Tensor grad_bias = output_mask[2] ? args.grad_bias.to(param_dtype) : at::Tensor();
    return std::make_tuple(args.y, grad_weight, grad_bias);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("_fused_rms_norm", &fused_rms_norm_vulkan);
    m.impl("_fused_rms_norm_backward", &fused_rms_norm_backward_vulkan);
}
