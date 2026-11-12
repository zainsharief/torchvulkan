#include <torch/extension.h>
#include <algorithm>
#include <tuple>
#include <vector>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "api/ops/helpers.h"

namespace {

enum class CtcMode {
    ALPHA = 0,
    BETA = 1,
    GRAD = 2
};

struct CtcProblem {
    at::Tensor log_probs;
    at::Tensor targets;
    at::Tensor target_offsets;
    at::Tensor input_lengths;
    at::Tensor target_lengths;
    int64_t time;
    int64_t batch;
    int64_t labels;
    int64_t max_target_length;
};

CtcProblem make_problem(const at::Tensor& log_probs, const at::Tensor& targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank)
{
    TORCH_CHECK(log_probs.numel() > 0, "log_probs tensor must not be empty");
    TORCH_CHECK(log_probs.dim() == 3, "Expected 3-dimensional tensor, but got ", log_probs.dim(), "-dimensional tensor for argument #1 'log_probs'");
    TORCH_CHECK(at::isFloatingType(log_probs.scalar_type()) && !is_low_prec(log_probs.scalar_type()),
        "\"ctc_loss_cpu\" not implemented for '", log_probs.scalar_type(), "'"
    );
    TORCH_CHECK(targets.scalar_type() == at::kLong || targets.scalar_type() == at::kInt, "ctc_loss: targets must be int32 or int64");
    TORCH_CHECK(targets.dim() >= 1 && targets.dim() <= 2, "ctc_loss: targets must have 1 or 2 dimensions");

    CtcProblem p;
    p.time = log_probs.size(0);
    p.batch = log_probs.size(1);
    p.labels = log_probs.size(2);
    TORCH_CHECK(0 <= blank && blank < p.labels, "blank must be in label range");
    TORCH_CHECK(static_cast<int64_t>(input_lengths.size()) == p.batch, "input_lengths must be of size batch_size");
    TORCH_CHECK(static_cast<int64_t>(target_lengths.size()) == p.batch, "target_lengths must be of size batch_size");

    at::Tensor flat_targets = targets.to(at::kLong).contiguous();
    std::vector<int64_t> offsets(p.batch);
    p.max_target_length = 0;
    int64_t position = 0;
    for (int64_t b = 0; b < p.batch; b++) {
        TORCH_CHECK(target_lengths[b] >= 0, "Expected target_lengths to have value at least 0, but got value ", target_lengths[b]);
        TORCH_CHECK(input_lengths[b] >= 0, "Expected input_lengths to have value at least 0, but got value ", input_lengths[b]);
        TORCH_CHECK(input_lengths[b] <= p.time, "Expected input_lengths to have value at most ", p.time, ", but got value ", input_lengths[b]);
        offsets[b] = targets.dim() == 1 ? position : b * flat_targets.size(1);
        position += target_lengths[b];
        p.max_target_length = std::max(p.max_target_length, target_lengths[b]);
    }
    if (targets.dim() == 1) {
        TORCH_CHECK(flat_targets.size(0) == position, "Expected tensor to have size ", position, " at dimension 0, but got size ", flat_targets.size(0));
    } else {
        TORCH_CHECK(flat_targets.size(0) == p.batch, "Expected tensor to have size ", p.batch, " at dimension 0, but got size ", flat_targets.size(0));
        TORCH_CHECK(flat_targets.size(1) >= p.max_target_length, "Expected tensor to have size at least ", p.max_target_length, " at dimension 1, but got size ",
            flat_targets.size(1)
        );
    }

    at::ScalarType dtype = (log_probs.scalar_type() == at::kDouble && is_dtype_supported(log_probs, at::kDouble)) ? at::kDouble : at::kFloat;
    at::TensorOptions long_options = log_probs.options().dtype(at::kLong);
    p.log_probs = log_probs.to(dtype).contiguous();
    p.targets = flat_targets.numel() > 0 ? flat_targets : at::zeros({1}, long_options);
    p.target_offsets = at::tensor(offsets, at::kLong).to(log_probs.device());
    p.input_lengths = at::tensor(input_lengths.vec(), at::kLong).to(log_probs.device());
    p.target_lengths = at::tensor(target_lengths.vec(), at::kLong).to(log_probs.device());
    return p;
}

void dispatch_ctc_shader(
    const CtcProblem& p,
    int64_t states,
    const at::Tensor& log_alpha,
    const at::Tensor& log_beta,
    const at::Tensor& nll,
    const at::Tensor& grad_out,
    const at::Tensor& grad,
    int64_t blank,
    bool zero_infinity,
    CtcMode mode)
{
    DeviceContext* device = get_device_context(p.log_probs);
    at::ScalarType dtype = p.log_probs.scalar_type();
    uint32_t workgroupSizeX = std::min<uint32_t>(256u, device->properties.limits.maxComputeWorkGroupInvocations);

    SpecializationBuilder spd{};
    spd.push(static_cast<uint32_t>(mode))
       .push(workgroupSizeX);

    auto address = [](const at::Tensor& t) { return t.defined() ? get_tensor_address(t) : uint64_t(0); };

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(p.log_probs))
       .push(get_tensor_address(p.targets))
       .push(get_tensor_address(p.target_offsets))
       .push(get_tensor_address(p.input_lengths))
       .push(get_tensor_address(p.target_lengths))
       .push(address(log_alpha))
       .push(address(log_beta))
       .push(address(nll))
       .push(address(grad_out))
       .push(address(grad))
       .push(static_cast<uint32_t>(p.time))
       .push(static_cast<uint32_t>(p.batch))
       .push(static_cast<uint32_t>(p.labels))
       .push(static_cast<uint32_t>(states))
       .push(static_cast<uint32_t>(blank))
       .push(static_cast<uint32_t>(zero_infinity));

    std::vector<at::Tensor> reads = {p.log_probs, p.targets, p.target_offsets, p.input_lengths, p.target_lengths};
    std::vector<at::Tensor> writes;
    if (mode == CtcMode::ALPHA) writes = {log_alpha, nll};
    if (mode == CtcMode::BETA) writes = {log_beta};
    if (mode == CtcMode::GRAD) {
        reads.insert(reads.end(), {log_alpha, log_beta, nll, grad_out});
        writes = {grad};
    }
    uint32_t groups = mode == CtcMode::GRAD ? get_group_count(static_cast<uint64_t>(p.time * p.batch * p.labels), workgroupSizeX) : static_cast<uint32_t>(p.batch);

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::CTC, dtype},
        device->default_subgroup_size,
        spd.build(),
        pcs.build(),
        /* read = */ reads,
        /* write = */ writes,
        groups, 1, 1
    );
}

std::tuple<at::Tensor, at::Tensor> ctc_loss_vulkan(
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    at::IntArrayRef input_lengths,
    at::IntArrayRef target_lengths,
    int64_t blank,
    bool /* zero_infinity */)
{
    CtcProblem p = make_problem(log_probs, targets, input_lengths, target_lengths, blank);
    int64_t states = 2 * p.max_target_length + 1;
    at::Tensor log_alpha = at::empty({p.batch, p.time, states}, p.log_probs.options());
    at::Tensor nll = at::empty({p.batch}, p.log_probs.options());
    dispatch_ctc_shader(p, states, log_alpha, at::Tensor(), nll, at::Tensor(), at::Tensor(), blank, false, CtcMode::ALPHA);
    return std::make_tuple(nll.to(log_probs.scalar_type()), log_alpha.to(log_probs.scalar_type()));
}

std::vector<int64_t> lengths_of(const at::Tensor& lengths)
{
    at::Tensor values = lengths.to(at::kCPU, at::kLong).contiguous();
    return std::vector<int64_t>(values.data_ptr<int64_t>(), values.data_ptr<int64_t>() + values.numel());
}

std::tuple<at::Tensor, at::Tensor> ctc_loss_tensor_vulkan(
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    const at::Tensor& input_lengths,
    const at::Tensor& target_lengths,
    int64_t blank,
    bool zero_infinity)
{
    return ctc_loss_vulkan(log_probs, targets, lengths_of(input_lengths), lengths_of(target_lengths), blank, zero_infinity);
}

at::Tensor ctc_loss_backward_vulkan(
    const at::Tensor& grad,
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    at::IntArrayRef input_lengths,
    at::IntArrayRef target_lengths,
    const at::Tensor& neg_log_likelihood,
    const at::Tensor& log_alpha,
    int64_t blank,
    bool zero_infinity)
{
    CtcProblem p = make_problem(log_probs, targets, input_lengths, target_lengths, blank);
    at::ScalarType dtype = p.log_probs.scalar_type();
    at::Tensor alpha = log_alpha.to(dtype).contiguous();
    int64_t states = alpha.size(2);
    at::Tensor beta = at::empty_like(alpha);
    at::Tensor nll = neg_log_likelihood.to(dtype).contiguous();
    at::Tensor grad_out = grad.to(dtype).expand({p.batch}).contiguous();
    at::Tensor result = at::empty({p.time, p.batch, p.labels}, p.log_probs.options());
    dispatch_ctc_shader(p, states, alpha, beta, nll, grad_out, at::Tensor(), blank, zero_infinity, CtcMode::BETA);
    dispatch_ctc_shader(p, states, alpha, beta, nll, grad_out, result, blank, zero_infinity, CtcMode::GRAD);
    return result.to(log_probs.scalar_type());
}

at::Tensor ctc_loss_backward_tensor_vulkan(
    const at::Tensor& grad,
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    const at::Tensor& input_lengths,
    const at::Tensor& target_lengths,
    const at::Tensor& neg_log_likelihood,
    const at::Tensor& log_alpha,
    int64_t blank,
    bool zero_infinity)
{
    return ctc_loss_backward_vulkan(grad, log_probs, targets, lengths_of(input_lengths), lengths_of(target_lengths), neg_log_likelihood, log_alpha, blank, zero_infinity);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("_ctc_loss", &ctc_loss_vulkan);
    m.impl("_ctc_loss.Tensor", &ctc_loss_tensor_vulkan);
    m.impl("_ctc_loss_backward", &ctc_loss_backward_vulkan);
    m.impl("_ctc_loss_backward.Tensor", &ctc_loss_backward_tensor_vulkan);
}
