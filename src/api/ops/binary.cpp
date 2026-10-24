#include <torch/extension.h>
#include <ATen/TensorIterator.h>
#include <c10/core/MemoryFormat.h>
#include <cmath>
#include <functional>
#include <optional>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "api/ops/helpers.h"
#include "api/ops/internal.h"
#include "api/fallback.h"

namespace {

enum class BinaryOp {
    ADD = 0,
    SUB = 1,
    RSUB = 2,
    MUL = 3,
    DIV = 4,
    MAX = 5,
    MIN = 6,
    POW = 7,
    RPOW = 8,
    ATAN2 = 9,
    THRESHOLD_BACKWARD = 10,
    FMAX = 11,
    FMIN = 12,
    FMOD = 13,
    REMAINDER = 14,
    HYPOT = 15,
    XLOGY = 16,
    LOGADDEXP = 17,
    LOGADDEXP2 = 18,
    FLOOR_DIVIDE = 19
};

// a 0-dim CPU tensor (or a wrapped Python number) mixed with vulkan tensors acts as a scalar
bool is_cpu_scalar(const at::Tensor& t)
{
    return t.unsafeGetTensorImpl()->is_wrapped_number() || (t.dim() == 0 && t.device().is_cpu());
}

std::optional<at::Tensor> dispatch_binary_shader(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& scalar,
    const at::Scalar& alpha,
    BinaryOp operation,
    c10::ScalarType promoted_type)
{
    DeviceContext* device = get_device_context(self);
    if (!is_dtype_supported(device, promoted_type)) return std::nullopt;

    bool use_scalar = !other.defined();
    at::Tensor a = self.to(promoted_type);
    at::Tensor b = use_scalar ? at::Tensor() : other.to(a.options());

    at::Tensor undefined_out;
    at::TensorIteratorConfig config;
    config.set_check_mem_overlap(true).add_output(undefined_out).add_input(a);
    if (!use_scalar) config.add_input(b);
    at::TensorIterator iter = config.build();

    at::Tensor out = iter.output();
    uint64_t numel = iter.numel();
    if (numel == 0) return out;

    int32_t out_dims = static_cast<int32_t>(iter.ndim());
    if (out_dims > MAX_DIMS || !fits_32bit_indexing(iter)) return std::nullopt;

    uint32_t vecSize = get_dtype_vec_size(promoted_type);
    uint32_t workgroupSizeX = get_dtype_workgroup_size(promoted_type, vecSize, device);
    uint32_t contiguous = iter.is_contiguous() ? 1 : 0;
    uint32_t has_alpha = (alpha.toDouble() != 1.0) ? 1u : 0u;

    SpecializationBuilder spd{};
    spd.push(static_cast<uint32_t>(operation))
       .push(contiguous)
       .push(use_scalar ? 1u : 0u)
       .push(out_dims)
       .push(workgroupSizeX)
       .push(has_alpha);

    MetadataBuilder metadataBuilder = get_iter_metadata(iter, {1, use_scalar ? -1 : 2, 0});
    Metadata metadata = metadataBuilder.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(a))
       .push(use_scalar ? uint64_t(0) : get_tensor_address(b))
       .push(get_tensor_address(out))
       .push_metadata(contiguous ? nullptr : &metadata)
       .push(numel)
       .push_scalar(alpha, promoted_type)
       .push_scalar(use_scalar ? scalar : at::Scalar(0), promoted_type);

    std::vector<at::Tensor> reads = {a};
    if (!use_scalar) reads.push_back(b);

    uint64_t threads = contiguous ? (numel + (vecSize - 1)) / vecSize : numel;
    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::BINARY, promoted_type},
        spd.build(),
        pcs.build(),
        /* read = */ reads,
        /* write = */ {out},
        get_group_count(threads, workgroupSizeX), 1, 1
    );

    return out;
}

at::Tensor binary_op_vulkan(
    const at::Tensor& self, 
    const at::Scalar& other, 
    const at::Scalar& alpha, 
    BinaryOp operation, 
    const std::function<at::Tensor(const at::Tensor&, const at::Scalar&)>& fallback)
{
    std::optional<at::Tensor> out = dispatch_binary_shader(self, at::Tensor(), other, alpha, operation, at::result_type(self, other));
    if (out.has_value()) return *out;

    TORCHVULKAN_CPU_FALLBACK("A binary op on ", self.scalar_type(), " can't run on the Vulkan device. Falling back to CPU.");
    return fallback(self.cpu(), other).to(self.device());
}

at::Tensor binary_op_vulkan(
    const at::Tensor& self, 
    const at::Tensor& other, 
    const at::Scalar& alpha, 
    BinaryOp operation, 
    const std::function<at::Tensor(const at::Tensor&, const at::Tensor&)>& fallback)
{    
    if (is_cpu_scalar(other) && self.device().type() == c10::DeviceType::PrivateUse1) 
    {
        return binary_op_vulkan(
            self, other.item(), alpha, operation,
            [&fallback, &other](const at::Tensor& a, const at::Scalar&) { return fallback(a, other); }
        );
    }

    at::Tensor lhs = is_cpu_scalar(self) ? self.to(other.device()) : self;
    at::Tensor rhs = other.device() == lhs.device() ? other : other.to(lhs.device());

    std::optional<at::Tensor> out = dispatch_binary_shader(lhs, rhs, at::Scalar(0), alpha, operation, at::result_type(self, other));
    if (out.has_value()) return *out;

    TORCHVULKAN_CPU_FALLBACK("A binary op on ", at::result_type(self, other), " can't run on the Vulkan device. Falling back to CPU.");
    return fallback(lhs.cpu(), rhs.cpu()).to(lhs.device());
}

} // namespace

at::Tensor torchvulkan::add_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) 
{    
    if (self.scalar_type() == at::kBool) {
        if (alpha.to<bool>() == false) return self.clone();
        return binary_op_vulkan(self, other, alpha, BinaryOp::MAX, [alpha](const at::Tensor& a, const at::Tensor& b) { return at::add(a, b, alpha); });
    }
    return binary_op_vulkan(self, other, alpha, BinaryOp::ADD, [alpha](const at::Tensor& a, const at::Tensor& b) { return at::add(a, b, alpha); });
}

at::Tensor torchvulkan::multiply_vulkan(const at::Tensor& self, const at::Tensor& other) {
    return binary_op_vulkan(self, other, (int)1, BinaryOp::MUL, [](const at::Tensor& a, const at::Tensor& b) { return at::mul(a, b); });
}

at::Tensor torchvulkan::divide_vulkan(const at::Tensor& self, const at::Tensor& other) {
    at::Tensor self_vulkan = self;
    at::Tensor other_vulkan = other;

    if (!self.is_floating_point() && !other.is_floating_point() && !self.is_complex() && !other.is_complex()) {
        self_vulkan = self.to(c10::kFloat);
        other_vulkan = other.to(c10::kFloat);
    }

    return binary_op_vulkan(self_vulkan, other_vulkan, (int)1, BinaryOp::DIV, [](const at::Tensor& a, const at::Tensor& b) { return at::div(a, b); });
}

at::Tensor torchvulkan::divide_scalar_vulkan(const at::Tensor& self, const at::Scalar& other) {
    at::Tensor self_vulkan = self;

    if (at::isIntegralType(self.scalar_type(), /* includeBool = */ true)) self_vulkan = self.to(c10::kFloat);

    return binary_op_vulkan(self_vulkan, other, (int)1, BinaryOp::DIV, [](const at::Tensor& a, const at::Scalar& b) { return at::div(a, b); });
}

namespace {

at::Tensor add_scalar_vulkan(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) 
{
    if (self.scalar_type() == at::kBool) {
        if (alpha.to<bool>() == false) return self.clone();
        return binary_op_vulkan(self, other, alpha, BinaryOp::MAX, [alpha](const at::Tensor& a, const at::Scalar& b) { return at::add(a, b, alpha); });
    }
    return binary_op_vulkan(self, other, alpha, BinaryOp::ADD, [alpha](const at::Tensor& a, const at::Scalar& b) { return at::add(a, b, alpha); });
}

at::Tensor subtract_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    return binary_op_vulkan(self, other, alpha, BinaryOp::SUB, [alpha](const at::Tensor& a, const at::Tensor& b) { return at::sub(a, b, alpha); });
}

at::Tensor subtract_scalar_vulkan(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
    return binary_op_vulkan(self, other, alpha, BinaryOp::SUB, [alpha](const at::Tensor& a, const at::Scalar& b) { return at::sub(a, b, alpha); });
}

at::Tensor rsub_scalar_vulkan(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
    return binary_op_vulkan(self, other, alpha, BinaryOp::RSUB, [alpha](const at::Tensor& a, const at::Scalar& b) { return at::rsub(a, b, alpha); });
}

at::Tensor multiply_scalar_vulkan(const at::Tensor& self, const at::Scalar& other) {
    return binary_op_vulkan(self, other, (int)1, BinaryOp::MUL, [](const at::Tensor& a, const at::Scalar& b) { return at::mul(a, b); });
}

at::Tensor maximum_vulkan(const at::Tensor& self, const at::Tensor& other) {
    return binary_op_vulkan(self, other, (int)1, BinaryOp::MAX, [](const at::Tensor& a, const at::Tensor& b) { return at::max(a, b); });
}

at::Tensor minimum_vulkan(const at::Tensor& self, const at::Tensor& other) {
    return binary_op_vulkan(self, other, (int)1, BinaryOp::MIN, [](const at::Tensor& a, const at::Tensor& b) { return at::min(a, b); });
}

at::Tensor pow_vulkan(const at::Tensor& self, const at::Tensor& other) {
    return binary_op_vulkan(self, other, (int)1, BinaryOp::POW, [](const at::Tensor& a, const at::Tensor& b) { return at::pow(a, b); });
}

at::Tensor pow_tensor_scalar_vulkan(const at::Tensor& self, const at::Scalar& other) {
    return binary_op_vulkan(self, other, (int)1, BinaryOp::POW, [](const at::Tensor& a, const at::Scalar& b) { return at::pow(a, b); });
}

at::Tensor pow_scalar_vulkan(const at::Scalar& self, const at::Tensor& other) {
    return binary_op_vulkan(other, self, (int)1, BinaryOp::RPOW, [](const at::Tensor& a, const at::Scalar& b) { return at::pow(b, a); });
}

at::Tensor atan2_vulkan(const at::Tensor& self, const at::Tensor& other) {
    at::Tensor self_vulkan = self;
    at::Tensor other_vulkan = other;

    if (!self.is_floating_point() && !other.is_floating_point() && !self.is_complex() && !other.is_complex()) {
        self_vulkan = self.to(c10::kFloat);
        other_vulkan = other.to(c10::kFloat);
    }

    return binary_op_vulkan(self_vulkan, other_vulkan, (int)1, BinaryOp::ATAN2, [](const at::Tensor& a, const at::Tensor& b) { return at::atan2(a, b); });
}

at::Tensor threshold_backward_vulkan(const at::Tensor& grad_output, const at::Tensor& self, const at::Scalar& threshold) {
    return binary_op_vulkan(grad_output, self, threshold, BinaryOp::THRESHOLD_BACKWARD, [threshold](const at::Tensor& a, const at::Tensor& b) { return at::threshold_backward(a, b, threshold); });
}

at::Tensor fmax_vulkan(const at::Tensor& self, const at::Tensor& other) {
    return binary_op_vulkan(self, other, (int)1, BinaryOp::FMAX, [](const at::Tensor& a, const at::Tensor& b) { return at::fmax(a, b); });
}

at::Tensor fmin_vulkan(const at::Tensor& self, const at::Tensor& other) {
    return binary_op_vulkan(self, other, (int)1, BinaryOp::FMIN, [](const at::Tensor& a, const at::Tensor& b) { return at::fmin(a, b); });
}

at::Tensor fmod_vulkan(const at::Tensor& self, const at::Tensor& other) {
    return binary_op_vulkan(self, other, (int)1, BinaryOp::FMOD, [](const at::Tensor& a, const at::Tensor& b) { return at::fmod(a, b); });
}

at::Tensor fmod_scalar_vulkan(const at::Tensor& self, const at::Scalar& other) {
    return binary_op_vulkan(self, other, (int)1, BinaryOp::FMOD, [](const at::Tensor& a, const at::Scalar& b) { return at::fmod(a, b); });
}

at::Tensor remainder_vulkan(const at::Tensor& self, const at::Tensor& other) {
    return binary_op_vulkan(self, other, (int)1, BinaryOp::REMAINDER, [](const at::Tensor& a, const at::Tensor& b) { return at::remainder(a, b); });
}

at::Tensor remainder_scalar_vulkan(const at::Tensor& self, const at::Scalar& other) {
    return binary_op_vulkan(self, other, (int)1, BinaryOp::REMAINDER, [](const at::Tensor& a, const at::Scalar& b) { return at::remainder(a, b); });
}

at::Tensor hypot_vulkan(const at::Tensor& self, const at::Tensor& other) {
    at::Tensor a = self, b = other;
    if (!self.is_floating_point() && !other.is_floating_point()) {
        a = self.to(c10::kFloat);
        b = other.to(c10::kFloat);
    }
    return binary_op_vulkan(a, b, (int)1, BinaryOp::HYPOT, [](const at::Tensor& x, const at::Tensor& y) { return at::hypot(x, y); });
}

at::Tensor xlogy_vulkan(const at::Tensor& self, const at::Tensor& other) {
    at::Tensor a = self, b = other;
    if (!self.is_floating_point() && !other.is_floating_point()) {
        a = self.to(c10::kFloat);
        b = other.to(c10::kFloat);
    }
    return binary_op_vulkan(a, b, (int)1, BinaryOp::XLOGY, [](const at::Tensor& x, const at::Tensor& y) { return at::xlogy(x, y); });
}

at::Tensor logaddexp_vulkan(const at::Tensor& self, const at::Tensor& other) {
    return binary_op_vulkan(self, other, (int)1, BinaryOp::LOGADDEXP, [](const at::Tensor& a, const at::Tensor& b) { return at::logaddexp(a, b); });
}

at::Tensor logaddexp2_vulkan(const at::Tensor& self, const at::Tensor& other) {
    return binary_op_vulkan(self, other, (int)1, BinaryOp::LOGADDEXP2, [](const at::Tensor& a, const at::Tensor& b) { return at::logaddexp2(a, b); });
}

at::Tensor clamp_min_scalar_vulkan(const at::Tensor& self, const at::Scalar& min) {
    at::Tensor r = binary_op_vulkan(self, min, (int)1, BinaryOp::MAX, [](const at::Tensor& a, const at::Scalar& b) { return at::clamp_min(a, b); });
    return r.scalar_type() == self.scalar_type() ? r : r.to(self.scalar_type());
}

at::Tensor clamp_max_scalar_vulkan(const at::Tensor& self, const at::Scalar& max) {
    at::Tensor r = binary_op_vulkan(self, max, (int)1, BinaryOp::MIN, [](const at::Tensor& a, const at::Scalar& b) { return at::clamp_max(a, b); });
    return r.scalar_type() == self.scalar_type() ? r : r.to(self.scalar_type());
}

at::Tensor clamp_min_tensor_vulkan(const at::Tensor& self, const at::Tensor& min) {
    return maximum_vulkan(self, min);
}

at::Tensor clamp_max_tensor_vulkan(const at::Tensor& self, const at::Tensor& max) {
    return minimum_vulkan(self, max);
}

at::Tensor clamp_scalar_vulkan(const at::Tensor& self, const c10::optional<at::Scalar>& min, const c10::optional<at::Scalar>& max) {
    TORCH_CHECK(min.has_value() || max.has_value(), "torch.clamp: At least one of 'min' or 'max' must not be None");
    at::Tensor result = self;
    if (min.has_value()) result = clamp_min_scalar_vulkan(result, *min);
    if (max.has_value()) result = clamp_max_scalar_vulkan(result, *max);
    return result;
}

at::Tensor clamp_tensor_vulkan(const at::Tensor& self, const c10::optional<at::Tensor>& min, const c10::optional<at::Tensor>& max) {
    bool has_min = min.has_value() && min->defined();
    bool has_max = max.has_value() && max->defined();
    TORCH_CHECK(has_min || has_max, "torch.clamp: At least one of 'min' or 'max' must not be None");
    at::Tensor result = self;
    if (has_min) result = maximum_vulkan(result, *min);
    if (has_max) result = minimum_vulkan(result, *max);
    return result;
}

at::Tensor& clamp_min_out_vulkan(const at::Tensor& self, const at::Scalar& min, at::Tensor& out) {
    return fill_out(out, clamp_min_scalar_vulkan(self, min));
}

at::Tensor& clamp_max_out_vulkan(const at::Tensor& self, const at::Scalar& max, at::Tensor& out) {
    return fill_out(out, clamp_max_scalar_vulkan(self, max));
}

at::Tensor& clamp_min_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& min, at::Tensor& out) {
    return fill_out(out, clamp_min_tensor_vulkan(self, min));
}

at::Tensor& clamp_max_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& max, at::Tensor& out) {
    return fill_out(out, clamp_max_tensor_vulkan(self, max));
}

at::Tensor& clamp_out_vulkan(const at::Tensor& self, const c10::optional<at::Scalar>& min, const c10::optional<at::Scalar>& max, at::Tensor& out) {
    return fill_out(out, clamp_scalar_vulkan(self, min, max));
}

at::Tensor& clamp_tensor_out_vulkan(const at::Tensor& self, const c10::optional<at::Tensor>& min, const c10::optional<at::Tensor>& max, at::Tensor& out) {
    return fill_out(out, clamp_tensor_vulkan(self, min, max));
}

at::Tensor& add_out_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha, at::Tensor& out) {
    return fill_out(out, torchvulkan::add_vulkan(self, other, alpha));
}

at::Tensor& subtract_out_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha, at::Tensor& out) {
    return fill_out(out, subtract_vulkan(self, other, alpha));
}

at::Tensor& multiply_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    return fill_out(out, torchvulkan::multiply_vulkan(self, other));
}

at::Tensor& divide_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    return fill_out(out, torchvulkan::divide_vulkan(self, other));
}

at::Tensor& atan2_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    return fill_out(out, atan2_vulkan(self, other));
}

at::Tensor& fmax_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    return fill_out(out, fmax_vulkan(self, other));
}

at::Tensor& fmin_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    return fill_out(out, fmin_vulkan(self, other));
}

at::Tensor& fmod_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    return fill_out(out, fmod_vulkan(self, other));
}

at::Tensor& hypot_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    return fill_out(out, hypot_vulkan(self, other));
}

at::Tensor& logaddexp_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    return fill_out(out, logaddexp_vulkan(self, other));
}

at::Tensor& logaddexp2_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    return fill_out(out, logaddexp2_vulkan(self, other));
}

at::Tensor& maximum_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    return fill_out(out, maximum_vulkan(self, other));
}

at::Tensor& minimum_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    return fill_out(out, minimum_vulkan(self, other));
}

at::Tensor& remainder_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    return fill_out(out, remainder_vulkan(self, other));
}

at::Tensor remainder_scalar_tensor_vulkan(const at::Scalar& self, const at::Tensor& other) {
    at::Tensor wrapped = at::scalar_tensor(self, other.options().dtype(at::result_type(other, self)));
    wrapped.unsafeGetTensorImpl()->set_wrapped_number(true);
    return remainder_vulkan(wrapped, other);
}

at::Tensor& xlogy_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    return fill_out(out, xlogy_vulkan(self, other));
}

at::Tensor& threshold_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& self, const at::Scalar& threshold, at::Tensor& grad_input) {
    return fill_out(grad_input, threshold_backward_vulkan(grad_output, self, threshold));
}

at::Tensor add_relu_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    return at::relu(torchvulkan::add_vulkan(self, other, alpha));
}

at::Tensor add_relu_scalar_vulkan(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
    return at::relu(add_scalar_vulkan(self, other, alpha));
}

at::Tensor& add_relu_out_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha, at::Tensor& out) {
    return fill_out(out, add_relu_vulkan(self, other, alpha));
}

at::Tensor& add_relu_vulkan_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    return copy_inplace_result(self, add_relu_vulkan(self, other, alpha));
}

at::Tensor& add_relu_scalar_vulkan_(at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
    return copy_inplace_result(self, add_relu_scalar_vulkan(self, other, alpha));
}

at::Tensor& add_vulkan_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    return copy_inplace_result(self, torchvulkan::add_vulkan(self, other, alpha));
}

at::Tensor& add_scalar_vulkan_(at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
    return copy_inplace_result(self, add_scalar_vulkan(self, other, alpha));
}

at::Tensor& subtract_vulkan_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    return copy_inplace_result(self, subtract_vulkan(self, other, alpha));
}

at::Tensor& subtract_scalar_vulkan_(at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
    return copy_inplace_result(self, subtract_scalar_vulkan(self, other, alpha));
}

at::Tensor& multiply_vulkan_(at::Tensor& self, const at::Tensor& other) {
    return copy_inplace_result(self, torchvulkan::multiply_vulkan(self, other));
}

at::Tensor& multiply_scalar_vulkan_(at::Tensor& self, const at::Scalar& other) {
    return copy_inplace_result(self, multiply_scalar_vulkan(self, other));
}

at::Tensor& divide_vulkan_(at::Tensor& self, const at::Tensor& other) {
    return copy_inplace_result(self, torchvulkan::divide_vulkan(self, other));
}

at::Tensor& divide_scalar_vulkan_(at::Tensor& self, const at::Scalar& other) {
    return copy_inplace_result(self, torchvulkan::divide_scalar_vulkan(self, other));
}

at::Tensor& addcmul_vulkan_(at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2, const at::Scalar& value) {
    at::Tensor product = torchvulkan::multiply_vulkan(tensor1, tensor2);
    return copy_inplace_result(self, torchvulkan::add_vulkan(self, product, value));
}

at::Tensor& addcdiv_vulkan_(at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2, const at::Scalar& value) {
    TORCH_CHECK(!(at::isIntegralType(tensor1.scalar_type(), true) && at::isIntegralType(tensor2.scalar_type(), true)),
        "Integer division with addcdiv is no longer supported, and in a future release addcdiv will perform a true division of tensor1 and tensor2."
    );
    at::Tensor quotient = torchvulkan::divide_vulkan(tensor1, tensor2);
    return copy_inplace_result(self, torchvulkan::add_vulkan(self, quotient, value));
}

// as torch computes it: from whichever end is nearer, so weight 1 gives exactly `end`
at::Tensor lerp_scalar_vulkan(const at::Tensor& self, const at::Tensor& end, const at::Scalar& weight) {
    at::Tensor diff = subtract_vulkan(end, self, (int)1);
    double w = weight.toDouble();
    if (w < 0.5) return torchvulkan::add_vulkan(self, diff, weight);
    return subtract_vulkan(end, diff, 1.0 - w);
}

at::Tensor& lerp_scalar_vulkan_(at::Tensor& self, const at::Tensor& end, const at::Scalar& weight) {
    return copy_inplace_result(self, lerp_scalar_vulkan(self, end, weight));
}

at::Tensor& lerp_scalar_vulkan_out(const at::Tensor& self, const at::Tensor& end, const at::Scalar& weight, at::Tensor& out) {
    return fill_out(out, lerp_scalar_vulkan(self, end, weight));
}

at::Tensor floor_divide_vulkan(const at::Tensor& self, const at::Tensor& other) {
    // an empty result never divides, so never raises
    bool empty = self.numel() == 0 || other.numel() == 0;
    if (!empty && at::isIntegralType(at::result_type(self, other), /*includeBool=*/true)) {
        TORCH_CHECK(!at::eq(other, 0).any().item<bool>(), "ZeroDivisionError");
    }
    return binary_op_vulkan(self, other, (int)1, BinaryOp::FLOOR_DIVIDE, [](const at::Tensor& a, const at::Tensor& b) { return at::floor_divide(a, b); });
}

at::Tensor& floor_divide_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    return fill_out(out, floor_divide_vulkan(self, other));
}

at::Tensor& floor_divide_vulkan_(at::Tensor& self, const at::Tensor& other) {
    return copy_inplace_result(self, floor_divide_vulkan(self, other));
}

at::Tensor rsub_tensor_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    return binary_op_vulkan(self, other, alpha, BinaryOp::RSUB, [alpha](const at::Tensor& a, const at::Tensor& b) { return at::rsub(a, b, alpha); });
}

// the CPU's two-sided formula, in float32 for half and bfloat16
at::Tensor lerp_tensor_vulkan(const at::Tensor& self, const at::Tensor& end, const at::Tensor& weight) {
    at::ScalarType dtype = self.scalar_type();
    TORCH_CHECK(dtype == end.scalar_type(), "expected dtype ", dtype, " for `end` but got dtype ", end.scalar_type());
    TORCH_CHECK(dtype == weight.scalar_type(), "expected dtype ", dtype, " for `weight` but got dtype ", weight.scalar_type());
    bool low = is_low_prec(dtype);
    at::Tensor s = low ? self.to(at::kFloat) : self;
    at::Tensor e = low ? end.to(at::kFloat) : end;
    at::Tensor w = weight.to(self.device(), s.scalar_type());
    at::Tensor diff = subtract_vulkan(e, s, (int)1);
    at::Tensor near_self = torchvulkan::add_vulkan(s, torchvulkan::multiply_vulkan(w, diff), (int)1);
    at::Tensor near_end = subtract_vulkan(e, torchvulkan::multiply_vulkan(diff, rsub_scalar_vulkan(w, 1, 1)), (int)1);
    at::Tensor r = at::where(at::lt(at::abs(w), 0.5), near_self, near_end);
    return low ? r.to(dtype) : r;
}

at::Tensor& lerp_tensor_vulkan_out(const at::Tensor& self, const at::Tensor& end, const at::Tensor& weight, at::Tensor& out) {
    return fill_out(out, lerp_tensor_vulkan(self, end, weight));
}

at::Tensor& pow_tensor_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& exponent, at::Tensor& out) {
    return fill_out(out, pow_vulkan(self, exponent));
}

at::Tensor& pow_tensor_scalar_out_vulkan(const at::Tensor& self, const at::Scalar& exponent, at::Tensor& out) {
    return fill_out(out, pow_tensor_scalar_vulkan(self, exponent));
}

at::Tensor& pow_scalar_out_vulkan(const at::Scalar& self, const at::Tensor& exponent, at::Tensor& out) {
    return fill_out(out, pow_scalar_vulkan(self, exponent));
}

at::ScalarType bits_dtype(at::ScalarType dtype) {
    switch (c10::elementSize(dtype)) {
        case 2: return at::kShort;
        case 4: return at::kInt;
        default: return at::kLong;
    }
}

// sign bits read through an integer view, so -0.0 and negative NaNs count as negative
at::Tensor signbit_vulkan(const at::Tensor& self) {
    if (at::isFloatingType(self.scalar_type())) return at::lt(self.contiguous().view(bits_dtype(self.scalar_type())), 0);
    if (at::isSignedType(self.scalar_type())) return at::lt(self, 0);
    return at::zeros(self.sizes(), self.options().dtype(at::kBool));
}

at::Tensor& signbit_out_vulkan(const at::Tensor& self, at::Tensor& out) {
    return fill_out(out, signbit_vulkan(self));
}

at::Tensor& copysign_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    at::ScalarType dtype = at::result_type(self, other);
    if (!at::isFloatingType(dtype)) dtype = c10::typeMetaToScalarType(at::get_default_dtype());
    at::Tensor magnitude = at::abs(self.to(dtype));
    at::Tensor sign_source = other.to(self.device(), dtype);
    if (sign_source.dim() == 0 && magnitude.dim() > 0) sign_source = sign_source.expand(magnitude.sizes());
    at::Tensor negative = signbit_vulkan(sign_source);
    return fill_out(out, at::where(negative, at::neg(magnitude), magnitude));
}

// from zero, the first step is the smallest subnormal with b's sign
at::Tensor& nextafter_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    at::ScalarType dtype = at::result_type(self, other);
    TORCH_CHECK_NOT_IMPLEMENTED(at::isFloatingType(dtype), "\"nextafter_cpu\" not implemented for '", dtype, "'");
    std::vector<int64_t> shape = at::infer_size(self.sizes(), other.sizes());
    at::Tensor a = self.to(dtype).expand(shape).contiguous();
    at::Tensor b = other.to(self.device(), dtype).expand(shape).contiguous();
    at::ScalarType bits = bits_dtype(dtype);

    at::Tensor a_bits = a.view(bits);
    at::Tensor toward_zero = at::ne(at::lt(a, b), at::gt(a, 0));
    at::Tensor stepped = at::where(toward_zero, at::sub(a_bits, 1), at::add(a_bits, 1)).view(dtype);
    int64_t sign_bit = std::numeric_limits<int64_t>::min() >> (64 - 8 * c10::elementSize(dtype));
    at::Tensor smallest = at::where(signbit_vulkan(b), at::full_like(a_bits, sign_bit | 1), at::full_like(a_bits, 1)).view(dtype);

    at::Tensor r = at::where(at::eq(a, 0), smallest, stepped);
    r = at::where(at::eq(a, b), b, r);
    r = at::where(at::logical_or(at::isnan(a), at::isnan(b)), at::add(a, b), r);
    return fill_out(out, r);
}

// as torch: round(x * 10^d) / 10^d, or round(x / 10^-d) * 10^-d for negative d
at::Tensor& round_decimals_out_vulkan(const at::Tensor& self, int64_t decimals, at::Tensor& out) {
    if (decimals == 0) return fill_out(out, at::round(self));
    TORCH_CHECK_NOT_IMPLEMENTED(at::isFloatingType(self.scalar_type()), "\"round_cpu\" not implemented for '", self.scalar_type(), "'");
    double ten_pow = std::pow(10.0, static_cast<double>(std::abs(decimals)));
    if (decimals > 0) return fill_out(out, at::div(at::round(at::mul(self, ten_pow)), ten_pow));
    return fill_out(out, at::mul(at::round(at::div(self, ten_pow)), ten_pow));
}

// works on the bit pattern alone: Metal flushes subnormals to zero in float arithmetic and comparisons
std::tuple<at::Tensor&, at::Tensor&> frexp_out_vulkan(const at::Tensor& self, at::Tensor& mantissa, at::Tensor& exponent) {
    at::ScalarType dtype = self.scalar_type();
    TORCH_CHECK(at::isFloatingType(dtype), "torch.frexp() only supports floating-point dtypes");

    int64_t mantissa_bits = 0, exponent_mask = 0, bias = 0;
    switch (dtype) {
        case at::kHalf: mantissa_bits = 10; exponent_mask = 0x1f; bias = 15; break;
        case at::kBFloat16: mantissa_bits = 7; exponent_mask = 0xff; bias = 127; break;
        case at::kFloat: mantissa_bits = 23; exponent_mask = 0xff; bias = 127; break;
        default: mantissa_bits = 52; exponent_mask = 0x7ff; bias = 1023; break;
    }
    int64_t mantissa_mask = (int64_t(1) << mantissa_bits) - 1;
    int64_t half_exponent = (bias - 1) << mantissa_bits;

    at::Tensor bits = self.contiguous().view(bits_dtype(dtype)).to(at::kLong);
    at::Tensor field = at::bitwise_and(at::bitwise_right_shift(bits, mantissa_bits), exponent_mask);
    at::Tensor fraction = at::bitwise_and(bits, mantissa_mask);
    at::Tensor sign = at::bitwise_and(bits, ~(exponent_mask << mantissa_bits | mantissa_mask));

    // a subnormal's leading bit, from the exponent of float(fraction); rounding up can overshoot by one
    at::Tensor lead = at::sub(at::bitwise_and(at::bitwise_right_shift(fraction.to(at::kFloat).view(at::kInt).to(at::kLong), 23), 0xff), 127);
    lead = at::where(at::gt(at::bitwise_left_shift(at::ones_like(lead), lead), fraction), at::sub(lead, 1), lead);

    at::Tensor normal_exponent = at::sub(field, bias - 1);
    at::Tensor normal_bits = at::bitwise_or(at::bitwise_or(sign, half_exponent), fraction);
    at::Tensor subnormal_exponent = at::add(lead, 2 - bias - mantissa_bits);
    at::Tensor subnormal_fraction = at::bitwise_and(at::bitwise_left_shift(fraction, at::rsub(lead, mantissa_bits)), mantissa_mask);
    at::Tensor subnormal_bits = at::bitwise_or(at::bitwise_or(sign, half_exponent), subnormal_fraction);

    at::Tensor is_subnormal = at::logical_and(at::eq(field, 0), at::ne(fraction, 0));
    at::Tensor is_special = at::logical_or(at::logical_and(at::eq(field, 0), at::eq(fraction, 0)), at::eq(field, exponent_mask));

    at::Tensor m_bits = at::where(is_subnormal, subnormal_bits, normal_bits);
    m_bits = at::where(is_special, bits, m_bits);
    at::Tensor e = at::where(is_subnormal, subnormal_exponent, normal_exponent);
    e = at::where(is_special, at::zeros_like(e), e);

    fill_out(mantissa, m_bits.to(bits_dtype(dtype)).view(dtype).view(self.sizes()));
    fill_out(exponent, e.to(at::kInt).view(self.sizes()));
    return std::forward_as_tuple(mantissa, exponent);
}

at::Tensor div_mode_vulkan(const at::Tensor& self, const at::Tensor& other, c10::optional<c10::string_view> rounding_mode)
{
    if (!rounding_mode.has_value()) return torchvulkan::divide_vulkan(self, other);
    TORCH_CHECK(*rounding_mode == "trunc" || *rounding_mode == "floor",
        "div expected rounding_mode to be one of None, 'trunc', or 'floor' but found '", *rounding_mode, "'"
    );
    if (*rounding_mode == "floor") return floor_divide_vulkan(self, other);
    at::ScalarType dtype = at::result_type(self, other);
    if (is_low_prec(dtype)) {
        // the float32 quotient is rounded to the dtype, then truncated, as the CPU does
        return at::trunc(torchvulkan::divide_vulkan(self.to(at::kFloat), other.to(at::kFloat)).to(dtype));
    }
    if (!at::isIntegralType(dtype, /*includeBool=*/true)) return at::trunc(torchvulkan::divide_vulkan(self, other));
    at::Tensor floored = floor_divide_vulkan(self, other);
    at::Tensor inexact = at::ne(fmod_vulkan(self, other), 0);
    at::Tensor signs_differ = at::ne(at::lt(self, 0), at::lt(other, 0));
    return at::where(at::logical_and(inexact, signs_differ), at::add(floored, 1), floored);
}

at::Tensor& div_out_mode_vulkan(const at::Tensor& self, const at::Tensor& other, c10::optional<c10::string_view> rounding_mode, at::Tensor& out)
{
    return fill_out(out, div_mode_vulkan(self, other, rounding_mode));
}

at::Tensor& addcmul_out_vulkan(const at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2, const at::Scalar& value, at::Tensor& out)
{
    return fill_out(out, torchvulkan::add_vulkan(self, torchvulkan::multiply_vulkan(tensor1, tensor2), value));
}

at::Tensor& addcdiv_out_vulkan(const at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2, const at::Scalar& value, at::Tensor& out)
{
    TORCH_CHECK(!(at::isIntegralType(tensor1.scalar_type(), true) && at::isIntegralType(tensor2.scalar_type(), true)),
        "Integer division with addcdiv is no longer supported, and in a future release addcdiv will perform a true division of tensor1 and tensor2."
    );
    return fill_out(out, torchvulkan::add_vulkan(self, torchvulkan::divide_vulkan(tensor1, tensor2), value));
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("add.Tensor", &torchvulkan::add_vulkan);
    m.impl("add.Scalar", &add_scalar_vulkan);
    m.impl("add_.Tensor", &add_vulkan_);
    m.impl("add_.Scalar", &add_scalar_vulkan_);
    m.impl("add.out", &add_out_vulkan);
    m.impl("sub.out", &subtract_out_vulkan);
    m.impl("mul.out", &multiply_out_vulkan);
    m.impl("div.out", &divide_out_vulkan);
    m.impl("atan2.out", &atan2_out_vulkan);
    m.impl("fmax.out", &fmax_out_vulkan);
    m.impl("fmin.out", &fmin_out_vulkan);
    m.impl("fmod.Tensor_out", &fmod_out_vulkan);
    m.impl("hypot.out", &hypot_out_vulkan);
    m.impl("logaddexp.out", &logaddexp_out_vulkan);
    m.impl("logaddexp2.out", &logaddexp2_out_vulkan);
    m.impl("maximum.out", &maximum_out_vulkan);
    m.impl("minimum.out", &minimum_out_vulkan);
    m.impl("remainder.Tensor_out", &remainder_out_vulkan);
    m.impl("remainder.Scalar_Tensor", &remainder_scalar_tensor_vulkan);
    m.impl("xlogy.OutTensor", &xlogy_out_vulkan);
    m.impl("threshold_backward.grad_input", &threshold_backward_out_vulkan);
    m.impl("_add_relu.Tensor", &add_relu_vulkan);
    m.impl("_add_relu.Scalar", &add_relu_scalar_vulkan);
    m.impl("_add_relu.out", &add_relu_out_vulkan);
    m.impl("_add_relu_.Tensor", &add_relu_vulkan_);
    m.impl("_add_relu_.Scalar", &add_relu_scalar_vulkan_);
    m.impl("sub.Tensor", &subtract_vulkan);
    m.impl("sub.Scalar", &subtract_scalar_vulkan);
    m.impl("sub_.Tensor", &subtract_vulkan_);
    m.impl("sub_.Scalar", &subtract_scalar_vulkan_);
    m.impl("rsub.Scalar", &rsub_scalar_vulkan);
    m.impl("mul.Tensor", &torchvulkan::multiply_vulkan);
    m.impl("mul.Scalar", &multiply_scalar_vulkan);
    m.impl("mul_.Tensor", &multiply_vulkan_);
    m.impl("mul_.Scalar", &multiply_scalar_vulkan_);
    m.impl("div.Tensor", &torchvulkan::divide_vulkan);
    m.impl("div.Scalar", &torchvulkan::divide_scalar_vulkan);
    m.impl("div_.Tensor", &divide_vulkan_);
    m.impl("div_.Scalar", &divide_scalar_vulkan_);
    m.impl("maximum", &maximum_vulkan);
    m.impl("minimum", &minimum_vulkan);
    m.impl("pow.Tensor_Tensor", &pow_vulkan);
    m.impl("pow.Tensor_Scalar", &pow_tensor_scalar_vulkan);
    m.impl("pow.Scalar", &pow_scalar_vulkan);
    m.impl("atan2", &atan2_vulkan);
    m.impl("fmax", &fmax_vulkan);
    m.impl("fmin", &fmin_vulkan);
    m.impl("fmod.Tensor", &fmod_vulkan);
    m.impl("fmod.Scalar", &fmod_scalar_vulkan);
    m.impl("remainder.Tensor", &remainder_vulkan);
    m.impl("remainder.Scalar", &remainder_scalar_vulkan);
    m.impl("hypot", &hypot_vulkan);
    m.impl("xlogy.Tensor", &xlogy_vulkan);
    m.impl("logaddexp", &logaddexp_vulkan);
    m.impl("logaddexp2", &logaddexp2_vulkan);
    m.impl("clamp", &clamp_scalar_vulkan);
    m.impl("clamp.Tensor", &clamp_tensor_vulkan);
    m.impl("clamp_min", &clamp_min_scalar_vulkan);
    m.impl("clamp_min.Tensor", &clamp_min_tensor_vulkan);
    m.impl("clamp_max", &clamp_max_scalar_vulkan);
    m.impl("clamp_max.Tensor", &clamp_max_tensor_vulkan);
    m.impl("clamp_min.out", &clamp_min_out_vulkan);
    m.impl("clamp_max.out", &clamp_max_out_vulkan);
    m.impl("clamp_min.Tensor_out", &clamp_min_tensor_out_vulkan);
    m.impl("clamp_max.Tensor_out", &clamp_max_tensor_out_vulkan);
    m.impl("clamp.out", &clamp_out_vulkan);
    m.impl("clamp.Tensor_out", &clamp_tensor_out_vulkan);
    m.impl("addcmul_", &addcmul_vulkan_);
    m.impl("addcdiv_", &addcdiv_vulkan_);
    m.impl("lerp.Scalar", &lerp_scalar_vulkan);
    m.impl("lerp_.Scalar", &lerp_scalar_vulkan_);
    m.impl("lerp.Scalar_out", &lerp_scalar_vulkan_out);
    m.impl("threshold_backward", &threshold_backward_vulkan);
    m.impl("floor_divide", &floor_divide_vulkan);
    m.impl("div.Tensor_mode", &div_mode_vulkan);
    m.impl("div.out_mode", &div_out_mode_vulkan);
    m.impl("addcmul.out", &addcmul_out_vulkan);
    m.impl("addcdiv.out", &addcdiv_out_vulkan);
    m.impl("floor_divide.out", &floor_divide_out_vulkan);
    m.impl("floor_divide_.Tensor", &floor_divide_vulkan_);
    m.impl("rsub.Tensor", &rsub_tensor_vulkan);
    m.impl("lerp.Tensor", &lerp_tensor_vulkan);
    m.impl("lerp.Tensor_out", &lerp_tensor_vulkan_out);
    m.impl("pow.Tensor_Tensor_out", &pow_tensor_tensor_out_vulkan);
    m.impl("pow.Tensor_Scalar_out", &pow_tensor_scalar_out_vulkan);
    m.impl("pow.Scalar_out", &pow_scalar_out_vulkan);
    m.impl("signbit.out", &signbit_out_vulkan);
    m.impl("copysign.out", &copysign_out_vulkan);
    m.impl("nextafter.out", &nextafter_out_vulkan);
    m.impl("round.decimals_out", &round_decimals_out_vulkan);
    m.impl("frexp.Tensor_out", &frexp_out_vulkan);
}
