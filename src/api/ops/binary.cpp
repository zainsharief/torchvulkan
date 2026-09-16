#include <torch/extension.h>
#include <ATen/TensorIterator.h>
#include <c10/core/MemoryFormat.h>
#include <functional>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "shaders/shader_registry.h"
#include "api/ops/helpers.h"
#include "api/ops/internal.h"

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
    LOGADDEXP2 = 18
};

at::Tensor binary_op_vulkan(
    const at::Tensor& self, 
    const at::Scalar& other, 
    const at::Scalar& alpha, 
    BinaryOp operation, 
    const std::function<at::Tensor(const at::Tensor&, const at::Scalar&)>& fallback)
{
    c10::ScalarType promoted_type = at::result_type(self, other);
    
    if (!is_dtype_supported(promoted_type)) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Vulkan device does not support ", promoted_type, ". Falling back to CPU.");
        at::Tensor out = fallback(self.cpu(), other);
        return out.to(self.device());
    }

    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    uint32_t alignment = device->properties.limits.minStorageBufferOffsetAlignment;
    
    at::Tensor self_dtype = self.to(promoted_type);
    size_t self_offset_bytes = self_dtype.storage_offset() * self_dtype.element_size();
    if (self_offset_bytes % alignment != 0) {
        self_dtype = self_dtype.clone();
    }
    
    at::Tensor out;
    at::TensorIterator iter = at::TensorIteratorConfig()
        .set_check_mem_overlap(true)
        .add_output(out)
        .add_input(self_dtype)
        .build();

    out = iter.output();
    uint64_t numel = iter.numel();
    if (numel == 0) return out;

    int32_t out_dims = static_cast<int32_t>(iter.ndim());
    if (out_dims > MAX_DIMS) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Coalesced dimensions (", out_dims, ") exceed maximum supported (", MAX_DIMS, "). Falling back to CPU.");
        at::Tensor out = fallback(self.cpu(), other);
        return out.to(self.device());
    }
    
    uint32_t vecSize = get_dtype_vec_size(promoted_type);
    uint32_t workgroupSizeX = get_dtype_workgroup_size(promoted_type, vecSize);

    uint32_t contiguous = iter.is_contiguous();
    torchvulkan::ShaderID shader_id = torchvulkan::get_shader_id_binaryop(promoted_type);
    uint32_t op = static_cast<uint32_t>(operation);
    uint32_t use_scalar = 1;

    SpecializationBuilder spd{};
    spd.push(op)
       .push(contiguous)
       .push(use_scalar)
       .push(out_dims)
       .push(workgroupSizeX);
    SpecializationArgs specialization = spd.build();

    IntDivider sizes; 
    uint32_t strides_a[MAX_DIMS] = {0};
    uint32_t strides_b[MAX_DIMS] = {0};
    uint32_t strides_out[MAX_DIMS] = {0};
    
    uint64_t metadata_address = 0;
    if (!contiguous) {
        int64_t el_size = iter.element_size(0);
        at::IntArrayRef iter_shape = iter.shape();
        at::IntArrayRef iter_strides_out = iter.strides(0);
        at::IntArrayRef iter_strides_a = iter.strides(1);

        for (int i = 0; i < out_dims; i++) {
            sizes.set(i, iter_shape[i]);
            strides_a[i] = iter_strides_a[i] / el_size;
            strides_out[i] = iter_strides_out[i] / el_size;
        }

        MetadataBuilder metadataBuilder{};
        metadataBuilder.push(sizes, out_dims)
                       .push_array(strides_a, out_dims)
                       .push_array(strides_b, out_dims)
                       .push_array(strides_out, out_dims);
        Metadata metadata = metadataBuilder.build();
        metadata_address = device->shader_manager->registerMetadata(metadata);
    }

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(self_dtype))
       .push(/* tensor_b = */ (uint64_t)0) // the second operand is a scalar, so the shader never reads it
       .push(get_tensor_address(out))
       .push(metadata_address)
       .push(numel)
       .push_scalar(alpha, promoted_type)
       .push_scalar(other, promoted_type);

    uint64_t numel_vec = !contiguous ? numel : (numel + (vecSize-1)) / vecSize;
    uint32_t groupX = (numel_vec + (workgroupSizeX - 1)) / workgroupSizeX;
    
    PushConstants pushConstants = { const_cast<void*>(pcs.data()), pcs.size() };
    device->shader_manager->dispatchShader(
        shader_id,
        specialization,
        pushConstants,
        /* read = */ {self_dtype},
        /* write = */ {out},
        groupX, 1, 1
    );

    return out;
}

at::Tensor binary_op_vulkan(
    const at::Tensor& self, 
    const at::Tensor& other, 
    const at::Scalar& alpha, 
    BinaryOp operation, 
    const std::function<at::Tensor(const at::Tensor&, const at::Tensor&)>& fallback)
{    
    // use the scalar function if other is a scalar wrapped in a tensor 
    if (other.unsafeGetTensorImpl()->is_wrapped_number()) 
    {
        return binary_op_vulkan(
            self, other.item(), alpha, operation,
            [&fallback, &other](const at::Tensor& a, const at::Scalar&) { return fallback(a, other); }
        );
    }

    c10::ScalarType promoted_type = at::result_type(self, other);
    
    if (!is_dtype_supported(promoted_type)) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Vulkan device does not support ", promoted_type, ". Falling back to CPU.");
        at::Tensor out = fallback(self.cpu(), other.cpu());
        return out.to(self.device());
    }

    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    uint32_t alignment = device->properties.limits.minStorageBufferOffsetAlignment;
    
    at::Tensor self_dtype = self.to(promoted_type);
    size_t self_offset_bytes = self_dtype.storage_offset() * self_dtype.element_size();
    if (self_offset_bytes % alignment != 0) {
        self_dtype = self_dtype.clone();
    }

    at::Tensor other_dtype = other.to(self_dtype.options());
    size_t other_offset_bytes = other_dtype.storage_offset() * other_dtype.element_size();
    if (other_offset_bytes % alignment != 0) {
        other_dtype = other_dtype.clone();
    }
    
    at::Tensor out;
    at::TensorIterator iter = at::TensorIteratorConfig()
        .set_check_mem_overlap(true)
        .add_output(out)
        .add_input(self_dtype)
        .add_input(other_dtype)
        .build();

    out = iter.output();
    uint64_t numel = iter.numel();
    if (numel == 0) return out;

    int32_t out_dims = static_cast<int32_t>(iter.ndim());
    if (out_dims > MAX_DIMS) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Coalesced dimensions (", out_dims, ") exceed maximum supported (", MAX_DIMS, "). Falling back to CPU.");
        at::Tensor out = fallback(self.cpu(), other.cpu());
        return out.to(self.device());
    }

    uint32_t vecSize = get_dtype_vec_size(promoted_type);
    uint32_t workgroupSizeX = get_dtype_workgroup_size(promoted_type, vecSize);

    uint32_t contiguous = iter.is_contiguous() ? 1 : 0;
    torchvulkan::ShaderID shader_id = torchvulkan::get_shader_id_binaryop(promoted_type);
    uint32_t op = static_cast<uint32_t>(operation);
    uint32_t use_scalar = 0;

    SpecializationBuilder spd{};
    spd.push(op)
       .push(contiguous)
       .push(use_scalar)
       .push(out_dims)
       .push(workgroupSizeX);
    SpecializationArgs specialization = spd.build();

    IntDivider sizes; 
    uint32_t strides_a[MAX_DIMS] = {0};
    uint32_t strides_b[MAX_DIMS] = {0};
    uint32_t strides_out[MAX_DIMS] = {0};
    
    uint64_t metadata_address = 0;
    if (!contiguous) 
    {
        int64_t el_size = iter.element_size(0);
        at::IntArrayRef iter_shape = iter.shape();
        at::IntArrayRef iter_strides_out = iter.strides(0);
        at::IntArrayRef iter_strides_a = iter.strides(1);
        at::IntArrayRef iter_strides_b = iter.strides(2);

        for (int i = 0; i < out_dims; i++) {
            sizes.set(i, iter_shape[i]);
            strides_a[i] = iter_strides_a[i] / el_size;
            strides_b[i] = iter_strides_b[i] / el_size;
            strides_out[i] = iter_strides_out[i] / el_size;
        }

        MetadataBuilder metadataBuilder{};
        metadataBuilder.push(sizes, out_dims)
                       .push_array(strides_a, out_dims)
                       .push_array(strides_b, out_dims)
                       .push_array(strides_out, out_dims);
        Metadata metadata = metadataBuilder.build();
        metadata_address = device->shader_manager->registerMetadata(metadata);
    }

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(self_dtype))
       .push(get_tensor_address(other_dtype))
       .push(get_tensor_address(out))
       .push(metadata_address)
       .push(numel)
       .push_scalar(alpha, promoted_type)
       .push_scalar((at::Scalar)0, promoted_type);
           
    uint64_t numel_vec = !contiguous ? numel : (numel + (vecSize - 1)) / vecSize;
    uint32_t groupX = (numel_vec + (workgroupSizeX - 1)) / workgroupSizeX;

    PushConstants pushConstants = { const_cast<void*>(pcs.data()), pcs.size() };
    device->shader_manager->dispatchShader(
        shader_id,
        specialization,
        pushConstants,
        /* read = */ {self_dtype, other_dtype},
        /* write = */ {out},
        groupX, 1, 1
    );

    return out;
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

    if (!self.is_floating_point() && !other.is_floating_point()) {
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

    if (!self.is_floating_point() && !other.is_floating_point()) {
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
    at::Tensor result = self;
    if (min.has_value()) result = clamp_min_scalar_vulkan(result, *min);
    if (max.has_value()) result = clamp_max_scalar_vulkan(result, *max);
    return result;
}

at::Tensor clamp_tensor_vulkan(const at::Tensor& self, const c10::optional<at::Tensor>& min, const c10::optional<at::Tensor>& max) {
    at::Tensor result = self;
    if (min.has_value()) result = maximum_vulkan(result, *min);
    if (max.has_value()) result = minimum_vulkan(result, *max);
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

at::Tensor& add_vulkan_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    return self.copy_(torchvulkan::add_vulkan(self, other, alpha));
}

at::Tensor& add_scalar_vulkan_(at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
    return self.copy_(add_scalar_vulkan(self, other, alpha));
}

at::Tensor& subtract_vulkan_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    return self.copy_(subtract_vulkan(self, other, alpha));
}

at::Tensor& subtract_scalar_vulkan_(at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
    return self.copy_(subtract_scalar_vulkan(self, other, alpha));
}

at::Tensor& multiply_vulkan_(at::Tensor& self, const at::Tensor& other) {
    return self.copy_(torchvulkan::multiply_vulkan(self, other));
}

at::Tensor& multiply_scalar_vulkan_(at::Tensor& self, const at::Scalar& other) {
    return self.copy_(multiply_scalar_vulkan(self, other));
}

at::Tensor& divide_vulkan_(at::Tensor& self, const at::Tensor& other) {
    return self.copy_(torchvulkan::divide_vulkan(self, other));
}

at::Tensor& divide_scalar_vulkan_(at::Tensor& self, const at::Scalar& other) {
    return self.copy_(torchvulkan::divide_scalar_vulkan(self, other));
}

at::Tensor& addcmul_vulkan_(at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2, const at::Scalar& value) {
    at::Tensor product = torchvulkan::multiply_vulkan(tensor1, tensor2);
    return self.copy_(torchvulkan::add_vulkan(self, product, value));
}

at::Tensor& addcdiv_vulkan_(at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2, const at::Scalar& value) {
    at::Tensor quotient = torchvulkan::divide_vulkan(tensor1, tensor2);
    return self.copy_(torchvulkan::add_vulkan(self, quotient, value));
}

at::Tensor lerp_scalar_vulkan(const at::Tensor& self, const at::Tensor& end, const at::Scalar& weight) {
    at::Tensor diff = subtract_vulkan(end, self, (int)1);
    return torchvulkan::add_vulkan(self, diff, weight);
}

at::Tensor& lerp_scalar_vulkan_(at::Tensor& self, const at::Tensor& end, const at::Scalar& weight) {
    return self.copy_(lerp_scalar_vulkan(self, end, weight));
}

at::Tensor& lerp_scalar_vulkan_out(const at::Tensor& self, const at::Tensor& end, const at::Scalar& weight, at::Tensor& out) {
    return out.copy_(lerp_scalar_vulkan(self, end, weight));
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("add.Tensor", &torchvulkan::add_vulkan);
    m.impl("add.Scalar", &add_scalar_vulkan);
    m.impl("add_.Tensor", &add_vulkan_);
    m.impl("add_.Scalar", &add_scalar_vulkan_);
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
}
