#include <torch/extension.h>
#include "api/ops/binary.h"

at::Tensor torchvulkan::binary_op_vulkan(
    const at::Tensor& self, 
    const at::Tensor& other, 
    const at::Scalar& alpha, 
    BinaryOp operation, 
    const std::function<at::Tensor(const at::Tensor&, const at::Tensor&)>& fallback)
{    
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
    uint32_t numel = iter.numel();
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
    uint32_t key = (workgroupSizeX << 10) | (out_dims << 6) | (use_scalar << 5) | (contiguous << 4) | op;
    SpecializationArgs specialization = {spd.data(), spd.offsets(), spd.sizes(), spd.numConstants(), key};

    IntDivider sizes; 
    uint32_t strides_a[MAX_DIMS] = {0};
    uint32_t strides_b[MAX_DIMS] = {0};
    uint32_t strides_out[MAX_DIMS] = {0};
    
    if (!contiguous) {
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
    }

    PushConstantBuilder pcs{};
    pcs.push(sizes)
       .push_array(strides_a)
       .push_array(strides_b)
       .push_array(strides_out)
       .push(numel)
       .push((uint32_t)0) // pad to align to 8-bytes
       .push_scalar(alpha, promoted_type)
       .push_scalar((at::Scalar)0, promoted_type);
           
    uint32_t numel_vec = !contiguous ? numel : (numel + (vecSize - 1)) / vecSize;
    uint32_t groupX = (numel_vec + (workgroupSizeX - 1)) / workgroupSizeX;

    VulkanShader shader(shader_id, specialization, device);
    shader.dispatch(
        &pcs, 
        pcs.size(), 
        {self_dtype, other_dtype, out}, 
        groupX, 1, 1
    );

    return out;
}

at::Tensor torchvulkan::binary_op_vulkan(
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
    uint32_t numel = iter.numel();
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
    uint32_t key = (workgroupSizeX << 10) | (out_dims << 6) | (use_scalar << 5) | (contiguous << 4) | op;
    SpecializationArgs specialization = {spd.data(), spd.offsets(), spd.sizes(), spd.numConstants(), key};

    IntDivider sizes; 
    uint32_t strides_a[MAX_DIMS] = {0};
    uint32_t strides_b[MAX_DIMS] = {0};
    uint32_t strides_out[MAX_DIMS] = {0};
    
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
    }

    PushConstantBuilder pcs{};
    pcs.push(sizes)
       .push_array(strides_a)
       .push_array(strides_b)
       .push_array(strides_out)
       .push(numel)
       .push((uint32_t)0) // pad to align to 8-bytes
       .push_scalar(alpha, promoted_type)
       .push_scalar(other, promoted_type);

    uint32_t numel_vec = !contiguous ? numel : (numel + (vecSize-1)) / vecSize;
    uint32_t groupX = (numel_vec + (workgroupSizeX - 1)) / workgroupSizeX;
    
    VulkanShader shader(shader_id, specialization, device);
    shader.dispatch(
        &pcs, 
        pcs.size(), 
        {self_dtype, self_dtype, out}, 
        groupX, 1, 1
    );

    return out;
}

at::Tensor torchvulkan::add_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) 
{    
    if (self.scalar_type() == at::kBool) {
        if (alpha.to<bool>() == false) return self.clone();
        return binary_op_vulkan(self, other, alpha, BinaryOp::MAX, [alpha](const at::Tensor& a, const at::Tensor& b) { return at::add(a, b, alpha); });
    }
    return binary_op_vulkan(self, other, alpha, BinaryOp::ADD, [alpha](const at::Tensor& a, const at::Tensor& b) { return at::add(a, b, alpha); });
}

at::Tensor torchvulkan::add_scalar_vulkan(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) 
{
    if (self.scalar_type() == at::kBool) {
        if (alpha.to<bool>() == false) return self.clone();
        return binary_op_vulkan(self, other, alpha, BinaryOp::MAX, [alpha](const at::Tensor& a, const at::Scalar& b) { return at::add(a, b, alpha); });
    }
    return binary_op_vulkan(self, other, alpha, BinaryOp::ADD, [alpha](const at::Tensor& a, const at::Scalar& b) { return at::add(a, b, alpha); });
}

at::Tensor torchvulkan::subtract_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    return binary_op_vulkan(self, other, alpha, BinaryOp::SUB, [alpha](const at::Tensor& a, const at::Tensor& b) { return at::sub(a, b, alpha); });
}

at::Tensor torchvulkan::subtract_scalar_vulkan(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
    return binary_op_vulkan(self, other, alpha, BinaryOp::SUB, [alpha](const at::Tensor& a, const at::Scalar& b) { return at::sub(a, b, alpha); });
}

at::Tensor torchvulkan::rsub_scalar_vulkan(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
    return binary_op_vulkan(self, other, alpha, BinaryOp::RSUB, [alpha](const at::Tensor& a, const at::Scalar& b) { return at::rsub(a, b, alpha); });
}

at::Tensor torchvulkan::multiply_vulkan(const at::Tensor& self, const at::Tensor& other) {
    return binary_op_vulkan(self, other, (int)1, BinaryOp::MUL, [](const at::Tensor& a, const at::Tensor& b) { return at::mul(a, b); });
}

at::Tensor torchvulkan::multiply_scalar_vulkan(const at::Tensor& self, const at::Scalar& other) {
    return binary_op_vulkan(self, other, (int)1, BinaryOp::MUL, [](const at::Tensor& a, const at::Scalar& b) { return at::mul(a, b); });
}

at::Tensor torchvulkan::divide_vulkan(const at::Tensor& self, const at::Tensor& other) {
    at::Tensor self_vulkan = self;
    at::Tensor other_vulkan = other;
    
    if (at::isIntegralType(self.scalar_type(), /* includeBool = */ true)) self_vulkan = self.to(c10::kFloat);
    if (at::isIntegralType(other.scalar_type(), /* includeBool = */ true)) other_vulkan = other.to(c10::kFloat);

    return binary_op_vulkan(self_vulkan, other_vulkan, (int)1, BinaryOp::DIV, [](const at::Tensor& a, const at::Tensor& b) { return at::div(a, b); });
}

at::Tensor torchvulkan::divide_scalar_vulkan(const at::Tensor& self, const at::Scalar& other) {
    at::Tensor self_vulkan = self;

    if (at::isIntegralType(self.scalar_type(), /* includeBool = */ true)) self_vulkan = self.to(c10::kFloat);

    return binary_op_vulkan(self_vulkan, other, (int)1, BinaryOp::DIV, [](const at::Tensor& a, const at::Scalar& b) { return at::div(a, b); });
}

at::Tensor torchvulkan::maximum_vulkan(const at::Tensor& self, const at::Tensor& other) {
    return binary_op_vulkan(self, other, (int)1, BinaryOp::MAX, [](const at::Tensor& a, const at::Tensor& b) { return at::max(a, b); });
}

at::Tensor torchvulkan::minimum_vulkan(const at::Tensor& self, const at::Tensor& other) {
    return binary_op_vulkan(self, other, (int)1, BinaryOp::MIN, [](const at::Tensor& a, const at::Tensor& b) { return at::min(a, b); });
}

at::Tensor torchvulkan::pow_vulkan(const at::Tensor& self, const at::Tensor& other) {
    return binary_op_vulkan(self, other, (int)1, BinaryOp::POW, [](const at::Tensor& a, const at::Tensor& b) { return at::pow(a, b); });
}

at::Tensor torchvulkan::pow_tensor_scalar_vulkan(const at::Tensor& self, const at::Scalar& other) {
    return binary_op_vulkan(self, other, (int)1, BinaryOp::POW, [](const at::Tensor& a, const at::Scalar& b) { return at::pow(a, b); });
}

at::Tensor torchvulkan::pow_scalar_vulkan(const at::Scalar& self, const at::Tensor& other) {
    return binary_op_vulkan(other, self, (int)1, BinaryOp::RPOW, [](const at::Tensor& a, const at::Scalar& b) { return at::pow(b, a); });
}

at::Tensor torchvulkan::atan2_vulkan(const at::Tensor& self, const at::Tensor& other) {
    at::Tensor self_vulkan = self;
    at::Tensor other_vulkan = other;

    if (at::isIntegralType(self.scalar_type(), /* includeBool = */ true)) self_vulkan = self.to(c10::kFloat);
    if (at::isIntegralType(other.scalar_type(), /* includeBool = */ true)) other_vulkan = other.to(c10::kFloat);

    return binary_op_vulkan(self_vulkan, other_vulkan, (int)1, BinaryOp::ATAN2, [](const at::Tensor& a, const at::Tensor& b) { return at::atan2(a, b); });
}