#include <torch/extension.h>
#include "api/ops/unary.h"
#include "api/ops/helpers.h"

at::Tensor& torchvulkan::fill_scalar_vulkan(
    at::Tensor& self, 
    const at::Scalar& value)
{
    at::TensorIterator iter = at::TensorIteratorConfig()
        .add_output(self)
        .add_input(self)
        .build();

    uint32_t numel = iter.numel();
    if (numel == 0) return self;
    
    if (!is_dtype_supported(iter.dtype())) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Vulkan device does not support ", iter.dtype(), ". Falling back to CPU.");
        at::Tensor cpu_temp = self.to(at::kCPU);
        cpu_temp.fill_(value);        
        self.copy_(cpu_temp); 
        return self;
    }

    int32_t out_dims = static_cast<int32_t>(iter.ndim());
    if (out_dims > MAX_DIMS) {
        TORCH_CHECK(false, "torchvulkan [WARNING]: Coalesced dimensions (", out_dims, ") exceed maximum supported (", MAX_DIMS, "). Falling back to CPU.");
    }

    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    torchvulkan::ShaderID shader_id = torchvulkan::get_shader_id_fill(self.scalar_type());
    uint32_t vecSize = get_dtype_vec_size(self.scalar_type()); // our workgroup must match the shader workgroup
    uint32_t workgroupSizeX = get_dtype_workgroup_size(self.scalar_type(), vecSize);
    uint32_t contiguous = iter.is_contiguous() ? 1 : 0;

    SpecializationBuilder spd{};
    spd.push(out_dims)
        .push(contiguous)
        .push(workgroupSizeX);
    uint32_t key = (workgroupSizeX << 8) | (contiguous << 4) | out_dims;
    SpecializationArgs specialization = {spd.data(), spd.offsets(), spd.sizes(), spd.numConstants(), key};

    IntDivider sizes; 
    uint32_t strides_in[MAX_DIMS] = {0};
    
    int64_t el_size = iter.element_size(0);
    at::IntArrayRef iter_shape = iter.shape();
    at::IntArrayRef iter_strides_in = iter.strides(0);

    if (!contiguous) 
    {
        for (int i = 0; i < out_dims; i++) {
            sizes.set(i, iter_shape[i]);
            strides_in[i] = iter_strides_in[i] / el_size;
        }
    }

    PushConstantBuilder pcs{};
    pcs.push(sizes)
        .push_array(strides_in)
        .push(numel)
        .push((uint32_t)0) // padding
        .push_scalar(value, self.scalar_type());

    uint32_t groupX = (numel + (workgroupSizeX - 1)) / workgroupSizeX;

    VulkanShader shader(shader_id, specialization, device);
    shader.dispatch(
        &pcs, 
        pcs.size(), 
        {self}, 
        groupX, 1, 1
    );

    return self;
}

at::Tensor& torchvulkan::fill_tensor_vulkan(at::Tensor& self, const at::Tensor& value) 
{
    return torchvulkan::fill_scalar_vulkan(self, value.item());
}

at::Tensor& torchvulkan::zero_vulkan(at::Tensor& self)
{
    return fill_scalar_vulkan(self, 0);
}

void torchvulkan::dispatch_unary_shader(
    const at::Tensor& src,
    const at::Tensor& dst,
    UnaryOp operation)
{
    at::TensorIterator iter = at::TensorIteratorConfig()
        .set_check_mem_overlap(true)
        .add_output(dst)
        .add_input(src)
        .build();

    uint32_t numel = iter.numel();
    if (numel == 0) return;

    int32_t out_dims = static_cast<int32_t>(iter.ndim());
    if (out_dims > MAX_DIMS) {
        TORCH_CHECK(false, "torchvulkan [WARNING]: Coalesced dimensions (", out_dims, ") exceed maximum supported (", MAX_DIMS, "). Falling back to CPU.");
    }

    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    uint32_t vecSize = get_dtype_vec_size(dst.scalar_type());
    uint32_t workgroupSizeX = get_dtype_workgroup_size(dst.scalar_type(), vecSize);

    uint32_t contiguous = iter.is_contiguous() ? 1 : 0;
    torchvulkan::ShaderID shader_id = torchvulkan::get_shader_id_unaryop(dst.scalar_type());
    uint32_t op = static_cast<uint32_t>(operation);

    SpecializationBuilder spd{};
    spd.push(op)
        .push(contiguous)
        .push(out_dims)
        .push(workgroupSizeX);
    uint32_t key = (workgroupSizeX << 9) | (out_dims << 5) | (contiguous << 4) | op;
    SpecializationArgs specialization = {spd.data(), spd.offsets(), spd.sizes(), spd.numConstants(), key};

    IntDivider sizes;
    uint32_t strides_in[MAX_DIMS] = {0};
    uint32_t strides_out[MAX_DIMS] = {0};

    if (!contiguous) {
        int64_t el_size = iter.element_size(0);
        at::IntArrayRef iter_shape = iter.shape();
        at::IntArrayRef iter_strides_out = iter.strides(0);
        at::IntArrayRef iter_strides_in = iter.strides(1);

        for (int i = 0; i < out_dims; i++) {
            sizes.set(i, iter_shape[i]);
            strides_in[i] = iter_strides_in[i] / el_size;
            strides_out[i] = iter_strides_out[i] / el_size;
        }
    }

    PushConstantBuilder pcs{};
    pcs.push(sizes)
        .push_array(strides_in)
        .push_array(strides_out)
        .push(numel);

    uint32_t numel_vec = !contiguous ? numel : (numel + (vecSize - 1)) / vecSize;
    uint32_t groupX = (numel_vec + (workgroupSizeX - 1)) / workgroupSizeX;

    VulkanShader shader(shader_id, specialization, device);
    shader.dispatch(
        &pcs,
        pcs.size(),
        {src, dst},
        groupX, 1, 1
    );
}

at::Tensor torchvulkan::unary_op_vulkan(
    const at::Tensor& self,
    UnaryOp operation,
    const std::function<at::Tensor(const at::Tensor&)>& fallback)
{
    if (!is_dtype_supported(self.scalar_type())) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Vulkan device does not support ", self.scalar_type(), ". Falling back to CPU.");
        return fallback(self.cpu()).to(self.device());
    }

    at::Tensor out = at::empty_like(self);
    dispatch_unary_shader(self, out, operation);
    return out;
}

namespace {

at::Tensor promote_to_float(const at::Tensor& self)
{
    if (self.is_floating_point() || self.is_complex()) return self;
    return self.to(c10::typeMetaToScalarType(at::get_default_dtype()));
}

} // namespace

at::Tensor torchvulkan::relu_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(self, UnaryOp::RELU, [](const at::Tensor& a) { return at::relu(a); });
}

at::Tensor torchvulkan::exp_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::EXP, [](const at::Tensor& a) { return at::exp(a); });
}

at::Tensor torchvulkan::log_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::LOG, [](const at::Tensor& a) { return at::log(a); });
}

at::Tensor torchvulkan::sqrt_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::SQRT, [](const at::Tensor& a) { return at::sqrt(a); });
}

at::Tensor torchvulkan::neg_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(self, UnaryOp::NEG, [](const at::Tensor& a) { return at::neg(a); });
}

at::Tensor torchvulkan::reciprocal_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::RECIPROCAL, [](const at::Tensor& a) { return at::reciprocal(a); });
}