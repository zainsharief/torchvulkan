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
    
    if (!is_dtype_supported(iter.dtype()) || !is_dtype_supported(value.type())) {
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

    IntDivider sizes[MAX_DIMS];
    uint32_t strides_in[MAX_DIMS] = {0};
    
    if (!contiguous) 
    {
        int64_t el_size = iter.element_size(0);
        at::IntArrayRef iter_shape = iter.shape();
        at::IntArrayRef iter_strides_in = iter.strides(0);

        for (int i = 0; i < out_dims; i++) {
            sizes[i] = IntDivider(iter_shape[i]);
            strides_in[i] = iter_strides_in[i] / el_size;
        }
    }

    PushConstantBuilder pcs{};
    pcs.push_array(sizes)
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