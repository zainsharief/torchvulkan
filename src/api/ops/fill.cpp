#include <torch/extension.h>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "api/ops/helpers.h"

namespace {

at::Tensor& fill_scalar_vulkan(
    at::Tensor& self, 
    const at::Scalar& value)
{
    at::TensorIterator iter = at::TensorIteratorConfig()
        .add_output(self)
        .add_input(self)
        .build();

    uint64_t numel = iter.numel();
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
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Coalesced dimensions (", out_dims, ") exceed maximum supported (", MAX_DIMS, "). Falling back to CPU.");
        at::Tensor cpu_temp = self.to(at::kCPU);
        cpu_temp.fill_(value);
        self.copy_(cpu_temp);
        return self;
    }

    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    ShaderKey shader_key{torchvulkan::Kernel::FILL, self.scalar_type()};
    uint32_t vecSize = get_dtype_vec_size(self.scalar_type()); // our workgroup must match the shader workgroup
    uint32_t workgroupSizeX = get_dtype_workgroup_size(self.scalar_type(), vecSize);
    uint32_t contiguous = iter.is_contiguous() ? 1 : 0;

    SpecializationBuilder spd{};
    spd.push(out_dims)
        .push(contiguous)
        .push(workgroupSizeX);
    SpecializationArgs specialization = spd.build();

    IntDivider sizes; 
    uint32_t strides_in[MAX_DIMS] = {0};
    
    int64_t el_size = iter.element_size(0);
    at::IntArrayRef iter_shape = iter.shape();
    at::IntArrayRef iter_strides_in = iter.strides(0);

    uint64_t metadata_address = 0;
    if (!contiguous) 
    {
        for (int i = 0; i < out_dims; i++) {
            sizes.set(i, iter_shape[i]);
            strides_in[i] = iter_strides_in[i] / el_size;
        }

        MetadataBuilder metadataBuilder{};
        metadataBuilder.push(sizes, out_dims)
                       .push_array(strides_in, out_dims);
        Metadata metadata = metadataBuilder.build();
        metadata_address = device->shader_manager->registerMetadata(metadata);
    }

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(self))
        .push(metadata_address)
        .push(numel)
        .push_scalar(value, self.scalar_type());

    uint32_t groupX = (numel + (workgroupSizeX - 1)) / workgroupSizeX;

    PushConstants pushConstants = { const_cast<void*>(pcs.data()), pcs.size() };
    device->shader_manager->dispatchShader(
        shader_key,
        specialization,
        pushConstants,
        /* read = */ {},
        /* write = */ {self},
        groupX, 1, 1
    );

    return self;
}

at::Tensor& fill_tensor_vulkan(at::Tensor& self, const at::Tensor& value) 
{
    return fill_scalar_vulkan(self, value.item());
}

at::Tensor& zero_vulkan(at::Tensor& self)
{
    return fill_scalar_vulkan(self, 0);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("fill_.Scalar", &fill_scalar_vulkan);
    m.impl("fill_.Tensor", &fill_tensor_vulkan);
    m.impl("zero_", &zero_vulkan);
}
