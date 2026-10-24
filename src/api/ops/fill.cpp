#include <torch/extension.h>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "api/ops/helpers.h"
#include "api/fallback.h"

namespace {

at::Tensor& fill_scalar_vulkan(
    at::Tensor& self, 
    const at::Scalar& value)
{
    at::TensorIterator iter = at::TensorIterator::borrowing_nullary_op(self);

    uint64_t numel = iter.numel();
    if (numel == 0) return self;
    
    int32_t out_dims = static_cast<int32_t>(iter.ndim());
    if (!is_dtype_supported(self, iter.dtype()) || out_dims > MAX_DIMS || !fits_32bit_indexing(iter)) {
        TORCHVULKAN_CPU_FALLBACK("Can't fill a ", iter.dtype(), " tensor of this shape on the Vulkan device. Falling back to CPU.");
        self.copy_(at::empty(self.sizes(), self.options().device(at::kCPU)).fill_(value));
        return self;
    }

    DeviceContext* device = get_device_context(self);
    ShaderKey shader_key{torchvulkan::Kernel::FILL, self.scalar_type()};
    uint32_t vecSize = get_dtype_vec_size(self.scalar_type()); // our workgroup must match the shader workgroup
    uint32_t workgroupSizeX = get_dtype_workgroup_size(self.scalar_type(), vecSize, device);
    uint32_t contiguous = iter.is_contiguous() ? 1 : 0;

    SpecializationBuilder spd{};
    spd.push(out_dims)
        .push(contiguous)
        .push(workgroupSizeX);

    MetadataBuilder metadataBuilder = get_iter_metadata(iter, {0});
    Metadata metadata = metadataBuilder.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(self))
        .push_metadata(contiguous ? nullptr : &metadata)
        .push(numel)
        .push_scalar(value, self.scalar_type());

    device->shader_manager->dispatchShader(
        shader_key,
        spd.build(),
        pcs.build(),
        /* read = */ {},
        /* write = */ {self},
        get_group_count(contiguous ? (numel + vecSize - 1) / vecSize : numel, workgroupSizeX), 1, 1
    );

    return self;
}

at::Tensor& fill_tensor_vulkan(at::Tensor& self, const at::Tensor& value) 
{
    TORCH_CHECK(value.dim() == 0, "fill_ only supports 0-dimension value tensor but got tensor with ", value.dim(), " dimensions.");
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
