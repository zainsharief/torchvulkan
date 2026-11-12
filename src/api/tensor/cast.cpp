#include <torch/extension.h>
#include <algorithm>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "api/ops/helpers.h"
#include "api/ops/internal.h"

void torchvulkan::dispatch_cast_shader(const at::Tensor& src, const at::Tensor& dst)
{
    at::TensorIterator iter = at::TensorIteratorConfig()
        .set_check_mem_overlap(true)
        .check_all_same_dtype(false)
        .add_output(dst)
        .add_input(src)
        .build();

    uint64_t numel = iter.numel();
    if (numel == 0) return;
    int32_t out_dims = static_cast<int32_t>(iter.ndim());
    TORCH_CHECK(out_dims <= MAX_DIMS, "torchvulkan [ERROR]: A cast has ", out_dims, " dimensions after coalescing; the cast shader supports ", MAX_DIMS, ".");
    TORCH_CHECK(fits_32bit_indexing(iter), "torchvulkan [ERROR]: A cast of ", numel, " elements is too large for the cast shader's 32-bit indexing.");

    DeviceContext* device = get_device_context(dst);
    ShaderKey shader_key{torchvulkan::Kernel::CAST, src.scalar_type(), dst.scalar_type()};
    uint32_t vecSize = std::min(get_dtype_vec_size(src.scalar_type()), get_dtype_vec_size(dst.scalar_type()));
    uint32_t workgroupSizeX = get_dtype_workgroup_size(dst.scalar_type(), vecSize, device);
    uint32_t isBoolCast = (dst.scalar_type() == at::kBool) ? 1 : 0;
    uint32_t isContig = iter.is_contiguous() ? 1 : 0;

    SpecializationBuilder spd{};
    spd.push(out_dims)
        .push(workgroupSizeX)
        .push(isBoolCast)
        .push(isContig);

    MetadataBuilder metadataBuilder = get_iter_metadata(iter, {1, 0});
    Metadata metadata = metadataBuilder.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(src))
        .push(get_tensor_address(dst))
        .push_metadata(isContig ? nullptr : &metadata)
        .push(numel);

    uint64_t threads = isContig ? (numel + (vecSize - 1)) / vecSize : numel;
    device->shader_manager->dispatchShader(
        shader_key,
        device->default_subgroup_size,
        spd.build(),
        pcs.build(),
        /* read = */ {src},
        /* write = */ {dst},
        get_group_count(threads, workgroupSizeX), 1, 1
    );
}
