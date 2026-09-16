#include <torch/extension.h>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "shaders/shader_registry.h"
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
    if (out_dims > MAX_DIMS) {
        TORCH_CHECK(false, "torchvulkan [WARNING]: Coalesced dimensions (", out_dims, ") exceed maximum supported (", MAX_DIMS, "). Falling back to CPU.");
    }

    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    torchvulkan::ShaderID shader_id = torchvulkan::get_shader_id_cast(src.scalar_type(), dst.scalar_type());
    uint32_t vecSize = get_dtype_vec_size(dst.scalar_type()); // our workgroup must match the shader workgroup
    uint32_t workgroupSizeX = get_dtype_workgroup_size(dst.scalar_type(), vecSize);
    uint32_t isBoolCast = (dst.scalar_type() == at::kBool) ? 1 : 0;

    SpecializationBuilder spd{};
    spd.push(out_dims)
        .push(workgroupSizeX)
        .push(isBoolCast);
    SpecializationArgs specialization = spd.build();

    IntDivider sizes; 
    uint32_t strides_in[MAX_DIMS] = {0};
    uint32_t strides_out[MAX_DIMS] = {0};
    
    int64_t el_size_out = iter.element_size(0);
    int64_t el_size_in = iter.element_size(1);
    at::IntArrayRef iter_shape = iter.shape();
    at::IntArrayRef iter_strides_out = iter.strides(0);
    at::IntArrayRef iter_strides_in = iter.strides(1);

    for (int i = 0; i < out_dims; i++) {
        sizes.set(i, iter_shape[i]);
        strides_in[i] = iter_strides_in[i] / el_size_in;
        strides_out[i] = iter_strides_out[i] / el_size_out;
    }

    MetadataBuilder metadataBuilder{};
    metadataBuilder.push(sizes, out_dims)
                   .push_array(strides_in, out_dims)
                   .push_array(strides_out, out_dims);
    Metadata metadata = metadataBuilder.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(src))
        .push(get_tensor_address(dst))
        .push(device->shader_manager->registerMetadata(metadata))
        .push(numel);

    uint32_t groupX = (numel + (workgroupSizeX - 1)) / workgroupSizeX;

    PushConstants pushConstants = { const_cast<void*>(pcs.data()), pcs.size() };
    device->shader_manager->dispatchShader(
        shader_id,
        specialization,
        pushConstants,
        /* read = */ {src},
        /* write = */ {dst},
        groupX, 1, 1
    );
}
