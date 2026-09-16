#include <torch/extension.h>
#include <c10/core/MemoryFormat.h>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "shaders/shader_registry.h"
#include "api/ops/helpers.h"
#include "api/ops/internal.h"

namespace {

void dispatch_copy_shader(const at::Tensor& src, const at::Tensor& dst) 
{
    at::TensorIterator iter = at::TensorIteratorConfig()
        .set_check_mem_overlap(true)
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
    torchvulkan::ShaderID shader_id = torchvulkan::get_shader_id_copy(dst.scalar_type());
    uint32_t vecSize = get_dtype_vec_size(dst.scalar_type()); // our workgroup must match the shader workgroup
    uint32_t workgroupSizeX = get_dtype_workgroup_size(dst.scalar_type(), vecSize);

    SpecializationBuilder spd{};
    spd.push(out_dims)
        .push(workgroupSizeX);
    uint32_t key = (workgroupSizeX << 4) | out_dims;
    SpecializationArgs specialization = {spd.data(), spd.offsets(), spd.sizes(), spd.numConstants(), key};

    IntDivider sizes; 
    uint32_t strides_in[MAX_DIMS] = {0};
    uint32_t strides_out[MAX_DIMS] = {0};
    
    int64_t el_size = iter.element_size(0);
    at::IntArrayRef iter_shape = iter.shape();
    at::IntArrayRef iter_strides_out = iter.strides(0);
    at::IntArrayRef iter_strides_in = iter.strides(1);

    for (int i = 0; i < out_dims; i++) {
        sizes.set(i, iter_shape[i]);
        strides_in[i] = iter_strides_in[i] / el_size;
        strides_out[i] = iter_strides_out[i] / el_size;
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

} // namespace

at::Tensor torchvulkan::copy_vulkan(
    const at::Tensor& self, 
    const at::Tensor& dst, 
    bool non_blocking) 
{
    if (dst.numel() == 0) return dst;
    TORCH_CHECK(!self.is_conj() && !self.is_neg(), "torchvulkan [NOT IMPLEMENTED]: Copying from a conjugated or negated source is not yet supported.");
    TORCH_CHECK(!dst.is_conj() && !dst.is_neg(), "torchvulkan [NOT IMPLEMENTED]: Copying into a conjugated or negated destination is not yet supported.");

    at::Tensor src = (self.sizes() == dst.sizes()) ? self : self.expand(dst.sizes());
    c10::DeviceType src_type = src.device().type();
    c10::DeviceType dst_type = dst.device().type();

    if (dst_type == c10::DeviceType::PrivateUse1) VulkanContext::SetCurrentDevice(dst.device().index());
    else if (src_type == c10::DeviceType::PrivateUse1) VulkanContext::SetCurrentDevice(src.device().index());
    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();

    if (src_type == at::DeviceType::CPU && dst_type == c10::DeviceType::PrivateUse1) 
    {
        if (src.scalar_type() != dst.scalar_type()) src = src.to(dst.scalar_type());

        void* dest_ptr = (void*)dst.storage().data_ptr().get_context();
        uint64_t dest_offset = dst.storage_offset() * dst.itemsize();
        
        if (!src.is_contiguous()) src = src.contiguous();
        if (dst.is_contiguous()) {
            globalVulkanAllocator.copy_host_to_device(dest_ptr, dest_offset, src.data_ptr(), dst.nbytes());
            return dst;
        }

        at::Tensor stagingBuffer = at::empty_like(dst, dst.options().memory_format(at::MemoryFormat::Contiguous));
        void* staging_buffer_ptr = (void*)stagingBuffer.storage().data_ptr().get_context();
        uint64_t staging_buffer_offset = stagingBuffer.storage_offset() * dst.itemsize();

        globalVulkanAllocator.copy_host_to_device(staging_buffer_ptr, staging_buffer_offset, src.data_ptr(), dst.nbytes());
        dispatch_copy_shader(stagingBuffer, dst);
    }
    else if (src_type == c10::DeviceType::PrivateUse1 && dst_type == at::DeviceType::CPU) 
    {
        if (src.scalar_type() != dst.scalar_type()) 
        {
            if (!is_dtype_supported(src.scalar_type()) || !is_dtype_supported(dst.scalar_type())) { 
                TORCH_WARN_ONCE("torchvulkan [WARNING]: Vulkan device does not support source or destination dtype. Falling back to CPU for copy.");
                at::Tensor out = dst.copy_(src.to(at::DeviceType::CPU), non_blocking);
                return out;
            }

            at::Tensor temp = at::empty_like(src, src.options().dtype(dst.scalar_type()).memory_format(at::MemoryFormat::Contiguous));
            torchvulkan::dispatch_cast_shader(src, temp);
            src = std::move(temp);
        }

        if (!src.is_contiguous()) src = src.contiguous();

        void* src_ptr = (void*)src.storage().data_ptr().get_context();
        uint64_t src_offset = src.storage_offset() * src.itemsize();

        if (dst.is_contiguous()) {    
            globalVulkanAllocator.copy_device_to_host(dst.data_ptr(), src_ptr, src_offset, dst.nbytes());
            return dst;
        }

        at::Tensor stagingBuffer = at::empty_like(dst, dst.options().memory_format(at::MemoryFormat::Contiguous));
        globalVulkanAllocator.copy_device_to_host(stagingBuffer.data_ptr(), src_ptr, src_offset, stagingBuffer.nbytes());
        dst.copy_(stagingBuffer, non_blocking);
    } 
    else if (src_type == c10::DeviceType::PrivateUse1 && dst_type == c10::DeviceType::PrivateUse1 && src.device().index() == dst.device().index()) 
    {        
        if (src.scalar_type() != dst.scalar_type()) 
        {
            if (!is_dtype_supported(src.scalar_type()) || !is_dtype_supported(dst.scalar_type())) { 
                TORCH_WARN_ONCE("torchvulkan [WARNING]: Vulkan device does not support source or destination dtype. Falling back to CPU for copy.");
                at::Tensor out = dst.copy_(src.to(at::DeviceType::CPU), non_blocking);
                return out;
            }

            torchvulkan::dispatch_cast_shader(src, dst);
            return dst;
        }
        
        if (!src.is_contiguous() || !dst.is_contiguous()) {
            dispatch_copy_shader(src, dst);
            return dst;
        }

        void* src_ptr = (void*)src.storage().data_ptr().get_context();
        uint64_t src_offset = src.storage_offset() * src.itemsize();
        void* dest_ptr = (void*)dst.storage().data_ptr().get_context();
        uint64_t dest_offset = dst.storage_offset() * dst.itemsize();
        globalVulkanAllocator.copy_device_to_device(dest_ptr, dest_offset, src_ptr, src_offset, dst.nbytes());  
    } 
    else if (src_type == c10::DeviceType::PrivateUse1 && dst_type == c10::DeviceType::PrivateUse1 && src.device().index() != dst.device().index()) 
    {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Source and destination are on different Vulkan devices. Falling back to CPU for copy.");
        dst.copy_(src.to(at::DeviceType::CPU), non_blocking);
        return dst;
    } 
    else {
        // should never get here, but just in case
        dst.copy_(src.to(at::DeviceType::CPU), non_blocking);
    }

    return dst;
}

namespace {

at::Tensor copy_from_vulkan(
    const at::Tensor& self, 
    const at::Tensor& dst, 
    bool non_blocking) 
{
    return torchvulkan::copy_vulkan(self, dst, non_blocking);
}

at::Tensor copy_from_and_resize_vulkan(
    const at::Tensor& self, 
    const at::Tensor& dst) 
{
    torchvulkan::resize_vulkan(dst, self.sizes(), c10::nullopt);
    return torchvulkan::copy_vulkan(self, dst, false);
}

at::Tensor& copy_vulkan_(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
    // swap the arguments to match the expected order
    torchvulkan::copy_vulkan(src, self, non_blocking);
    return self;
}

at::Scalar local_scalar_dense_vulkan(const at::Tensor& self)
{
    return self.cpu().item();
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("_copy_from", &copy_from_vulkan);
    m.impl("_copy_from_and_resize", &copy_from_and_resize_vulkan);
    m.impl("copy_", &copy_vulkan_);
    m.impl("_local_scalar_dense", &local_scalar_dense_vulkan);
}
