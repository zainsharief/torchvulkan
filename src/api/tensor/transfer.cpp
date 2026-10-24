#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/MemoryFormat.h>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "api/ops/helpers.h"
#include "api/ops/internal.h"
#include "api/fallback.h"

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
    TORCH_CHECK(out_dims <= MAX_DIMS, "torchvulkan [ERROR]: A strided copy has ", out_dims, " dimensions after coalescing; the copy shader supports ", MAX_DIMS, ".");
    TORCH_CHECK(fits_32bit_indexing(iter), "torchvulkan [ERROR]: A strided copy of ", numel, " elements is too large for the copy shader's 32-bit indexing.");

    DeviceContext* device = get_device_context(dst);
    ShaderKey shader_key{torchvulkan::Kernel::COPY, get_dtype_width_variant(dst.scalar_type())};
    uint32_t workgroupSizeX = get_dtype_workgroup_size(dst.scalar_type(), 1, device);

    SpecializationBuilder spd{};
    spd.push(out_dims)
        .push(workgroupSizeX);

    MetadataBuilder metadataBuilder = get_iter_metadata(iter, {1, 0});
    Metadata metadata = metadataBuilder.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(src))
        .push(get_tensor_address(dst))
        .push_metadata(&metadata)
        .push(numel);

    device->shader_manager->dispatchShader(
        shader_key,
        spd.build(),
        pcs.build(),
        /* read = */ {src},
        /* write = */ {dst},
        get_group_count(numel, workgroupSizeX), 1, 1
    );
}

void copy_host_to_vulkan(at::Tensor src, const at::Tensor& dst)
{
    if (src.scalar_type() != dst.scalar_type()) src = src.to(dst.scalar_type());
    if (!src.is_contiguous()) src = src.contiguous();

    void* dest_ptr = extract_buffer(dst.storage());
    uint64_t dest_offset = dst.storage_offset() * dst.itemsize();
    if (dst.is_contiguous()) {
        globalVulkanAllocator.copy_host_to_device(dest_ptr, dest_offset, src.data_ptr(), dst.nbytes());
        return;
    }

    at::Tensor staging = at::empty_like(dst, dst.options().memory_format(at::MemoryFormat::Contiguous));
    void* staging_ptr = extract_buffer(staging.storage());
    uint64_t staging_offset = staging.storage_offset() * dst.itemsize();
    globalVulkanAllocator.copy_host_to_device(staging_ptr, staging_offset, src.data_ptr(), dst.nbytes());
    dispatch_copy_shader(staging, dst);
}

void copy_vulkan_to_host(at::Tensor src, const at::Tensor& dst)
{
    if (src.scalar_type() != dst.scalar_type())
    {
        if (!is_dtype_supported(src, src.scalar_type()) || !is_dtype_supported(src, dst.scalar_type())) {
            // the bytes copy over unchanged; the CPU converts them
            at::Tensor host = at::empty(src.sizes(), src.options().device(at::kCPU).memory_format(at::MemoryFormat::Contiguous));
            copy_vulkan_to_host(src, host);
            dst.copy_(host);
            return;
        }

        at::Tensor temp = at::empty_like(src, src.options().dtype(dst.scalar_type()).memory_format(at::MemoryFormat::Contiguous));
        torchvulkan::dispatch_cast_shader(src, temp);
        src = std::move(temp);
    }

    if (!src.is_contiguous()) src = src.contiguous();

    void* src_ptr = extract_buffer(src.storage());
    uint64_t src_offset = src.storage_offset() * src.itemsize();

    if (dst.is_contiguous()) {
        globalVulkanAllocator.copy_device_to_host(dst.data_ptr(), src_ptr, src_offset, dst.nbytes());
        return;
    }

    at::Tensor staging = at::empty_like(dst, dst.options().memory_format(at::MemoryFormat::Contiguous));
    globalVulkanAllocator.copy_device_to_host(staging.data_ptr(), src_ptr, src_offset, staging.nbytes());
    dst.copy_(staging);
}

void copy_complex_vulkan_to_vulkan(const at::Tensor& src, const at::Tensor& dst)
{
    if (src.is_complex() && dst.is_complex()) {
        at::view_as_real(dst).copy_(at::view_as_real(src));
    } else if (dst.is_complex()) {
        at::Tensor pairs = at::view_as_real(dst);
        pairs.select(-1, 0).copy_(src);
        pairs.select(-1, 1).zero_();
    } else {
        dst.copy_(at::view_as_real(src).select(-1, 0));
    }
}

void copy_vulkan_to_vulkan(const at::Tensor& src, const at::Tensor& dst)
{
    if (src.scalar_type() != dst.scalar_type() && (src.is_complex() || dst.is_complex()))
    {
        copy_complex_vulkan_to_vulkan(src, dst);
        return;
    }

    if (src.scalar_type() != dst.scalar_type())
    {
        if (!is_dtype_supported(src, src.scalar_type()) || !is_dtype_supported(src, dst.scalar_type())) {
            TORCHVULKAN_CPU_FALLBACK("Vulkan device does not support source or destination dtype. Falling back to CPU for copy.");
            dst.copy_(src.cpu());
            return;
        }

        torchvulkan::dispatch_cast_shader(src, dst);
        return;
    }

    if (src.is_contiguous() && dst.is_contiguous())
    {
        void* src_ptr = extract_buffer(src.storage());
        uint64_t src_offset = src.storage_offset() * src.itemsize();
        void* dest_ptr = extract_buffer(dst.storage());
        uint64_t dest_offset = dst.storage_offset() * dst.itemsize();
        globalVulkanAllocator.copy_device_to_device(dest_ptr, dest_offset, src_ptr, src_offset, dst.nbytes());
        return;
    }

    dispatch_copy_shader(src, dst);
}

} // namespace

at::Tensor torchvulkan::copy_vulkan(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool /* non_blocking */)
{
    if (dst.numel() == 0) return dst;

    // as at::native::copy_, copying a tensor onto its own memory is a no-op (and vkCmdCopyBuffer forbids overlapping regions)
    if (self.is_alias_of(dst) && self.storage_offset() == dst.storage_offset() && self.sizes().equals(dst.sizes()) &&
        self.strides().equals(dst.strides()) && self.scalar_type() == dst.scalar_type() &&
        self.is_conj() == dst.is_conj() && self.is_neg() == dst.is_neg()) {
        return dst;
    }

    TORCH_CHECK(!dst.is_conj() && !dst.is_neg(), "torchvulkan [NOT IMPLEMENTED]: Copying into a conjugated or negated destination is not yet supported.");

    // a lazily conjugated or negated source copies its stored values, then the copy applies the pending op
    if (self.is_conj() || self.is_neg()) {
        if (self.device().is_cpu()) return torchvulkan::copy_vulkan(self.resolve_conj().resolve_neg(), dst, false);
        at::Tensor stored = self;
        if (stored.is_conj()) stored = at::_conj(stored);
        if (stored.is_neg()) stored = at::_neg_view(stored);
        at::Tensor resolved = stored.clone();
        at::Tensor pairs = resolved.is_complex() ? at::view_as_real(resolved) : resolved;
        if (self.is_conj()) pairs.select(-1, 1).neg_();
        if (self.is_neg()) pairs.neg_();
        return torchvulkan::copy_vulkan(resolved, dst, false);
    }

    at::Tensor src = (self.sizes() == dst.sizes()) ? self : self.expand(dst.sizes());
    bool src_vulkan = src.device().type() == c10::DeviceType::PrivateUse1;
    bool dst_vulkan = dst.device().type() == c10::DeviceType::PrivateUse1;

    // work happens on the vulkan side's device; the guard puts the caller's device back afterwards
    c10::OptionalDeviceGuard guard(dst_vulkan ? dst.device() : src.device());

    if (!src_vulkan && dst_vulkan) copy_host_to_vulkan(src.cpu(), dst);
    else if (src_vulkan && dst.device().is_cpu()) copy_vulkan_to_host(src, dst);
    else if (src_vulkan && !dst_vulkan) dst.copy_(src.cpu());
    else if (src.device() == dst.device()) copy_vulkan_to_vulkan(src, dst);
    else {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Source and destination are on different Vulkan devices. Copying through the CPU.");
        dst.copy_(src.cpu());
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

// without this, Module.to("vulkan") replaces each Parameter instead of setting .data, which unties shared weights
bool has_compatible_shallow_copy_type_vulkan(const at::Tensor& self, const at::Tensor& from)
{
    auto dense = [](const at::Tensor& t) { return t.layout() == at::kStrided && (t.is_cpu() || t.is_privateuseone()); };
    return self.unsafeGetTensorImpl()->has_compatible_shallow_copy_type(from.key_set()) || (dense(self) && dense(from));
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("_copy_from", &copy_from_vulkan);
    m.impl("_copy_from_and_resize", &copy_from_and_resize_vulkan);
    m.impl("copy_", &copy_vulkan_);
    m.impl("_local_scalar_dense", &local_scalar_dense_vulkan);
    m.impl("_has_compatible_shallow_copy_type", &has_compatible_shallow_copy_type_vulkan);
}
