#include "allocator.h"
#include "shader.h"

#include <cstring>
#include <memory>

VulkanAllocator globalVulkanAllocator;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &globalVulkanAllocator);

bool VulkanAllocator::initialized()
{
    return VulkanContext::IsInitialized();
}

void VulkanAllocator::emptyCache(c10::MempoolId_t /* mempool_id */)
{
    for (DeviceContext* device : VulkanContext::Instance().devices) {
        for (MemoryUsage usage : {MemoryUsage::DEVICE_ONLY, MemoryUsage::HOST_TO_DEVICE, MemoryUsage::DEVICE_TO_HOST}) device->releaseCachedBuffers(SIZE_MAX, usage);
    }
}

void VulkanAllocator::recordStream(const c10::DataPtr& /* ptr */, c10::Stream /* stream */) {}

c10::CachingDeviceAllocator::DeviceStats VulkanAllocator::getDeviceStats(c10::DeviceIndex index)
{
    DeviceContext* device = VulkanContext::Instance().GetDeviceContext(index);
    c10::CachingDeviceAllocator::DeviceStats stats;
    auto aggregate = static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
    for (c10::CachingAllocator::StatArray* stat : {&stats.allocated_bytes, &stats.active_bytes, &stats.requested_bytes}) {
        c10::CachingAllocator::Stat& s = (*stat)[aggregate];
        s.current = static_cast<int64_t>(device->allocated_bytes.load());
        s.peak = static_cast<int64_t>(device->peak_allocated_bytes.load());
        s.allocated = static_cast<int64_t>(device->total_allocated_bytes.load());
        s.freed = static_cast<int64_t>(device->total_freed_bytes.load());
    }
    c10::CachingAllocator::Stat& reserved = stats.reserved_bytes[aggregate];
    reserved.current = static_cast<int64_t>(device->reservedBytes());
    reserved.peak = std::max(reserved.current, static_cast<int64_t>(device->peak_reserved_bytes.load()));
    return stats;
}

void VulkanAllocator::resetAccumulatedStats(c10::DeviceIndex index)
{
    DeviceContext* device = VulkanContext::Instance().GetDeviceContext(index);
    device->total_allocated_bytes.store(0);
    device->total_freed_bytes.store(0);
}

void VulkanAllocator::resetPeakStats(c10::DeviceIndex index)
{
    DeviceContext* device = VulkanContext::Instance().GetDeviceContext(index);
    device->peak_allocated_bytes.store(device->allocated_bytes.load());
    device->peak_reserved_bytes.store(device->reservedBytes());
}

std::pair<size_t, size_t> VulkanAllocator::getMemoryInfo(c10::DeviceIndex index)
{
    DeviceContext* device = VulkanContext::Instance().GetDeviceContext(index);
    return {device->memoryFree(), device->memoryTotal()};
}

c10::DataPtr VulkanAllocator::allocate(size_t nbytes)
{
    c10::Device device(c10::DeviceType::PrivateUse1, VulkanContext::CurrentDevice());
    if (nbytes == 0) return {nullptr, nullptr, &VulkanAllocator::deleter, device};

    DeviceContext* context = VulkanContext::Instance().CurrentDeviceContext();
    VulkanBuffer* buffer = allocateBuffer(context, nbytes, MemoryUsage::DEVICE_ONLY);
    context->recordAllocation(buffer->size());
    c10::reportMemoryUsageToProfiler(buffer, static_cast<int64_t>(buffer->size()), context->allocated_bytes.load(), context->reservedBytes(), device);
    return {buffer, buffer, &VulkanAllocator::deleter, device};
}

void VulkanAllocator::deleter(void* ptr)
{
    // after shutdown the owning device is gone; the process is exiting, so leave the memory to it
    if (ptr == nullptr || VulkanContext::IsInBadFork() || VulkanContext::IsShutDown()) return;
    VulkanBuffer* buffer = static_cast<VulkanBuffer*>(ptr);
    DeviceContext* owner = buffer->owner();
    owner->recordFree(buffer->size());
    c10::reportMemoryUsageToProfiler(buffer, -static_cast<int64_t>(buffer->size()), owner->allocated_bytes.load(), owner->reservedBytes(),
                                     c10::Device(c10::DeviceType::PrivateUse1, owner->index));

    // a buffer that spilled into system RAM is freed, not cached
    if (buffer->deviceLocal()) buffer->owner()->cache.deleteBuffer(buffer);
    else buffer->owner()->shader_manager->destroyAfterFlush(buffer);
}

void VulkanAllocator::copy_data(void* dest, const void* src, std::size_t count) const
{
    copy_device_to_device(dest, /* dest_offset = */ 0, src, /* src_offset = */ 0, count);
}

namespace {

// below this a staging copy costs about as much as importing the pages
constexpr size_t HOST_IMPORT_MIN_BYTES = 4ull << 20;

// the host memory around [ptr, ptr + count) as a buffer the GPU copies to or from directly, with ptr's offset in it;
// null when the device can't import it (no VK_EXT_external_memory_host, a mapping the driver refuses, ...)
std::unique_ptr<VulkanBuffer> import_host_range(DeviceContext* device, const void* ptr, std::size_t count, uint64_t& offset)
{
    if (!device->support_external_memory_host || count < HOST_IMPORT_MIN_BYTES) return nullptr;
    const uintptr_t align = device->host_pointer_alignment;
    const uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t base = start & ~(align - 1);
    const uintptr_t end = (start + count + align - 1) & ~(align - 1);
    auto buffer = std::make_unique<VulkanBuffer>(device);
    if (buffer->importHostPointer(reinterpret_cast<void*>(base), end - base) != VK_SUCCESS) return nullptr;
    offset = start - base;
    return buffer;
}

} // namespace

void VulkanAllocator::copy_host_to_device(void* dest, uint64_t dest_offset, const void* src, std::size_t count) const
{
    if (count == 0) return;
    TORCH_CHECK(src != nullptr, "torchvulkan [ERROR]: Source tensor cannot be null.");
    TORCH_CHECK(dest != nullptr, "torchvulkan [ERROR]: Destination tensor cannot be null.");
    VulkanBuffer* dstBuffer = static_cast<VulkanBuffer*>(dest);
    DeviceContext* device = dstBuffer->owner();

    // the GPU reads the source in place; the caller may change or free it once this returns, so wait for the copy
    uint64_t offset = 0;
    if (std::unique_ptr<VulkanBuffer> host = import_host_range(device, src, count, offset)) {
        device->shader_manager->dispatchCopy(host.get(), offset, dstBuffer, dest_offset, count);
        device->flush();
        return;
    }

    VulkanBuffer* stagingBuffer = allocateBuffer(device, count, MemoryUsage::HOST_TO_DEVICE);
    std::memcpy(stagingBuffer->data(), src, count);
    stagingBuffer->flush();

    device->shader_manager->dispatchCopy(stagingBuffer, /* srcOffset = */ 0, dstBuffer, dest_offset, count, /* release_after = */ stagingBuffer);
}

void VulkanAllocator::copy_device_to_host(void* dest, const void* src, uint64_t src_offset, std::size_t count) const
{
    if (count == 0) return;
    TORCH_CHECK(src != nullptr, "torchvulkan [ERROR]: Source tensor cannot be null.");
    TORCH_CHECK(dest != nullptr, "torchvulkan [ERROR]: Destination tensor cannot be null.");
    VulkanBuffer* srcBuffer = static_cast<VulkanBuffer*>(const_cast<void*>(src));
    DeviceContext* device = srcBuffer->owner();

    // the GPU writes straight into the destination: no staging buffer and no memcpy
    uint64_t offset = 0;
    if (std::unique_ptr<VulkanBuffer> host = import_host_range(device, dest, count, offset)) {
        device->shader_manager->dispatchCopy(srcBuffer, src_offset, host.get(), offset, count);
        device->flush();
        return;
    }

    VulkanBuffer* stagingBuffer = allocateBuffer(device, count, MemoryUsage::DEVICE_TO_HOST);
    device->shader_manager->dispatchCopy(srcBuffer, src_offset, stagingBuffer, /* dstOffset = */ 0, count);

    try {
        device->flush();
    } catch (...) {
        device->cache.deleteBuffer(stagingBuffer);
        throw;
    }

    stagingBuffer->invalidate();
    std::memcpy(dest, stagingBuffer->data(), count);
    device->cache.deleteBuffer(stagingBuffer);
}

void VulkanAllocator::copy_device_to_device(void* dest, uint64_t dest_offset, const void* src, uint64_t src_offset, std::size_t count) const
{
    if (count == 0) return;
    TORCH_CHECK(src != nullptr, "torchvulkan [ERROR]: Source tensor cannot be null.");
    TORCH_CHECK(dest != nullptr, "torchvulkan [ERROR]: Destination tensor cannot be null.");
    VulkanBuffer* srcBuffer = static_cast<VulkanBuffer*>(const_cast<void*>(src));
    VulkanBuffer* dstBuffer = static_cast<VulkanBuffer*>(dest);
    DeviceContext* srcDevice = srcBuffer->owner();
    DeviceContext* dstDevice = dstBuffer->owner();

    if (srcDevice == dstDevice)
    {
        dstBuffer->owner()->shader_manager->dispatchCopy(srcBuffer, src_offset, dstBuffer, dest_offset, count);
        return;
    }

    // TODO: upgrade this to VK_EXT_external_memory_host if supported by the device
    VulkanBuffer* download = allocateBuffer(srcDevice, count, MemoryUsage::DEVICE_TO_HOST);
    VulkanBuffer* upload = nullptr;
    try {
        srcDevice->shader_manager->dispatchCopy(srcBuffer, src_offset, download, /* dstOffset = */ 0, count);
        srcDevice->flush();
        download->invalidate();

        upload = allocateBuffer(dstDevice, count, MemoryUsage::HOST_TO_DEVICE);
        std::memcpy(upload->data(), download->data(), count);
        upload->flush();
    } catch (...) {
        srcDevice->cache.deleteBuffer(download);
        throw;
    }
    srcDevice->cache.deleteBuffer(download);

    dstDevice->shader_manager->dispatchCopy(upload, /* srcOffset = */ 0, dstBuffer, dest_offset, count, /* release_after = */ upload);
}

VulkanBuffer* VulkanAllocator::createBuffer(DeviceContext* device, size_t size, MemoryUsage usage)
{
    auto buffer = std::make_unique<VulkanBuffer>(device);
    if (buffer->createBuffer(size, usage) != VK_SUCCESS) return nullptr;
    return buffer.release();
}

VulkanBuffer* VulkanAllocator::allocateBuffer(DeviceContext* device, size_t size, MemoryUsage usage)
{
    size = VulkanCache::roundBufferSize(size);
    if (VulkanBuffer* buffer = device->cache.allocateBuffer(size, usage)) return buffer;

    size_t budget = device->memoryBudget();
    size_t used = device->memoryUsage();

    if (usage != MemoryUsage::DEVICE_ONLY || used + size <= budget)
    {
        if (VulkanBuffer* buffer = createBuffer(device, size, usage)) return buffer;
    }

    if (device->releaseCachedBuffers(size, usage))
    {
        if (VulkanBuffer* buffer = createBuffer(device, size, usage)) return buffer;
    }

    // on unified memory staging buffers share the heap, so freeing the other pools can make room too
    // this is bad for discrete GPUs, but should only happen on an OOM so acceptable
    for (const MemoryUsage& m : {MemoryUsage::DEVICE_ONLY, MemoryUsage::DEVICE_TO_HOST, MemoryUsage::HOST_TO_DEVICE})
    {        
        if (m != usage && device->releaseCachedBuffers(size, m))
        {
            if (VulkanBuffer* buffer = createBuffer(device, size, usage)) return buffer;
        }
    }

    if (!VulkanContext::Instance().strictAllocationEnabled()) 
    {
        if (VulkanBuffer* buffer = createBuffer(device, size, usage))
        {
            TORCH_WARN_ONCE("torchvulkan [WARNING]: Vulkan memory may be allocated using host memory, which may cause performance degradation. Set TORCHVULKAN_STRICT_ALLOCATION=1 to raise an out-of-memory error instead.");
            return buffer;
        }
    }

    TORCH_CHECK_WITH(OutOfMemoryError, false, "torchvulkan [ERROR]: Out of memory on ", device->properties.deviceName, ". Tried to allocate ",
        size, " bytes with ", device->memoryUsage(), " of ", device->memoryBudget(), " bytes in use."
    );
}