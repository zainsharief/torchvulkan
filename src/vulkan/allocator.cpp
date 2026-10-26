#include "allocator.h"
#include "shader.h"

#include <cstring>

VulkanAllocator globalVulkanAllocator;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &globalVulkanAllocator);

c10::DataPtr VulkanAllocator::allocate(size_t nbytes)
{
    c10::Device device(c10::DeviceType::PrivateUse1, VulkanContext::CurrentDevice());
    if (nbytes == 0) return {nullptr, nullptr, &VulkanAllocator::deleter, device};

    DeviceContext* context = VulkanContext::Instance().CurrentDeviceContext();
    VulkanBuffer* buffer = allocateBuffer(context, nbytes, MemoryUsage::DEVICE_ONLY);
    context->recordAllocation(buffer->size());
    return {buffer, buffer, &VulkanAllocator::deleter, device};
}

void VulkanAllocator::deleter(void* ptr)
{
    if (ptr == nullptr || VulkanContext::IsInBadFork()) return;
    VulkanBuffer* buffer = static_cast<VulkanBuffer*>(ptr);
    buffer->owner()->recordFree(buffer->size());
    buffer->owner()->cache.deleteBuffer(buffer);
}

void VulkanAllocator::copy_data(void* dest, const void* src, std::size_t count) const
{
    copy_device_to_device(dest, /* dest_offset = */ 0, src, /* src_offset = */ 0, count);
}

void VulkanAllocator::copy_host_to_device(void* dest, uint64_t dest_offset, const void* src, std::size_t count) const
{
    if (count == 0) return;
    TORCH_CHECK(src != nullptr, "torchvulkan [ERROR]: Source tensor cannot be null.");
    TORCH_CHECK(dest != nullptr, "torchvulkan [ERROR]: Destination tensor cannot be null.");
    VulkanBuffer* dstBuffer = static_cast<VulkanBuffer*>(dest);
    DeviceContext* device = dstBuffer->owner();

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

    if (VulkanBuffer* buffer = createBuffer(device, size, usage))
    {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Vulkan memory may be allocated using host memory, which may cause performance degradation.");
        return buffer;
    }

    TORCH_CHECK_WITH(OutOfMemoryError, false, "torchvulkan [ERROR]: Out of memory on ", device->properties.deviceName, ". Tried to allocate ",
        size, " bytes with ", device->memoryUsage(), " of ", device->memoryBudget(), " bytes in use."
    );
}