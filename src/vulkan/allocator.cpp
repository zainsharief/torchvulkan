#include "allocator.h"

VulkanAllocator globalVulkanAllocator;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &globalVulkanAllocator);

c10::DataPtr VulkanAllocator::allocate(size_t nbytes)
{
    if (nbytes == 0) return {nullptr, nullptr, &VulkanAllocator::deleter, c10::Device(c10::DeviceType::PrivateUse1, VulkanContext::CurrentDevice())};
    sync_memory_budget(nbytes);
    VulkanBuffer* buffer = out_of_memory_buffer(nbytes, MemoryUsage::DEVICE_ONLY);
    allocated_bytes.fetch_add(nbytes, std::memory_order_relaxed);
    return {buffer, buffer, &VulkanAllocator::deleter, c10::Device(c10::DeviceType::PrivateUse1, VulkanContext::CurrentDevice())};
}

void VulkanAllocator::deleter(void* ptr)
{    
    if (ptr == nullptr) return;
    VulkanBuffer* buffer = static_cast<VulkanBuffer*>(ptr);
    globalVulkanAllocator.allocated_bytes.fetch_sub(buffer->size(), std::memory_order_relaxed);
    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    device->cache.deleteBuffer(buffer, MemoryUsage::DEVICE_ONLY);
}

void VulkanAllocator::copy_data(void* dest, const void* src, std::size_t count) const
{
    copy_device_to_device(dest, /* dest_offset = */ 0, src, /* src_offset = */ 0, count);
}

void VulkanAllocator::copy_host_to_device(void* dest, uint64_t dest_offset, const void* src, std::size_t count) const
{
    if (count == 0) return;
    TORCH_CHECK(src != nullptr, "torchvulkan [ERROR]: Source tensor cannot be null.")
    TORCH_CHECK(dest != nullptr, "torchvulkan [ERROR]: Destination tensor cannot be null.")
    VulkanBuffer* dstBuffer = static_cast<VulkanBuffer*>(dest);
    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();

    VulkanBuffer* stagingBuffer = out_of_memory_buffer(count, MemoryUsage::HOST_TO_DEVICE);
    memcpy(stagingBuffer->data(), src, count);
    stagingBuffer->flush();

    std::unique_lock<std::mutex> lock(mutex_);
    deleteQueue.push_back(stagingBuffer);
    lock.unlock();

    device->shader_manager->dispatchCopy(stagingBuffer, /* srcOffset = */ 0, dstBuffer, dest_offset, count);
}

void VulkanAllocator::copy_device_to_host(void* dest, const void* src, uint64_t src_offset, std::size_t count) const
{
    if (count == 0) return;
    TORCH_CHECK(src != nullptr, "torchvulkan [ERROR]: Source tensor cannot be null.")
    TORCH_CHECK(dest != nullptr, "torchvulkan [ERROR]: Destination tensor cannot be null.")
    VulkanBuffer* srcBuffer = (VulkanBuffer*)(src);
    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();

    VulkanBuffer* stagingBuffer = out_of_memory_buffer(count, MemoryUsage::DEVICE_TO_HOST);
    device->shader_manager->dispatchCopy(srcBuffer, src_offset, stagingBuffer, /* dstOffset = */ 0, count);

    device->flush();
    clearResources();
    stagingBuffer->invalidate();
    memcpy(dest, stagingBuffer->data(), count);
    device->cache.deleteBuffer(stagingBuffer, MemoryUsage::DEVICE_TO_HOST);
}

void VulkanAllocator::copy_device_to_host(void* dest, VulkanBuffer* stagingBuffer, std::size_t count) const
{
    if (count == 0) return;
    TORCH_CHECK(stagingBuffer != nullptr, "torchvulkan [ERROR]: Source tensor cannot be null.")
    TORCH_CHECK(dest != nullptr, "torchvulkan [ERROR]: Destination tensor cannot be null.")
    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();

    device->flush();
    clearResources();
    stagingBuffer->invalidate();
    memcpy(dest, stagingBuffer->data(), count);
    device->cache.deleteBuffer(stagingBuffer, MemoryUsage::DEVICE_TO_HOST);
}

void VulkanAllocator::copy_device_to_device(void* dest, uint64_t dest_offset, const void* src, uint64_t src_offset, std::size_t count) const
{
    if (count == 0) return;
    TORCH_CHECK(src != nullptr, "torchvulkan [ERROR]: Source tensor cannot be null.")
    TORCH_CHECK(dest != nullptr, "torchvulkan [ERROR]: Destination tensor cannot be null.")
    VulkanBuffer* dstBuffer = static_cast<VulkanBuffer*>(dest);
    VulkanBuffer* srcBuffer = (VulkanBuffer*)(src);
    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();

    device->shader_manager->dispatchCopy(srcBuffer, src_offset, dstBuffer, dest_offset, count);
}

VulkanBuffer* VulkanAllocator::out_of_memory_buffer(size_t size, MemoryUsage usage) const
{
    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();

    VulkanBuffer* buffer = device->cache.allocateBuffer(size, usage);
    if (buffer != VK_NULL_HANDLE) return buffer;

    VkResult result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    if (allocated_bytes.load() < vram_limit.load()) {
        buffer = new VulkanBuffer(device->allocator, device->device);
        result = buffer->createBuffer(size, usage);
        if (result == VK_SUCCESS) return buffer;
        delete buffer;
        buffer = nullptr;
    }

    device->flush();
    clearResources();
    device->cache.clearCache();
    sync_memory_budget(0);
    
    if (allocated_bytes.load() < vram_limit.load()) {
        if (buffer == VK_NULL_HANDLE) buffer = new VulkanBuffer(device->allocator, device->device);
        result = buffer->createBuffer(size, usage);
        if (result == VK_SUCCESS) return buffer;
        delete buffer;
    }

    uint64_t bytes_remaining = vram_limit.load() - (allocated_bytes.load() - size);
    TORCH_CHECK(false, "torchvulkan [ERROR]: Allocation failed with error code ", result, 
        ". Tried to allocate ", size, " bytes, but there is only ", bytes_remaining, " bytes remaining");
}

void VulkanAllocator::sync_memory_budget(size_t size) const 
{
    uint32_t num_allocations = num_allocations_since_sync.fetch_add(1, std::memory_order_relaxed);
    if (size != 0 && num_allocations < SYNC_THRESHOLD) return;
    
    DeviceContext* device_context = VulkanContext::Instance().CurrentDeviceContext();
    VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
    vmaGetHeapBudgets(device_context->allocator, budgets); 

    uint32_t heap_idx = device_context->vram_heap_index;
    size_t os_budget = budgets[heap_idx].budget;
    size_t actual_vma_usage = budgets[heap_idx].usage;

    vram_limit.store(os_budget * VRAM_USAGE_LIMIT, std::memory_order_relaxed);
    allocated_bytes.store(actual_vma_usage, std::memory_order_relaxed);
    num_allocations_since_sync.store(0, std::memory_order_relaxed);
}

void VulkanAllocator::clearResources() const
{    
    std::lock_guard<std::mutex> lock(mutex_);
    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    for (size_t i = 0; i < deleteQueue.size(); i++) {
        VulkanBuffer* buffer = deleteQueue[i];
        device->cache.deleteBuffer(buffer, buffer->usage());
    }

    deleteQueue.clear();
}