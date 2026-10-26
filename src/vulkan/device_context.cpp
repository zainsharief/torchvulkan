#include "device_context.h"
#include "shader.h"

static constexpr double VRAM_USAGE_LIMIT = 0.95;

void DeviceContext::flush()
{
    shader_manager->flush();
}

size_t DeviceContext::memoryBudget()
{
    VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
    vmaGetHeapBudgets(allocator, budgets);
    double limit = support_memory_budget ? VRAM_USAGE_LIMIT : 1.0;
    return static_cast<size_t>(budgets[vram_heap_index].budget * limit);
}

bool DeviceContext::releaseCachedBuffers(size_t size, MemoryUsage usage)
{
    // no cached buffer is in use once pending work has run
    shader_manager->flush();
    vmaSetCurrentFrameIndex(allocator, 0);
    size_t budget = memoryBudget();

    // on unified memory staging buffers share the heap, so once the requested pool is empty free the other one too
    MemoryUsage other = usage == MemoryUsage::DEVICE_ONLY ? MemoryUsage::HOST_TO_DEVICE : MemoryUsage::DEVICE_ONLY;
    while (size > budget || memoryUsage() > budget - size) {
        if (!cache.releaseOldestBuffer(usage) && !cache.releaseOldestBuffer(other)) break; // cache empty
    }

    return size <= budget && memoryUsage() <= budget - size;
}

size_t DeviceContext::memoryUsage()
{
    VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
    vmaGetHeapBudgets(allocator, budgets);
    return budgets[vram_heap_index].usage;
}

size_t DeviceContext::memoryTotal()
{
    const VkPhysicalDeviceMemoryProperties* props = nullptr;
    vmaGetMemoryProperties(allocator, &props);
    return props->memoryHeaps[vram_heap_index].size;
}

size_t DeviceContext::memoryFree()
{
    vmaSetCurrentFrameIndex(allocator, 0);
    size_t budget = memoryBudget();
    size_t used = memoryUsage();
    return budget > used ? budget - used : 0;
}

void DeviceContext::recordAllocation(size_t bytes)
{
    size_t now = allocated_bytes.fetch_add(bytes) + bytes;
    size_t peak = peak_allocated_bytes.load();
    while (now > peak && !peak_allocated_bytes.compare_exchange_weak(peak, now)) {}
}

void DeviceContext::recordFree(size_t bytes)
{
    allocated_bytes.fetch_sub(bytes);
}

DeviceContext::~DeviceContext()
{
    delete shader_manager;
    shader_manager = nullptr;

    cache.clear();

    if (commandPool != VK_NULL_HANDLE) device_table.vkDestroyCommandPool(device, commandPool, nullptr);
    if (allocator != VK_NULL_HANDLE) vmaDestroyAllocator(allocator);
    if (device != VK_NULL_HANDLE) device_table.vkDestroyDevice(device, nullptr);
}
