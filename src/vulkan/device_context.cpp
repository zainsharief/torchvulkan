#include "device_context.h"
#include "shader.h"
#include "autotuner.h"

static constexpr double VRAM_USAGE_LIMIT = 0.95;

void DeviceContext::flush()
{
    shader_manager->flush();
}

size_t DeviceContext::driverBudget()
{
    VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
    vmaGetHeapBudgets(allocator, budgets);
    double limit = support_memory_budget ? VRAM_USAGE_LIMIT : 1.0;
    return static_cast<size_t>(budgets[vram_heap_index].budget * limit);
}

size_t DeviceContext::memoryBudget()
{
    size_t budget = driverBudget();
    size_t held = held_budget.load();
    if (budget < held) return held;
    if (held != 0) held_budget.compare_exchange_strong(held, 0); // the driver has caught up with our last free
    return budget;
}

bool DeviceContext::releaseCachedBuffers(size_t size, MemoryUsage usage)
{
    // no cached buffer is in use once pending work has run
    shader_manager->flush();
    vmaSetCurrentFrameIndex(allocator, 0);
    size_t budget = memoryBudget();
    size_t driver = driverBudget();

    // on unified memory staging buffers share the heap, so once the requested pool is empty free the other one too
    MemoryUsage other = usage == MemoryUsage::DEVICE_ONLY ? MemoryUsage::HOST_TO_DEVICE : MemoryUsage::DEVICE_ONLY;
    bool freed = false;
    while (size > budget || memoryUsage() > budget - size) {
        if (!cache.releaseOldestBuffer(usage) && !cache.releaseOldestBuffer(other)) break; // cache empty
        freed = true;
    }

    if (freed) {
        held_budget.store(driver);
        vmaSetCurrentFrameIndex(allocator, 0);
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
    total_allocated_bytes.fetch_add(bytes);
    size_t peak = peak_allocated_bytes.load();
    while (now > peak && !peak_allocated_bytes.compare_exchange_weak(peak, now)) {}
    size_t reserved = reservedBytes();
    size_t peak_reserved = peak_reserved_bytes.load();
    while (reserved > peak_reserved && !peak_reserved_bytes.compare_exchange_weak(peak_reserved, reserved)) {}
}

void DeviceContext::recordFree(size_t bytes)
{
    allocated_bytes.fetch_sub(bytes);
    total_freed_bytes.fetch_add(bytes);
}

void DeviceContext::trackBuffer(VulkanBuffer* buffer)
{
    std::lock_guard<std::mutex> lock(live_mutex);
    live_buffers.insert(buffer);
}

void DeviceContext::untrackBuffer(VulkanBuffer* buffer)
{
    std::lock_guard<std::mutex> lock(live_mutex);
    live_buffers.erase(buffer);
}

DeviceContext::~DeviceContext()
{
    // finish in-flight work before freeing anything it may still use
    if (device != VK_NULL_HANDLE) device_table.vkDeviceWaitIdle(device);

    delete tuner;
    tuner = nullptr;
    delete shader_manager;
    shader_manager = nullptr;

    cache.clear();

    // Buffers still alive here belong to tensors Python never freed: interpreter shutdown doesn't free every object,
    // and a daemon thread (ComfyUI's prompt worker) keeps its frame's tensors until the process ends. Vulkan requires
    // a device's child objects to be destroyed before the device (VUID-vkDestroyDevice-device-05137) and VMA every
    // allocation before the allocator, so destroy them; their deleters return early once the context is shut down.
    std::unordered_set<VulkanBuffer*> leftover;
    {
        std::lock_guard<std::mutex> lock(live_mutex);
        leftover.swap(live_buffers);
    }
    for (VulkanBuffer* buffer : leftover) delete buffer;

    if (commandPool != VK_NULL_HANDLE) device_table.vkDestroyCommandPool(device, commandPool, nullptr);
    if (allocator != VK_NULL_HANDLE) vmaDestroyAllocator(allocator);
    if (device != VK_NULL_HANDLE) device_table.vkDestroyDevice(device, nullptr);
}
