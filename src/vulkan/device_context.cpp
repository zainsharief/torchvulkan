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

    bool freed = false;
    while (size > budget || memoryUsage() > budget - size) {
        if (!cache.releaseOldestBuffer(usage)) break; // pool empty
        freed = true;
    }

    if (freed) {
        held_budget.store(std::max(held_budget.load(), driver));
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
    buffer->live_next_ = live_head;
    if (live_head != nullptr) live_head->live_prev_ = buffer;
    live_head = buffer;
}

void DeviceContext::untrackBuffer(VulkanBuffer* buffer)
{
    std::lock_guard<std::mutex> lock(live_mutex);
    if (buffer->live_prev_ != nullptr) buffer->live_prev_->live_next_ = buffer->live_next_;
    else live_head = buffer->live_next_;
    if (buffer->live_next_ != nullptr) buffer->live_next_->live_prev_ = buffer->live_prev_;
    buffer->live_prev_ = buffer->live_next_ = nullptr;
}

void DeviceContext::destroy()
{
    // finish in-flight work before freeing anything it may still use
    if (device != VK_NULL_HANDLE) device_table.vkDeviceWaitIdle(device);

    delete tuner;
    tuner = nullptr;
    delete shader_manager;
    shader_manager = nullptr;

    cache.clear();

    // buffers still alive here belong to tensors Python never freed
    {
        std::lock_guard<std::mutex> lock(live_mutex);
        for (VulkanBuffer* buffer = live_head; buffer != nullptr;) 
        {
            VulkanBuffer* next = buffer->live_next_;
            buffer->live_prev_ = buffer->live_next_ = nullptr;
            buffer->release();
            buffer = next;
        }
        live_head = nullptr;
    }

    if (commandPool != VK_NULL_HANDLE) device_table.vkDestroyCommandPool(device, commandPool, nullptr);
    commandPool = VK_NULL_HANDLE;
    if (allocator != VK_NULL_HANDLE) vmaDestroyAllocator(allocator);
    allocator = VK_NULL_HANDLE;
    if (device != VK_NULL_HANDLE) device_table.vkDestroyDevice(device, nullptr);
    device = VK_NULL_HANDLE;
}

DeviceContext::~DeviceContext()
{
    destroy();
}
