#include "device_context.h"
#include "shader.h"
#include "autotuner.h"

#include <chrono>
#include <thread>

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
    size_t blocks_before = heapBlockBytes();

    // on unified memory staging buffers share the heap, so once the requested pool is empty free the other one too
    MemoryUsage other = usage == MemoryUsage::DEVICE_ONLY ? MemoryUsage::HOST_TO_DEVICE : MemoryUsage::DEVICE_ONLY;
    bool freed = false;
    while (size > budget || memoryUsage() > budget - size) {
        if (!cache.releaseOldestBuffer(usage) && !cache.releaseOldestBuffer(other)) break; // cache empty
        freed = true;
    }

    if (freed && support_memory_budget && heapBlockBytes() < blocks_before) {
        // RADV's budget is heap size - device usage + process usage (radv_get_memory_budget_properties). Freeing drops the
        // process usage at once but the device usage only when TTM's delayed delete runs (~15 ms on amdgpu), and a buffer
        // created while VRAM still looks full to the kernel is placed in GTT (amdgpu a755906fb2b8). So wait until
        // the driver's budget is back where it was before the free, i.e. everything freed has been reclaimed. Skipped
        // when no VkDeviceMemory was freed (sub-allocations in a block VMA keeps) or the budget is VMA's own estimate.
        const size_t slack = size_t(16) << 20; // page rounding and other processes' small allocations
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
        do {
            vmaSetCurrentFrameIndex(allocator, 0);
            if (driverBudget() + slack >= driver) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (std::chrono::steady_clock::now() < deadline);
    }
    if (freed) {
        held_budget.store(driver);
        vmaSetCurrentFrameIndex(allocator, 0);
    }

    return size <= budget && memoryUsage() <= budget - size;
}

size_t DeviceContext::heapBlockBytes()
{
    VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
    vmaGetHeapBudgets(allocator, budgets);
    return budgets[vram_heap_index].statistics.blockBytes; // every VkDeviceMemory of the heap, dedicated ones too
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

DeviceContext::~DeviceContext()
{
    delete tuner;
    tuner = nullptr;
    delete shader_manager;
    shader_manager = nullptr;

    cache.clear();

    if (commandPool != VK_NULL_HANDLE) device_table.vkDestroyCommandPool(device, commandPool, nullptr);
    if (allocator != VK_NULL_HANDLE) vmaDestroyAllocator(allocator);
    if (device != VK_NULL_HANDLE) device_table.vkDestroyDevice(device, nullptr);
}
