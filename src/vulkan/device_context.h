#pragma once
#include <ATen/core/Generator.h>
#include <c10/core/Device.h>
#include <volk.h>

#include <atomic>

#include "vk_mem_alloc.h"
#include "cache.h"

class VulkanShaderManager;

struct DeviceContext {
    c10::DeviceIndex index = -1;
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    uint32_t computeQueueFamily = UINT32_MAX;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;
    uint32_t vram_heap_index = 0;
    VolkDeviceTable device_table{};
    VkPhysicalDeviceProperties properties{};
    VkDriverId driver_id = static_cast<VkDriverId>(0);
    VulkanShaderManager* shader_manager = nullptr;
    VulkanCache cache;
    at::Generator generator;
    bool valid = true;

    bool support_float64 = false;
    bool support_int64 = false;
    bool support_float32 = false;
    bool support_int32 = false;
    bool support_float16 = false;
    bool support_bfloat16 = false;
    bool support_int16 = false;
    bool support_int8 = false;

    bool support_coopmat = false;
    bool support_bfloat16_coopmat = false;
    bool support_subgroup_control = false;
    bool support_subgroup_extended_types = false;
    bool support_subgroup_arithmetic = false;
    bool support_pipeline_statistics = false;
    bool support_memory_budget = false;
    uint32_t subgroup_size = 0;

    std::atomic<size_t> allocated_bytes{0};
    std::atomic<size_t> peak_allocated_bytes{0};

    DeviceContext() { cache.setOwner(this); }
    DeviceContext(const DeviceContext&) = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    ~DeviceContext();

    size_t memoryBudget();
    size_t memoryUsage();
    size_t memoryTotal();
    size_t memoryFree();
    bool releaseCachedBuffers(size_t size, MemoryUsage usage);
    void recordAllocation(size_t bytes);
    void recordFree(size_t bytes);
    void flush();

private:
    std::atomic<size_t> held_budget{0};
    size_t driverBudget();
};