#pragma once
#include <c10/core/Device.h>
#include <volk.h>

#include "vk_mem_alloc.h"
#include "cache.h"

struct DeviceContext {
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    uint32_t computeQueueFamily = UINT32_MAX;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;
    uint32_t vram_heap_index = 0;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VolkDeviceTable device_table;
    VkPhysicalDeviceProperties properties{};
    VulkanCache cache;
    std::mutex mutex_;
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
    bool support_subgroup_control = false;
    bool support_subgroup_extended_types = false;
    bool support_subgroup_arithmetic = false;
    bool support_buffer_device_address = false;
    uint32_t subgroup_size = 0;

    static constexpr size_t PENDING_BYTES_FLUSH_THRESHOLD = 256ull << 20;
    size_t pending_bytes = 0;

    VkCommandBuffer getCommandBuffer();
    void flush();
    ~DeviceContext();
};