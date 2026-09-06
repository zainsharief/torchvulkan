#pragma once
#include <volk.h>

#include "vk_mem_alloc.h"

enum class MemoryUsage {
    HOST_TO_DEVICE,
    DEVICE_TO_HOST,
    DEVICE_ONLY
};

class VulkanBuffer {
public:
    VulkanBuffer(VmaAllocator allocator, VkDevice device) : allocator_(allocator), device(device) {};
    ~VulkanBuffer();

    // delete copy constructors
    VulkanBuffer(const VulkanBuffer&) = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;

    VkResult createBuffer(size_t size, MemoryUsage usage);
    
    void* data() { return mappedData_; }
    size_t size() const { return size_; }
    MemoryUsage usage() const { return usage_; }
    VkBuffer buffer() const { return buffer_; }
    uint64_t bufferAddress() const { return bufferAddress_; }

    void invalidate();
    void flush();

private:
    VmaAllocator allocator_;
    VkDevice device = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VmaAllocationInfo allocInfo_{};
    VkBuffer buffer_ = VK_NULL_HANDLE;
    size_t size_;
    MemoryUsage usage_;
    void* mappedData_ = nullptr;
    uint64_t bufferAddress_ = 0;
};