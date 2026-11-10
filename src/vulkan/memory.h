#pragma once
#include <volk.h>
#include <c10/core/Storage.h>

#include "vk_mem_alloc.h"

struct DeviceContext;

enum class MemoryUsage {
    HOST_TO_DEVICE,
    DEVICE_TO_HOST,
    DEVICE_ONLY
};

class VulkanBuffer {
public:
    explicit VulkanBuffer(DeviceContext* owner) : owner_(owner) {}
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
    bool deviceLocal() const { return deviceLocal_; }
    DeviceContext* owner() const { return owner_; }

    void invalidate();
    void flush();

private:
    DeviceContext* owner_ = nullptr;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VmaAllocationInfo allocInfo_{};
    VkBuffer buffer_ = VK_NULL_HANDLE;
    size_t size_ = 0;
    MemoryUsage usage_ = MemoryUsage::DEVICE_ONLY;
    void* mappedData_ = nullptr;
    uint64_t bufferAddress_ = 0;
    bool deviceLocal_ = true;
};

inline VulkanBuffer* extract_buffer(const c10::Storage& storage)
{
    return static_cast<VulkanBuffer*>(storage.data_ptr().get());
}