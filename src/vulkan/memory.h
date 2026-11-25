#pragma once
#include <volk.h>
#include <c10/core/Storage.h>
#include <c10/core/impl/COW.h>

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
    void release();

private:
    friend struct DeviceContext;
    VulkanBuffer* live_prev_ = nullptr;
    VulkanBuffer* live_next_ = nullptr;

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

inline c10::DeleterFnPtr& storage_slice_deleter()
{
    static c10::DeleterFnPtr deleter = nullptr;
    return deleter;
}

inline bool is_storage_slice(const c10::DataPtr& ptr)
{
    return storage_slice_deleter() != nullptr && ptr.get_deleter() == storage_slice_deleter() && ptr.get_context() != nullptr;
}

inline VulkanBuffer* extract_buffer(const c10::Storage& storage)
{
    const c10::DataPtr& ptr = storage.data_ptr();
    if (is_storage_slice(ptr)) return extract_buffer(c10::Storage(c10::intrusive_ptr<c10::StorageImpl>::reclaim_copy(static_cast<c10::StorageImpl*>(ptr.get_context()))));
    return static_cast<VulkanBuffer*>(ptr.get());
}

inline uint64_t extract_buffer_offset(const c10::Storage& storage)
{
    const c10::DataPtr& ptr = storage.data_ptr();
    if (!is_storage_slice(ptr)) return 0;
    auto* parent = static_cast<c10::StorageImpl*>(ptr.get_context());
    c10::Storage parent_storage(c10::intrusive_ptr<c10::StorageImpl>::reclaim_copy(parent));
    uint64_t start = static_cast<uint64_t>(static_cast<const char*>(ptr.get()) - static_cast<const char*>(parent->data_ptr().get()));
    return extract_buffer_offset(parent_storage) + start;
}