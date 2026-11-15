#pragma once
#include <torch/extension.h>
#include <c10/core/Allocator.h>
#include <c10/core/CachingDeviceAllocator.h>

#include "vulkan_context.h"
#include "memory.h"

class VulkanAllocator : public c10::DeviceAllocator {
public:
    bool initialized() override;
    void emptyCache(c10::MempoolId_t mempool_id = {0, 0}) override;
    void recordStream(const c10::DataPtr& ptr, c10::Stream stream) override;
    c10::CachingDeviceAllocator::DeviceStats getDeviceStats(c10::DeviceIndex device) override;
    void resetAccumulatedStats(c10::DeviceIndex device) override;
    void resetPeakStats(c10::DeviceIndex device) override;
    std::pair<size_t, size_t> getMemoryInfo(c10::DeviceIndex device) override;

    /**
     * @warning PyTorch expects `c10::DataPtr` to hold a raw byte
     * pointer. Because Vulkan memory is opaque to the CPU, we store the `VulkanBuffer*`
     * instance in BOTH the `data` and `ctx` fields. When extracting the buffer elsewhere,
     * use `extract_buffer(tensor.storage())`, which reads the `data` field.
     */
    c10::DataPtr allocate(size_t nbytes) override;
    static void deleter(void* ptr);

    void copy_data(void* dest, const void* src, std::size_t count) const override;
    void copy_host_to_device(void* dest, uint64_t dest_offset, const void* src, std::size_t count) const;
    void copy_device_to_host(void* dest, const void* src, uint64_t src_offset, std::size_t count) const;
    void copy_device_to_device(void* dest, uint64_t dest_offset, const void* src, uint64_t src_offset, std::size_t count) const;

    // a buffer from the device's cache, or a new one; frees cached memory and retries before running out
    static VulkanBuffer* allocateBuffer(DeviceContext* device, size_t size, MemoryUsage usage);

private:
    static VulkanBuffer* createBuffer(DeviceContext* device, size_t size, MemoryUsage usage);
};

extern VulkanAllocator globalVulkanAllocator;