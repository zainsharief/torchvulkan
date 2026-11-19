#pragma once
#include <atomic>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <atomic>
#include <deque>
#include <list>
#include <vector>
#include <mutex>
#include <string>
#include <unordered_map>

#include "volk.h"
#include "memory.h"

#define VK_CHECK(x)                                                                         \
    do {                                                                                    \
        VkResult vk_check_result_ = (x);                                                    \
        TORCH_CHECK(vk_check_result_ == VK_SUCCESS,                                         \
            "torchvulkan [ERROR]: ", #x, " failed with VkResult ", vk_check_result_, ".");  \
    } while (0)

class VulkanCache {
public:
    void setOwner(DeviceContext* owner) { owner_ = owner; }

    VkCommandBuffer allocateCommandBuffer();
    void deleteCommandBuffer(VkCommandBuffer commandBuffer);

    VkFence allocateFence();
    void deleteFence(VkFence fence);

    static size_t roundBufferSize(size_t size);
    VulkanBuffer* allocateBuffer(size_t size, MemoryUsage usage);
    void deleteBuffer(VulkanBuffer* buffer);
    bool releaseOldestBuffer(MemoryUsage usage);
    void releaseBuffers();
    size_t cachedDeviceBytes() const { return cached_device_bytes.load(); }

    void clear();

    std::string cachePath(const char* kind, const char* extension) const;
    void createPipelineCache();
    VkPipelineCache pipelineCache() const { return pipelineCache_; }
    void markPipelineCacheDirty() { pipelineCacheDirty_ = true; }
    void savePipelineCache();

private:
    DeviceContext* owner_ = nullptr;
    std::mutex mutex_;
    std::vector<VkCommandBuffer> commandBufferPool;
    std::vector<VkFence> fencePool;

    std::mutex pipeline_mutex_;
    VkPipelineCache pipelineCache_ = VK_NULL_HANDLE;
    std::atomic<bool> pipelineCacheDirty_{false};

    struct BufferPool 
    {
        std::list<VulkanBuffer*> lru; // oldest first
        std::unordered_map<size_t, std::deque<std::list<VulkanBuffer*>::iterator>> by_size; // keyed by the rounded buffer size, oldest first
    };
    BufferPool deviceBufferPool;
    BufferPool stagingBufferPool;
    std::atomic<size_t> cached_device_bytes{0};
    void countCached(const VulkanBuffer* buffer, bool added);
    BufferPool& poolFor(MemoryUsage usage);
};