#pragma once
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <deque>
#include <list>
#include <vector>
#include <mutex>
#include <unordered_map>

#include "volk.h"
#include "memory.h"

#define VK_CHECK(x)                                                                         \
    do {                                                                                    \
        VkResult vk_check_result_ = (x);                                                    \
        TORCH_CHECK(vk_check_result_ == VK_SUCCESS,                                         \
            "torchvulkan [ERROR]: ", #x, " failed with VkResult ", vk_check_result_, ".");  \
    } while (0)

inline constexpr VkComponentTypeKHR COMPONENT_TYPE_BFLOAT16 = static_cast<VkComponentTypeKHR>(1000141000);

struct CoopMatConfig {
    uint32_t m, n, k;
};

struct CoopMatParams {
    bool is_valid;
    uint32_t block_size;
    uint32_t block_size_acc;
    uint32_t workgroup_size;
    uint32_t subgroup_size;
    uint32_t warps_m;
    uint32_t warps_n;
    uint32_t warp_frags_m;
    uint32_t warp_frags_n;
    uint32_t bk;
    bool use_shared_memory;
};

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

    void clear();

    void addCoopMatConfig(VkComponentTypeKHR aType, VkComponentTypeKHR bType, VkComponentTypeKHR cType, VkComponentTypeKHR resultType, CoopMatConfig config);
    std::vector<CoopMatConfig> getCoopMatConfig(c10::ScalarType aType, c10::ScalarType bType, c10::ScalarType cType, c10::ScalarType resultType) const;
    const CoopMatParams& getCoopMatParams(c10::ScalarType dtype, const std::vector<uint32_t>& available_sizes, uint32_t M, uint32_t N);

private:
    DeviceContext* owner_ = nullptr;
    std::mutex mutex_;
    std::vector<VkCommandBuffer> commandBufferPool;
    std::vector<VkFence> fencePool;

    struct BufferPool 
    {
        std::list<VulkanBuffer*> lru; // oldest first
        std::unordered_map<size_t, std::deque<std::list<VulkanBuffer*>::iterator>> by_size; // keyed by the rounded buffer size, oldest first
    };
    BufferPool deviceBufferPool;
    BufferPool stagingBufferPool;
    BufferPool& poolFor(MemoryUsage usage);

    std::unordered_map<uint64_t, std::vector<CoopMatConfig>> coopMatCache;
    std::unordered_map<uint64_t, CoopMatParams> coopMatParamCache;
    CoopMatParams searchCoopMatParams(c10::ScalarType dtype, const std::vector<uint32_t>& available_sizes, uint32_t M, uint32_t N) const;
};