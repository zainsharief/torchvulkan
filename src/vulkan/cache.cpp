#include "cache.h"
#include "device_context.h"
#include "generated/shader_kernels.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>

VkCommandBuffer VulkanCache::allocateCommandBuffer()
{
    VkCommandBuffer cmd = VK_NULL_HANDLE;

    // the command pool is externally synchronised, so every use of it holds the lock
    std::lock_guard<std::mutex> lock(mutex_);
    if (!commandBufferPool.empty()) {
        cmd = commandBufferPool.back();
        commandBufferPool.pop_back();
        VK_CHECK(owner_->device_table.vkResetCommandBuffer(cmd, 0));
    }
    else
    {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = owner_->commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        VK_CHECK(owner_->device_table.vkAllocateCommandBuffers(owner_->device, &allocInfo, &cmd));
    }

    // a reused buffer is no longer recording, so every buffer we hand out is begun here
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(owner_->device_table.vkBeginCommandBuffer(cmd, &beginInfo));

    return cmd;
}

void VulkanCache::deleteCommandBuffer(VkCommandBuffer commandBuffer)
{
    std::lock_guard<std::mutex> lock(mutex_);
    commandBufferPool.push_back(commandBuffer);
}

VkFence VulkanCache::allocateFence()
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (!fencePool.empty()) {
        VkFence fence = fencePool.back();
        fencePool.pop_back();
        lock.unlock();
        VK_CHECK(owner_->device_table.vkResetFences(owner_->device, 1, &fence));
        return fence;
    }
    lock.unlock();

    VkFence fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK_CHECK(owner_->device_table.vkCreateFence(owner_->device, &fenceInfo, nullptr, &fence));
    return fence;
}

void VulkanCache::deleteFence(VkFence fence)
{
    std::lock_guard<std::mutex> lock(mutex_);
    fencePool.push_back(fence);
}

size_t VulkanCache::roundBufferSize(size_t size)
{
    static constexpr size_t MIN_BUFFER_SIZE = 256;
    if (size <= MIN_BUFFER_SIZE) return MIN_BUFFER_SIZE;

    // size is in (2^k, 2^(k+1)]; round up to a multiple of 2^k / 4
    size_t k = std::bit_width(size - 1) - 1;
    size_t step = size_t(1) << (k - 2);
    return ((size + step - 1) / step) * step;
}

VulkanCache::BufferPool& VulkanCache::poolFor(MemoryUsage usage)
{
    return usage == MemoryUsage::DEVICE_ONLY ? deviceBufferPool : stagingBufferPool;
}

VulkanBuffer* VulkanCache::allocateBuffer(size_t size, MemoryUsage usage)
{
    size_t rounded = roundBufferSize(size);

    std::lock_guard<std::mutex> lock(mutex_);
    BufferPool& pool = poolFor(usage);
    auto it = pool.by_size.find(rounded);
    if (it == pool.by_size.end()) return nullptr;

    // the newest buffer of this size, so the oldest ones stay at the front of the LRU to be freed first
    auto entry = it->second.back();
    it->second.pop_back();
    if (it->second.empty()) pool.by_size.erase(it);

    VulkanBuffer* buffer = *entry;
    pool.lru.erase(entry);
    return buffer;
}

void VulkanCache::deleteBuffer(VulkanBuffer* buffer)
{
    std::lock_guard<std::mutex> lock(mutex_);
    BufferPool& pool = poolFor(buffer->usage());
    pool.lru.push_back(buffer);
    pool.by_size[buffer->size()].push_back(std::prev(pool.lru.end()));
}

bool VulkanCache::releaseOldestBuffer(MemoryUsage usage)
{
    std::lock_guard<std::mutex> lock(mutex_);
    BufferPool& pool = poolFor(usage);
    if (pool.lru.empty()) return false;

    VulkanBuffer* buffer = pool.lru.front();
    auto it = pool.by_size.find(buffer->size());
    if (it == pool.by_size.end() || *it->second.front() != buffer) return false;
    it->second.pop_front();
    if (it->second.empty()) pool.by_size.erase(it);

    pool.lru.pop_front();
    delete buffer;
    return true;
}

void VulkanCache::releaseBuffers()
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (BufferPool* pool : {&deviceBufferPool, &stagingBufferPool}) {
        for (VulkanBuffer* buffer : pool->lru) delete buffer;
        pool->lru.clear();
        pool->by_size.clear();
    }
}

void VulkanCache::clear()
{
    if (owner_ == nullptr || owner_->device == VK_NULL_HANDLE) return;
    releaseBuffers();

    std::lock_guard<std::mutex> lock(mutex_);
    for (VkFence fence : fencePool) owner_->device_table.vkDestroyFence(owner_->device, fence, nullptr);
    fencePool.clear();

    if (!commandBufferPool.empty()) {
        owner_->device_table.vkFreeCommandBuffers(owner_->device, owner_->commandPool, static_cast<uint32_t>(commandBufferPool.size()), commandBufferPool.data());
        commandBufferPool.clear();
    }
}

static uint8_t coop_component_to_scalar(VkComponentTypeKHR dtype)
{
    switch (dtype) {
        case VK_COMPONENT_TYPE_FLOAT16_KHR: return static_cast<uint8_t>(at::ScalarType::Half);
        case VK_COMPONENT_TYPE_FLOAT32_KHR: return static_cast<uint8_t>(at::ScalarType::Float);
        case VK_COMPONENT_TYPE_FLOAT64_KHR: return static_cast<uint8_t>(at::ScalarType::Double);
        case COMPONENT_TYPE_BFLOAT16:       return static_cast<uint8_t>(at::ScalarType::BFloat16);
        case VK_COMPONENT_TYPE_SINT8_KHR:   return static_cast<uint8_t>(at::ScalarType::Char);
        case VK_COMPONENT_TYPE_SINT16_KHR:  return static_cast<uint8_t>(at::ScalarType::Short);
        case VK_COMPONENT_TYPE_SINT32_KHR:  return static_cast<uint8_t>(at::ScalarType::Int);
        case VK_COMPONENT_TYPE_SINT64_KHR:  return static_cast<uint8_t>(at::ScalarType::Long);
        case VK_COMPONENT_TYPE_UINT8_KHR:   return static_cast<uint8_t>(at::ScalarType::Byte);
        case VK_COMPONENT_TYPE_UINT16_KHR:  return static_cast<uint8_t>(at::ScalarType::UInt16);
        case VK_COMPONENT_TYPE_UINT32_KHR:  return static_cast<uint8_t>(at::ScalarType::UInt32);
        case VK_COMPONENT_TYPE_UINT64_KHR:  return static_cast<uint8_t>(at::ScalarType::UInt64);
        default:                            return static_cast<uint8_t>(c10::ScalarType::Undefined);
    }
}

static uint32_t coop_config_key(uint8_t a, uint8_t b, uint8_t c, uint8_t result)
{
    return (uint32_t(a) << 24) | (uint32_t(b) << 16) | (uint32_t(c) << 8) | uint32_t(result);
}

void VulkanCache::addCoopMatConfig(VkComponentTypeKHR aType, VkComponentTypeKHR bType, VkComponentTypeKHR cType, VkComponentTypeKHR resultType, CoopMatConfig config)
{
    uint32_t key = coop_config_key(coop_component_to_scalar(aType), coop_component_to_scalar(bType), coop_component_to_scalar(cType), coop_component_to_scalar(resultType));
    coopMatCache[key].push_back(config);
}

std::vector<CoopMatConfig> VulkanCache::getCoopMatConfig(c10::ScalarType aType, c10::ScalarType bType, c10::ScalarType cType, c10::ScalarType resultType) const
{
    uint32_t key = coop_config_key(static_cast<uint8_t>(aType), static_cast<uint8_t>(bType), static_cast<uint8_t>(cType), static_cast<uint8_t>(resultType));
    auto it = coopMatCache.find(key);
    if (it != coopMatCache.end()) return it->second;
    return {};
}

const CoopMatParams& VulkanCache::getCoopMatParams(c10::ScalarType dtype, const std::vector<uint32_t>& available_sizes, uint32_t M, uint32_t N)
{
    auto bucket = [](uint32_t x) -> uint64_t {
        uint64_t b = 1;
        while (b < x && b < 256) b <<= 1;
        return b;
    };

    // available_sizes depends only on the dtype, so it does not need to be part of the key
    const uint64_t key = (uint64_t)(uint8_t)dtype | (bucket(std::max(M, 1u)) << 16) | (bucket(std::max(N, 1u)) << 32);

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = coopMatParamCache.find(key);
    if (it != coopMatParamCache.end()) return it->second;
    return coopMatParamCache.emplace(key, searchCoopMatParams(dtype, available_sizes, std::max(M, 1u), std::max(N, 1u))).first->second;
}

// picks a tiling for matmul_coop: the smallest BK whose tile fits shared memory, preferring
// tiles no more than twice the problem size, then the least padding
CoopMatParams VulkanCache::searchCoopMatParams(c10::ScalarType dtype, const std::vector<uint32_t>& available_sizes, uint32_t M, uint32_t N) const
{
    CoopMatParams params{};

    uint32_t element_size = c10::elementSize(dtype);
    uint32_t max_shared_memory = owner_->properties.limits.maxComputeSharedMemorySize;
    uint32_t max_workgroup_size = owner_->properties.limits.maxComputeWorkGroupInvocations;
    params.subgroup_size = owner_->subgroup_size;
    if (params.subgroup_size == 0) return params;

    uint32_t max_subgroups = std::min<uint32_t>(8, max_workgroup_size / params.subgroup_size);
    if (max_subgroups == 0) return params;

    static constexpr std::array<uint32_t, 4> pref_1_byte = {64, 32, 16, 8};
    static constexpr std::array<uint32_t, 4> pref_2_byte = {32, 64, 16, 8};
    static constexpr std::array<uint32_t, 4> pref_4_byte = {16, 32, 8, 64};
    static constexpr std::array<uint32_t, 4> pref_8_byte = {8, 16, 32, 64};
    const std::array<uint32_t, 4>* preferences = nullptr;

    switch (element_size)
    {
        case 1: preferences = &pref_1_byte; break;
        case 2: preferences = &pref_2_byte; break;
        case 4: preferences = &pref_4_byte; break;
        case 8: preferences = &pref_8_byte; break;
        default: return params;
    }

    for (uint32_t pref : *preferences)
    {
        if (std::find(available_sizes.begin(), available_sizes.end(), pref) == available_sizes.end()) continue;
        params.block_size = pref;
        params.block_size_acc = pref;
        break;
    }
    if (params.block_size == 0) return params;

    // MoltenVK's cooperative matrices load fastest straight from device memory; the autotuner will replace this
    params.use_shared_memory = owner_->driver_id != VK_DRIVER_ID_MOLTENVK;

    // mirrors the Shared<T> sizes in src/shaders/ops/matmul_coop.slang
    auto shared_bytes = [&](uint64_t rows_a, uint64_t cols_b, uint32_t bk) -> uint64_t
    {
        const uint64_t pad = 4; // PAD in matmul_coop.slang
        if (params.use_shared_memory) return 2 * (rows_a * (bk + pad) + (uint64_t)bk * (cols_b + pad)) * element_size;
        return (rows_a * bk + (uint64_t)bk * cols_b) * element_size;
    };

    struct Candidate {
        uint32_t subgroups, warps_m, warps_n, frags_m, frags_n;
    };

    // every way to split up to max_subgroups subgroups into a warps_m x warps_n grid, each with a fragment shape
    static constexpr uint32_t frag_preferences[4][2] = {{4, 2}, {2, 4}, {4, 4}, {2, 2}};
    std::vector<Candidate> candidates;
    for (uint32_t subgroups = max_subgroups; subgroups >= 1; subgroups /= 2)
    {
        for (uint32_t wm = (uint32_t)std::sqrt((double)subgroups); wm >= 1; --wm)
        {
            if (subgroups % wm) continue;
            for (uint32_t warps_m : {wm, subgroups / wm}) {
                for (const auto& frags : frag_preferences) candidates.push_back({subgroups, warps_m, subgroups / warps_m, frags[0], frags[1]});
            }
        }
    }

    const uint64_t overshoot_limit = 2;
    for (bool allow_overshoot : {false, true})
    {
        for (uint32_t bk = params.block_size_acc; bk <= 64; bk *= 2)
        {
            uint64_t best_waste = UINT64_MAX;
            for (const Candidate& c : candidates)
            {
                const uint64_t rows_a = (uint64_t)c.warps_m * c.frags_m * params.block_size;
                const uint64_t cols_b = (uint64_t)c.warps_n * c.frags_n * params.block_size;

                if (shared_bytes(rows_a, cols_b, bk) > max_shared_memory) continue;
                if (!allow_overshoot && (rows_a > overshoot_limit * M || cols_b > overshoot_limit * N)) continue;

                const uint64_t pm = ((M + rows_a - 1) / rows_a) * rows_a;
                const uint64_t pn = ((N + cols_b - 1) / cols_b) * cols_b;
                const uint64_t waste = allow_overshoot ? (pm * pn * 1024) / ((uint64_t)M * N) : 0;
                if (waste >= best_waste) continue;

                best_waste = waste;
                params.workgroup_size = c.subgroups * params.subgroup_size;
                params.warps_m = c.warps_m;
                params.warps_n = c.warps_n;
                params.warp_frags_m = c.frags_m;
                params.warp_frags_n = c.frags_n;
                params.bk = bk;
                params.is_valid = true;
            }
            if (params.is_valid) return params;
        }
    }

    return params;
}
