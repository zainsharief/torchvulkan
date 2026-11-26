#include "cache.h"
#include "device_context.h"
#include "generated/shader_kernels.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>

VulkanCache::VulkanCache()
{
    const char* env = std::getenv("TORCHVULKAN_STAGING_CACHE_MB");
    stagingCacheCap_ = (env && env[0] ? static_cast<size_t>(std::strtoull(env, nullptr, 10)) : size_t(256)) << 20;
}

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
    switch (usage) {
        case MemoryUsage::DEVICE_ONLY: return deviceBufferPool;
        case MemoryUsage::HOST_TO_DEVICE: return uploadBufferPool;
        default: return downloadBufferPool;
    }
}

void VulkanCache::countCached(const VulkanBuffer* buffer, bool added)
{
    if (buffer->usage() != MemoryUsage::DEVICE_ONLY) 
    {
        BufferPool& pool = poolFor(buffer->usage());
        if (added) pool.bytes += buffer->size();
        else pool.bytes -= buffer->size();
        return;
    }
    if (added) cached_device_bytes.fetch_add(buffer->size());
    else cached_device_bytes.fetch_sub(buffer->size());
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
    countCached(buffer, false);
    return buffer;
}

void VulkanCache::deleteBuffer(VulkanBuffer* buffer)
{
    std::lock_guard<std::mutex> lock(mutex_);
    BufferPool& pool = poolFor(buffer->usage());
    pool.lru.push_back(buffer);
    pool.by_size[buffer->size()].push_back(std::prev(pool.lru.end()));
    countCached(buffer, true);

    if (&pool != &deviceBufferPool) {
        while (pool.bytes > stagingCacheCap_ && releaseOldestLocked(pool)) {}
    }
}

bool VulkanCache::releaseOldestBuffer(MemoryUsage usage)
{
    std::lock_guard<std::mutex> lock(mutex_);
    return releaseOldestLocked(poolFor(usage));
}

bool VulkanCache::releaseOldestLocked(BufferPool& pool)
{
    if (pool.lru.empty()) return false;

    VulkanBuffer* buffer = pool.lru.front();
    auto it = pool.by_size.find(buffer->size());
    if (it == pool.by_size.end() || *it->second.front() != buffer) return false;
    it->second.pop_front();
    if (it->second.empty()) pool.by_size.erase(it);

    pool.lru.pop_front();
    countCached(buffer, false);
    delete buffer;
    return true;
}

void VulkanCache::releaseBuffers()
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (BufferPool* pool : {&deviceBufferPool, &uploadBufferPool, &downloadBufferPool}) {
        for (VulkanBuffer* buffer : pool->lru) 
        {
            countCached(buffer, false);
            delete buffer;
        }
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

    std::lock_guard<std::mutex> pipelineLock(pipeline_mutex_);
    if (pipelineCache_ != VK_NULL_HANDLE) owner_->device_table.vkDestroyPipelineCache(owner_->device, pipelineCache_, nullptr);
    pipelineCache_ = VK_NULL_HANDLE;
}

std::string VulkanCache::cachePath(const char* kind, const char* extension) const
{
    const char* setting = std::getenv("TORCHVULKAN_PIPELINE_CACHE");
    if (setting != nullptr && std::strcmp(setting, "0") == 0) return {};

    std::filesystem::path dir;
    if (const char* custom = std::getenv("TORCHVULKAN_CACHE_DIR")) dir = custom;
#ifdef _WIN32
    else if (const char* local = std::getenv("LOCALAPPDATA")) dir = std::filesystem::path(local) / "torchvulkan";
#else
    else if (const char* xdg = std::getenv("XDG_CACHE_HOME")) dir = std::filesystem::path(xdg) / "torchvulkan";
    else if (const char* home = std::getenv("HOME")) dir = std::filesystem::path(home) / ".cache" / "torchvulkan";
#endif
    else return {};

    // the driver checks the cache header too; the name keeps one file per device and driver
    char uuid[2 * VK_UUID_SIZE + 1] = {0};
    for (int i = 0; i < VK_UUID_SIZE; i++) std::snprintf(uuid + 2 * i, 3, "%02x", owner_->properties.pipelineCacheUUID[i]);
    uint32_t info_hash = 2166136261u;  // FNV-1a
    for (unsigned char c : owner_->driver_info) info_hash = (info_hash ^ c) * 16777619u;
    char name[128];
    std::snprintf(name, sizeof(name), "%s-%04x-%04x-%s-%08x.%s", kind, owner_->properties.vendorID, owner_->properties.deviceID, uuid, info_hash, extension);
    return (dir / name).string();
}

void VulkanCache::createPipelineCache()
{
    std::vector<char> initial;
    std::string path = cachePath("pipelines", "bin");
    if (!path.empty()) {
        std::ifstream file(path, std::ios::binary);
        if (file) initial.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    }

    VkPipelineCacheCreateInfo cacheInfo{};
    cacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    cacheInfo.initialDataSize = initial.size();
    cacheInfo.pInitialData = initial.empty() ? nullptr : initial.data();

    // a driver may reject stale data, so retry with an empty cache before giving up on caching
    if (owner_->device_table.vkCreatePipelineCache(owner_->device, &cacheInfo, nullptr, &pipelineCache_) == VK_SUCCESS) return;
    cacheInfo.initialDataSize = 0;
    cacheInfo.pInitialData = nullptr;
    if (owner_->device_table.vkCreatePipelineCache(owner_->device, &cacheInfo, nullptr, &pipelineCache_) != VK_SUCCESS) pipelineCache_ = VK_NULL_HANDLE;
}

void VulkanCache::savePipelineCache()
{
    std::lock_guard<std::mutex> lock(pipeline_mutex_);
    if (pipelineCache_ == VK_NULL_HANDLE || !pipelineCacheDirty_) return;
    std::string path = cachePath("pipelines", "bin");
    if (path.empty()) return;

    size_t size = 0;
    if (owner_->device_table.vkGetPipelineCacheData(owner_->device, pipelineCache_, &size, nullptr) != VK_SUCCESS || size == 0) return;
    std::vector<char> data(size);
    if (owner_->device_table.vkGetPipelineCacheData(owner_->device, pipelineCache_, &size, data.data()) != VK_SUCCESS) return;

    // written to a temporary file and renamed, so a concurrent reader never sees half a cache
    std::error_code ec;
    std::filesystem::path target(path);
    std::filesystem::create_directories(target.parent_path(), ec);
    std::filesystem::path temp = target;
    temp += ".tmp" + std::to_string(reinterpret_cast<uintptr_t>(this));
    {
        std::ofstream file(temp, std::ios::binary | std::ios::trunc);
        if (!file) return;
        file.write(data.data(), static_cast<std::streamsize>(size));
        if (!file) return;
    }
    std::filesystem::rename(temp, target, ec);
    if (ec) std::filesystem::remove(temp, ec);
    else pipelineCacheDirty_ = false;
}
