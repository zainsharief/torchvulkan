#include "shader.h"
#include "generated/shader_registry.h"
#include "allocator.h"
#include "vulkan_context.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>

namespace {

constexpr size_t NUM_SCALAR_TYPES = static_cast<size_t>(c10::ScalarType::NumOptions);

size_t keySlot(torchvulkan::Kernel kernel, c10::ScalarType dtype, c10::ScalarType dtype2)
{
    size_t k = static_cast<size_t>(kernel);
    size_t d = static_cast<size_t>(dtype);
    size_t d2 = static_cast<size_t>(dtype2);
    return (k * NUM_SCALAR_TYPES + d) * NUM_SCALAR_TYPES + d2;
}

} // namespace

VulkanShaderManager::VulkanShaderManager(DeviceContext* device)
    : device(device), dispatcher(device)
{
    binaryIndex.assign(static_cast<size_t>(torchvulkan::Kernel::COUNT) * NUM_SCALAR_TYPES * NUM_SCALAR_TYPES, -1);
    for (size_t i = 0; i < torchvulkan::NUM_SHADER_INDEX; i++) {
        const torchvulkan::ShaderIndexEntry& entry = torchvulkan::SHADER_INDEX[i];
        binaryIndex[keySlot(entry.kernel, entry.dtype, entry.dtype2)] = static_cast<int32_t>(entry.binary);
    }

    shaderCache.resize(torchvulkan::NUM_SHADER_BINARIES);
    shaderModuleCache.assign(torchvulkan::NUM_SHADER_BINARIES, VK_NULL_HANDLE);

    metadata_buffer = std::make_unique<VulkanBuffer>(device);
    VkResult result = metadata_buffer->createBuffer(METADATA_BUFFER_SIZE, MemoryUsage::HOST_TO_DEVICE);
    TORCH_CHECK(result == VK_SUCCESS, "torchvulkan [ERROR]: Failed to allocate the metadata buffer.");

    VkPushConstantRange pushConstant{};
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstant.offset = 0;
    pushConstant.size = MAX_PUSH_CONSTANT_BYTES;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstant;
    VK_CHECK(device->device_table.vkCreatePipelineLayout(device->device, &pipelineLayoutInfo, nullptr, &pipelineLayout));

    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device->physicalDevice, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device->physicalDevice, &familyCount, families.data());
    if (device->computeQueueFamily < familyCount) timestampValidBits = families[device->computeQueueFamily].timestampValidBits;

    createPipelineCache();
}

VulkanShaderManager::~VulkanShaderManager()
{
    clearCache();
}

uint64_t VulkanShaderManager::registerMetadata(const Metadata& metadata)
{
    // assume active mutex
    metadata_offset = (metadata_offset + 7) & ~uint64_t(7);
    if (metadata_offset + metadata.size > METADATA_BUFFER_SIZE) submit();

    uint64_t block_offset = metadata_offset;
    uint8_t* block = static_cast<uint8_t*>(metadata_buffer->data()) + block_offset;
    std::memcpy(block, metadata.data, metadata.size);
    metadata_offset += metadata.size;

    // the trailing table holds each section's offset in the block; turn those into device addresses
    size_t table_offset = metadata.size - metadata.count * sizeof(uint64_t);
    const uint64_t* offsets = reinterpret_cast<const uint64_t*>(static_cast<const uint8_t*>(metadata.data) + table_offset);

    uint64_t addresses[MAX_METADATA_SECTIONS] = {0};
    for (uint32_t i = 0; i < metadata.count; i++) addresses[i] = metadata_buffer->bufferAddress() + block_offset + offsets[i];
    std::memcpy(block + table_offset, addresses, metadata.count * sizeof(uint64_t));

    return metadata_buffer->bufferAddress() + block_offset + table_offset;
}

void VulkanShaderManager::flush()
{
    std::lock_guard<std::mutex> lock(record_mutex_);
    submit();
}

void VulkanShaderManager::submit()
{
    // assume active mutex
    if (metadata_offset > 0) metadata_buffer->flush();
    metadata_offset = 0;
    if (operations.empty()) return;

    std::vector<std::unique_ptr<OpInfo>> batch = std::move(operations);
    std::vector<VulkanBuffer*> release = std::move(release_after_flush);
    operations.clear();
    release_after_flush.clear();
    pending_buffers.clear();
    pending_bytes = 0;

    // if this throws the GPU may still be using the staging buffers, so they are leaked rather than reused
    dispatcher.dispatch(batch);

    for (VulkanBuffer* buffer : release) device->cache.deleteBuffer(buffer);
}

OpInfo* VulkanShaderManager::buildComputeOp(
    ShaderKey key,
    uint32_t subgroupSize,
    const SpecializationArgs& specConstants,
    const PushConstants& pushConstants,
    at::TensorList readTensors,
    at::TensorList writeTensors,
    uint32_t groupX, uint32_t groupY, uint32_t groupZ)
{
    const VkPhysicalDeviceLimits& limits = device->properties.limits;
    if (groupX > limits.maxComputeWorkGroupCount[0] && groupY == 1 && groupZ == 1) {
        groupY = (groupX + GROUP_ROW - 1) / GROUP_ROW;
        groupX = GROUP_ROW;
    }
    TORCH_CHECK(groupX <= limits.maxComputeWorkGroupCount[0] && groupY <= limits.maxComputeWorkGroupCount[1] && groupZ <= limits.maxComputeWorkGroupCount[2],
        "torchvulkan [ERROR]: ", torchvulkan::KERNEL_NAMES[static_cast<size_t>(key.kernel)], " needs ", groupX, " x ", groupY, " x ", groupZ,
        " workgroups, but the device allows at most ", limits.maxComputeWorkGroupCount[0], " x ", limits.maxComputeWorkGroupCount[1], " x ",
        limits.maxComputeWorkGroupCount[2], "."
    );
    TORCH_CHECK(pushConstants.size <= MAX_PUSH_CONSTANT_BYTES, "torchvulkan [ERROR]: Push constants exceeded ", MAX_PUSH_CONSTANT_BYTES, " bytes!");

    uint32_t binary = resolveBinary(key);
    ShaderSubmitInfo submit = allocateShader(binary, subgroupSize, specConstants);

    OpInfo* info = new OpInfo();
    info->type = OpType::COMPUTE;
    info->pipeline = submit.pipeline;
    info->pipelineLayout = submit.pipelineLayout;
    info->groupX = groupX;
    info->groupY = groupY;
    info->groupZ = groupZ;
    info->pushConstantSize = pushConstants.size;
    if (pushConstants.size > 0) std::memcpy(info->pushConstantData.data(), pushConstants.data, pushConstants.size);

    auto range_of = [](const at::Tensor& t) 
    {
        return MemoryRange{
            reinterpret_cast<uint64_t>(extract_buffer(t.storage())),
            static_cast<size_t>(t.storage_offset() * t.element_size()),
            t.nbytes()
        };
    };

    info->inputs.reserve(readTensors.size());
    for (const at::Tensor& in : readTensors) info->inputs.push_back(range_of(in));
    info->outputs.reserve(writeTensors.size());
    for (const at::Tensor& out : writeTensors) info->outputs.push_back(range_of(out));
    return info;
}

void VulkanShaderManager::dispatchShader(
    ShaderKey key,
    uint32_t subgroupSize,
    const SpecializationArgs& specConstants,
    const PushConstants& pushConstants,
    at::TensorList readTensors,
    at::TensorList writeTensors,
    uint32_t groupX, uint32_t groupY, uint32_t groupZ)
{
    OpInfo* info = buildComputeOp(key, subgroupSize, specConstants, pushConstants, readTensors, writeTensors, groupX, groupY, groupZ);

    std::lock_guard<std::mutex> lock(record_mutex_);
    if (pushConstants.metadata != nullptr) {
        uint64_t address = registerMetadata(*pushConstants.metadata);
        std::memcpy(info->pushConstantData.data() + pushConstants.metadata_offset, &address, sizeof(address));
    }

    for (const MemoryRange& range : info->inputs) trackPending(range);
    for (const MemoryRange& range : info->outputs) trackPending(range);
    operations.emplace_back(info);
    flushIfPending();
}

double VulkanShaderManager::getTimestamp(
    ShaderKey key,
    uint32_t subgroupSize,
    const SpecializationArgs& specConstants,
    const PushConstants& pushConstants,
    at::TensorList readTensors,
    at::TensorList writeTensors,
    uint32_t groupX, uint32_t groupY, uint32_t groupZ)
{
    TORCH_CHECK(timestampValidBits > 0, "torchvulkan [ERROR]: the compute queue of '", device->properties.deviceName, "' does not support timestamps.");

    OpInfo* built = buildComputeOp(key, subgroupSize, specConstants, pushConstants, readTensors, writeTensors, groupX, groupY, groupZ);
    OpInfo op = *built;
    delete built;

    std::lock_guard<std::mutex> lock(record_mutex_);
    submit();

    if (pushConstants.metadata != nullptr) {
        uint64_t address = registerMetadata(*pushConstants.metadata);
        std::memcpy(op.pushConstantData.data() + pushConstants.metadata_offset, &address, sizeof(address));
        metadata_buffer->flush();
    }

    if (timestampPool == VK_NULL_HANDLE)
    {
        VkQueryPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        poolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        poolInfo.queryCount = 2;
        VK_CHECK(device->device_table.vkCreateQueryPool(device->device, &poolInfo, nullptr, &timestampPool));
    }

    VkCommandBuffer cmd = device->cache.allocateCommandBuffer();
    device->device_table.vkCmdResetQueryPool(cmd, timestampPool, 0, 2);

    VkMemoryBarrier2 before{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    before.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    before.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    before.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    before.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo beforeDep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    beforeDep.memoryBarrierCount = 1;
    beforeDep.pMemoryBarriers = &before;
    device->device_table.vkCmdPipelineBarrier2(cmd, &beforeDep);

    device->device_table.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, op.pipeline);
    if (op.pushConstantSize > 0) {
        device->device_table.vkCmdPushConstants(cmd, op.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, op.pushConstantSize, op.pushConstantData.data());
    }

    VkMemoryBarrier2 between{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    between.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    between.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    between.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    between.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo betweenDep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    betweenDep.memoryBarrierCount = 1;
    betweenDep.pMemoryBarriers = &between;

    device->device_table.vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, timestampPool, 0);
    for (uint32_t run = 0; run < TIMESTAMP_RUNS; run++)
    {
        if (run > 0) device->device_table.vkCmdPipelineBarrier2(cmd, &betweenDep);
        device->device_table.vkCmdDispatch(cmd, op.groupX, op.groupY, op.groupZ);
    }
    device->device_table.vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, timestampPool, 1);

    VkMemoryBarrier2 after{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    after.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    after.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    after.dstStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT;
    after.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    VkDependencyInfo afterDep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    afterDep.memoryBarrierCount = 1;
    afterDep.pMemoryBarriers = &after;
    device->device_table.vkCmdPipelineBarrier2(cmd, &afterDep);

    VK_CHECK(device->device_table.vkEndCommandBuffer(cmd));

    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    VkFence fence = device->cache.allocateFence();
    VkResult submit_result = device->device_table.vkQueueSubmit(device->computeQueue, 1, &submitInfo, fence);
    if (submit_result != VK_SUCCESS) {
        device->cache.deleteCommandBuffer(cmd);
        device->cache.deleteFence(fence);
        TORCH_CHECK(false, "torchvulkan [ERROR]: vkQueueSubmit failed with VkResult ", submit_result, ".");
    }

    VkResult wait_result = device->device_table.vkWaitForFences(device->device, 1, &fence, VK_TRUE, UINT64_MAX);
    TORCH_CHECK(wait_result == VK_SUCCESS, "torchvulkan [ERROR]: Wait for fence failed with VkResult ", wait_result, ".");
    device->cache.deleteCommandBuffer(cmd);
    device->cache.deleteFence(fence);
    metadata_offset = 0;

    uint64_t ticks[2] = {0, 0};
    VK_CHECK(device->device_table.vkGetQueryPoolResults(device->device, timestampPool, 0, 2, sizeof(ticks), ticks, sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT)
    );

    const uint64_t mask = timestampValidBits >= 64 ? ~uint64_t(0) : (uint64_t(1) << timestampValidBits) - 1;
    const uint64_t elapsed = (ticks[1] - ticks[0]) & mask;
    return static_cast<double>(elapsed) / TIMESTAMP_RUNS * device->properties.limits.timestampPeriod;
}

void VulkanShaderManager::trackPending(const MemoryRange& range)
{
    // assume active mutex
    if (range.base_addr == 0) return; // an empty tensor has no buffer behind it
    if (!pending_buffers.insert(range.base_addr).second) return;

    pending_bytes += reinterpret_cast<VulkanBuffer*>(range.base_addr)->size();
}

void VulkanShaderManager::flushIfPending()
{
    // assume active mutex
    bool too_many_bytes = pending_bytes > MIN_PENDING_BYTES_FLUSH_THRESHOLD && pending_bytes > device->memoryBudget() / 4;
    if (too_many_bytes || operations.size() >= MAX_PENDING_OPS) submit();
}

void VulkanShaderManager::dispatchCopy(
    VulkanBuffer* src, uint64_t srcOffset,
    VulkanBuffer* dst, uint64_t dstOffset,
    size_t count,
    VulkanBuffer* release_after)
{
    auto info = std::make_unique<OpInfo>();
    info->type = OpType::COPY;
    info->srcBuffer = src->buffer();
    info->dstBuffer = dst->buffer();
    info->region.srcOffset = srcOffset;
    info->region.dstOffset = dstOffset;
    info->region.size = count;
    info->inputs.push_back(MemoryRange{reinterpret_cast<uint64_t>(src), srcOffset, count});
    info->outputs.push_back(MemoryRange{reinterpret_cast<uint64_t>(dst), dstOffset, count});

    std::lock_guard<std::mutex> lock(record_mutex_);
    for (const MemoryRange& range : info->inputs) trackPending(range);
    for (const MemoryRange& range : info->outputs) trackPending(range);
    operations.push_back(std::move(info));
    if (release_after != nullptr) release_after_flush.push_back(release_after);
    flushIfPending();
}

uint32_t VulkanShaderManager::resolveBinary(ShaderKey key) const
{
    int32_t binary = binaryIndex[keySlot(key.kernel, key.dtype, key.dtype2)];
    if (binary >= 0) return static_cast<uint32_t>(binary);

    const char* kernel = torchvulkan::KERNEL_NAMES[static_cast<size_t>(key.kernel)];
    if (key.dtype2 == c10::ScalarType::Undefined) {
        TORCH_CHECK(false, "torchvulkan [ERROR]: no ", kernel, " shader was compiled for ", key.dtype, ".");
    }
    TORCH_CHECK(false, "torchvulkan [ERROR]: no ", kernel, " shader was compiled for ", key.dtype, " -> ", key.dtype2, ".");
}

ShaderSubmitInfo VulkanShaderManager::allocateShader(uint32_t binary, uint32_t subgroupSize, const SpecializationArgs& spec)
{
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto& shaderMap = shaderCache[binary][subgroupSize];

    auto it = shaderMap.find(spec.key);
    if (it != shaderMap.end()) return it->second;

    ShaderSubmitInfo info{createPipeline(binary, subgroupSize, spec), pipelineLayout};
    shaderMap.emplace(std::string(spec.key), info);
    return info;
}

VkShaderModule VulkanShaderManager::allocateShaderModule(uint32_t binary)
{
    // assume active mutex
    if (shaderModuleCache[binary] != VK_NULL_HANDLE) return shaderModuleCache[binary];

    const torchvulkan::ShaderBinary& shader = torchvulkan::SHADER_BINARIES[binary];

    VkShaderModuleCreateInfo shaderInfo{};
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.codeSize = shader.size;
    shaderInfo.pCode = shader.code;

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkResult result = device->device_table.vkCreateShaderModule(device->device, &shaderInfo, nullptr, &shaderModule);
    TORCH_CHECK(result == VK_SUCCESS, "torchvulkan [ERROR]: Failed to create shader module for ", shader.name, " (VkResult ", result, ").");
    shaderModuleCache[binary] = shaderModule;
    return shaderModule;
}

VkPipeline VulkanShaderManager::createPipeline(uint32_t binary, uint32_t subgroupSize, const SpecializationArgs& spec)
{
    // assume active mutex
    VkShaderModule shaderModule = allocateShaderModule(binary);

    std::vector<VkSpecializationMapEntry> mapEntries(spec.numConstants);
    size_t totalSize = 0;
    for (size_t i = 0; i < spec.numConstants; i++)
    {
        mapEntries[i].constantID = static_cast<uint32_t>(i);
        mapEntries[i].offset = static_cast<uint32_t>(spec.offsets[i]);
        mapEntries[i].size = spec.sizes[i];
        totalSize += spec.sizes[i];
    }

    VkSpecializationInfo specInfo{};
    specInfo.mapEntryCount = static_cast<uint32_t>(mapEntries.size());
    specInfo.pMapEntries = mapEntries.data();
    specInfo.dataSize = totalSize;
    specInfo.pData = spec.data;

    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule;
    stageInfo.pName = "main";
    stageInfo.pSpecializationInfo = &specInfo;

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo requiredSize{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    if (device->support_subgroup_control)
    {
        requiredSize.requiredSubgroupSize = subgroupSize;
        stageInfo.pNext = &requiredSize;
    }

    VkPipelineCreateFlags flags = 0;
    const bool capturePipelineStatistics = device->support_pipeline_statistics && VulkanContext::Instance().profilingEnabled();
    if (capturePipelineStatistics) flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.flags = flags;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult result = device->device_table.vkCreateComputePipelines(device->device, pipelineCache, 1, &pipelineInfo, nullptr, &pipeline);
    TORCH_CHECK(result == VK_SUCCESS, "torchvulkan [ERROR]: Failed to create compute pipeline for ",
        torchvulkan::SHADER_BINARIES[binary].name, " (VkResult ", result, ")."
    );
    pipelineCacheDirty = true;

    if (capturePipelineStatistics) displayPipelineStatistics(pipeline);
    return pipeline;
}

std::string VulkanShaderManager::pipelineCachePath() const
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
    for (int i = 0; i < VK_UUID_SIZE; i++) std::snprintf(uuid + 2 * i, 3, "%02x", device->properties.pipelineCacheUUID[i]);
    char name[128];
    std::snprintf(name, sizeof(name), "pipelines-%04x-%04x-%s.bin", device->properties.vendorID, device->properties.deviceID, uuid);
    return (dir / name).string();
}

void VulkanShaderManager::createPipelineCache()
{
    std::vector<char> initial;
    std::string path = pipelineCachePath();
    if (!path.empty()) {
        std::ifstream file(path, std::ios::binary);
        if (file) initial.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    }

    VkPipelineCacheCreateInfo cacheInfo{};
    cacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    cacheInfo.initialDataSize = initial.size();
    cacheInfo.pInitialData = initial.empty() ? nullptr : initial.data();

    // a driver may reject stale data, so retry with an empty cache before giving up on caching
    if (device->device_table.vkCreatePipelineCache(device->device, &cacheInfo, nullptr, &pipelineCache) == VK_SUCCESS) return;
    cacheInfo.initialDataSize = 0;
    cacheInfo.pInitialData = nullptr;
    if (device->device_table.vkCreatePipelineCache(device->device, &cacheInfo, nullptr, &pipelineCache) != VK_SUCCESS) pipelineCache = VK_NULL_HANDLE;
}

void VulkanShaderManager::savePipelineCache()
{
    std::lock_guard<std::mutex> lock(cache_mutex_);
    if (pipelineCache == VK_NULL_HANDLE || !pipelineCacheDirty) return;
    std::string path = pipelineCachePath();
    if (path.empty()) return;

    size_t size = 0;
    if (device->device_table.vkGetPipelineCacheData(device->device, pipelineCache, &size, nullptr) != VK_SUCCESS || size == 0) return;
    std::vector<char> data(size);
    if (device->device_table.vkGetPipelineCacheData(device->device, pipelineCache, &size, data.data()) != VK_SUCCESS) return;

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
    else pipelineCacheDirty = false;
}

void VulkanShaderManager::displayPipelineStatistics(VkPipeline pipeline)
{
    VkPipelineExecutableInfoKHR pipelineExecInfo{};
    pipelineExecInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR;
    pipelineExecInfo.pipeline = pipeline;
    pipelineExecInfo.executableIndex = 0;

    uint32_t statisticCount = 0;
    VK_CHECK(device->device_table.vkGetPipelineExecutableStatisticsKHR(device->device, &pipelineExecInfo, &statisticCount, nullptr));

    if (statisticCount == 0) {
        std::cerr << "torchvulkan: no pipeline statistics available." << std::endl;
        return;
    }

    std::vector<VkPipelineExecutableStatisticKHR> statistics(statisticCount);
    for (auto& stat : statistics) {
        stat.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
        stat.pNext = nullptr;
    }
    VK_CHECK(device->device_table.vkGetPipelineExecutableStatisticsKHR(device->device, &pipelineExecInfo, &statisticCount, statistics.data()));

    for (const auto& stat : statistics)
    {
        std::cerr << "Metric: " << stat.name << " (" << stat.description << ") = ";
        switch (stat.format) {
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: std::cerr << (stat.value.b32 ? "true" : "false"); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: std::cerr << stat.value.i64; break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: std::cerr << stat.value.u64; break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR: std::cerr << stat.value.f64; break;
            default: std::cerr << "Unknown format"; break;
        }
        std::cerr << " | ";
    }
    std::cerr << "\n";
}

void VulkanShaderManager::clearCache()
{
    for (auto& bySubgroup : shaderCache)
    {
        for (auto& [size, shaderMap] : bySubgroup) {
            for (auto& [key, info] : shaderMap) device->device_table.vkDestroyPipeline(device->device, info.pipeline, nullptr);
        }
        bySubgroup.clear();
    }

    for (VkShaderModule shaderModule : shaderModuleCache) {
        if (shaderModule != VK_NULL_HANDLE) device->device_table.vkDestroyShaderModule(device->device, shaderModule, nullptr);
    }
    shaderModuleCache.assign(shaderModuleCache.size(), VK_NULL_HANDLE);

    if (pipelineLayout != VK_NULL_HANDLE) device->device_table.vkDestroyPipelineLayout(device->device, pipelineLayout, nullptr);
    pipelineLayout = VK_NULL_HANDLE;
    if (pipelineCache != VK_NULL_HANDLE) device->device_table.vkDestroyPipelineCache(device->device, pipelineCache, nullptr);
    pipelineCache = VK_NULL_HANDLE;
    if (timestampPool != VK_NULL_HANDLE) device->device_table.vkDestroyQueryPool(device->device, timestampPool, nullptr);
    timestampPool = VK_NULL_HANDLE;
}
