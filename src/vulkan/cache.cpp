#include "cache.h"
#include "vulkan_context.h"

#include <iostream>

#ifdef _MSC_VER
#include <intrin.h>
static inline int __builtin_clzll(unsigned long long x) {
    unsigned long index;
    if (_BitScanReverse64(&index, x)) {
        return 63 - index;
    }
    return 64;
}
#endif

VkCommandBuffer VulkanCache::allocateCommandBuffer(VkCommandPool commandPool)
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (!commandBufferPool.empty()) { 
        VkCommandBuffer commandBuffer = commandBufferPool.back();
        commandBufferPool.pop_back();
        VK_CHECK(device_table.vkResetCommandBuffer(commandBuffer, 0));
        return commandBuffer;
    }
    lock.unlock();
    
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    VK_CHECK(device_table.vkAllocateCommandBuffers(device_, &allocInfo, &cmd));
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
        VK_CHECK(device_table.vkResetFences(device_, 1, &fence));
        return fence;
    }
    lock.unlock();

    VkFence fence;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = 0;
    VK_CHECK(device_table.vkCreateFence(device_, &fenceInfo, nullptr, &fence));
    return fence;
}

void VulkanCache::deleteFence(VkFence fence)
{
    std::lock_guard<std::mutex> lock(mutex_);
    fencePool.push_back(fence);
}

size_t VulkanCache::getBinIndex(size_t size)
{
    if (size == 0) return 0;
    return 64 - __builtin_clzll(size - 1); // builtin compiler method (count 0's)
}

size_t VulkanCache::nextPowerOf2(size_t n) 
{
    if (n <= 1) return 0; 
    
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= (n >> 16) >> 16; 
    n++;
    
    return n;
}

VulkanBuffer* VulkanCache::allocateBuffer(size_t size, MemoryUsage usage)
{
    std::vector<VulkanBuffer*>* pools = (usage == MemoryUsage::DEVICE_ONLY) ? deviceBufferPool : stagingBufferPool;
    size_t binIndex = getBinIndex(nextPowerOf2(size));
    std::lock_guard<std::mutex> lock(mutex_);

    if (!pools[binIndex].empty()) {
        VulkanBuffer* buffer = pools[binIndex].back();
        if (buffer->size() >= size) {
            pools[binIndex].pop_back();
            return buffer;
        }
    }

    size_t nextBin = binIndex + 1;
    if (nextBin < NUM_BINS && !pools[nextBin].empty()) {
        VulkanBuffer* buffer = pools[nextBin].back();
        pools[nextBin].pop_back();
        return buffer;
    }

    return VK_NULL_HANDLE;
}

void VulkanCache::deleteBuffer(VulkanBuffer* buffer, MemoryUsage usage)
{
    std::vector<VulkanBuffer*>* pools = (usage == MemoryUsage::DEVICE_ONLY) ? deviceBufferPool : stagingBufferPool;
    size_t binIndex = getBinIndex(nextPowerOf2(buffer->size()));

    std::lock_guard<std::mutex> lock(mutex_);
    pools[binIndex].push_back(buffer);
}

ShaderSubmitInfo* VulkanCache::allocateShader(const torchvulkan::ShaderID shaderID, const SpecializationArgs spec)
{
    size_t id = static_cast<std::size_t>(shaderID);
    std::lock_guard<std::mutex> lock(mutex_);
    auto& shaderMap = shaderCache[id];

    auto it = shaderMap.find(spec.packedArgs);
    if (it != shaderMap.end()) return it->second;

    torchvulkan::Shader shader = torchvulkan::getShader(shaderID);
    ShaderSubmitInfo* shaderSubmitInfo = allocatePipeline(shader, spec);
    shaderMap[spec.packedArgs] = shaderSubmitInfo;
    return shaderSubmitInfo;
}

VkShaderModule VulkanCache::allocateShaderModule(const torchvulkan::Shader shader)
{
    // assume active mutex
    auto it = shaderModuleCache.find(static_cast<uint32_t>(shader.shaderId));
    if (it != shaderModuleCache.end()) return it->second;

    const uint32_t* spvCode = shader.binaryCode;
    size_t spvSize = shader.binarySize;
    
    VkShaderModuleCreateInfo shaderInfo{};
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.codeSize = spvSize;
    shaderInfo.pCode = spvCode;

    VkShaderModule shaderModule;
    if (device_table.vkCreateShaderModule(device_, &shaderInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        TORCH_CHECK(false, "torchvulkan [ERROR]: Failed to create shader module.");
    }
    shaderModuleCache[static_cast<uint32_t>(shader.shaderId)] = shaderModule;
    return shaderModule;
}

VkDescriptorSetLayout VulkanCache::allocateDescriptorSetLayout(const torchvulkan::Shader shader)
{
    // assume active mutex
    auto it = descriptorSetLayoutCache.find(shader.numBindings);
    if (it != descriptorSetLayoutCache.end()) return it->second;

    VkDescriptorSetLayout descriptorSetLayout;
    std::vector<VkDescriptorSetLayoutBinding> bindings(shader.numBindings);
    for (uint32_t i = 0; i < shader.numBindings; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = nullptr;
    }
    
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = shader.numBindings;
    layoutInfo.pBindings = bindings.data();
    layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;

    if (device_table.vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        TORCH_CHECK(false, "torchvulkan [ERROR]: Failed to create descriptor set layout.");
    }

    descriptorSetLayoutCache[shader.numBindings] = descriptorSetLayout;
    return descriptorSetLayout;
}

VkPipelineLayout VulkanCache::allocatePipelineLayout(const torchvulkan::Shader shader)
{
    // assume active mutex
    uint64_t layoutKey = (static_cast<uint64_t>(shader.numBindings) << 32) | shader.pushConstantSize;
    auto it = pipelineLayoutCache.find(layoutKey);
    if (it != pipelineLayoutCache.end()) return it->second;

    VkPipelineLayout pipelineLayout;
    VkDescriptorSetLayout descriptorSetLayout = allocateDescriptorSetLayout(shader);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    VkPushConstantRange pushConstant{};
    if (shader.pushConstantSize > 0) {
        pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstant.offset = 0;
        pushConstant.size = shader.pushConstantSize;

        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstant;
    }

    if (device_table.vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        TORCH_CHECK(false, "torchvulkan [ERROR]: Failed to create pipeline layout.");
    }

    pipelineLayoutCache[layoutKey] = pipelineLayout;
    return pipelineLayout;
}

ShaderSubmitInfo* VulkanCache::allocatePipeline(const torchvulkan::Shader shader, const SpecializationArgs spec)
{
    // assume active mutex
    
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout = allocatePipelineLayout(shader);
    VkShaderModule shaderModule = allocateShaderModule(shader);

    std::vector<VkSpecializationMapEntry> mapEntries(spec.numConstants);
    size_t totalSize = 0;
    for (size_t i = 0; i < spec.numConstants; i++)
    {
        mapEntries[i].constantID = i;
        mapEntries[i].offset = spec.offsets[i];
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

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = pipelineLayout;
    
    if (device_table.vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
        TORCH_CHECK(false, "torchvulkan [ERROR]: Failed to create compute pipeline.");
    }

    ShaderSubmitInfo* info = new ShaderSubmitInfo{ pipeline, pipelineLayout };
    return info;
}

inline c10::ScalarType component_to_torch_dtype(VkComponentTypeKHR dtype) 
{
    switch(dtype) {
        case VK_COMPONENT_TYPE_FLOAT16_KHR: return at::ScalarType::Half;
        case VK_COMPONENT_TYPE_FLOAT32_KHR: return at::ScalarType::Float;
        case VK_COMPONENT_TYPE_FLOAT64_KHR: return at::ScalarType::Double;
        case VK_COMPONENT_TYPE_SINT8_KHR: return at::ScalarType::Char;
        case VK_COMPONENT_TYPE_SINT16_KHR: return at::ScalarType::Short;
        case VK_COMPONENT_TYPE_SINT32_KHR: return at::ScalarType::Int;
        case VK_COMPONENT_TYPE_SINT64_KHR: return at::ScalarType::Long;
        case VK_COMPONENT_TYPE_UINT8_KHR: return at::ScalarType::Byte;
        case VK_COMPONENT_TYPE_UINT16_KHR: return at::ScalarType::UInt16;
        case VK_COMPONENT_TYPE_UINT32_KHR: return at::ScalarType::UInt32;
        case VK_COMPONENT_TYPE_UINT64_KHR: return at::ScalarType::UInt64;
        default: return c10::ScalarType::Undefined;
    }
}

void VulkanCache::addCoopMatConfig(VkComponentTypeKHR aType, VkComponentTypeKHR bType, VkComponentTypeKHR cType, VkComponentTypeKHR resultType, CoopMatConfig config)
{
    auto comp_to_scalar = [](VkComponentTypeKHR dtype) -> uint8_t {
        switch(dtype) {
            case VK_COMPONENT_TYPE_FLOAT16_KHR: return static_cast<uint8_t>(at::ScalarType::Half);
            case VK_COMPONENT_TYPE_FLOAT32_KHR: return static_cast<uint8_t>(at::ScalarType::Float);
            case VK_COMPONENT_TYPE_FLOAT64_KHR: return static_cast<uint8_t>(at::ScalarType::Double);
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
    };
    
    uint32_t key = (comp_to_scalar(aType) << 24) | (comp_to_scalar(bType) << 16) | (comp_to_scalar(cType) << 8) | comp_to_scalar(resultType);
    coopMatCache[key].push_back(config);
}

std::vector<CoopMatConfig> VulkanCache::getCoopMatConfig(c10::ScalarType aType, c10::ScalarType bType, c10::ScalarType cType, c10::ScalarType resultType) const
{
    uint32_t key = (static_cast<uint8_t>(aType) << 24) | (static_cast<uint8_t>(bType) << 16) | (static_cast<uint8_t>(cType) << 8) | static_cast<uint8_t>(resultType);
    auto it = coopMatCache.find(key);
    if (it != coopMatCache.end()) return it->second;
    else return {};
}

void VulkanCache::softClearCache()
{
    if (device_ == VK_NULL_HANDLE) return;
    DeviceContext* ctx = VulkanContext::Instance().CurrentDeviceContext();

    for (VkFence fence : fencePool) device_table.vkDestroyFence(device_, fence, nullptr);
    fencePool.clear();

    for (size_t i = 0; i < NUM_BINS; ++i) {
        for (VulkanBuffer* buffer : deviceBufferPool[i]) {
            delete buffer;
        }
        deviceBufferPool[i].clear();

        for (VulkanBuffer* buffer : stagingBufferPool[i]) {
            delete buffer;
        }
        stagingBufferPool[i].clear();
    }

    if (!commandBufferPool.empty()) {
        device_table.vkFreeCommandBuffers(device_, ctx->commandPool, static_cast<uint32_t>(commandBufferPool.size()), commandBufferPool.data());
        commandBufferPool.clear();
    }
}

void VulkanCache::clearCache()
{
    if (device_ == VK_NULL_HANDLE) return;

    softClearCache();

    for (auto& pair : shaderModuleCache) device_table.vkDestroyShaderModule(device_, pair.second, nullptr);
    shaderModuleCache.clear();

    for (auto& pair : descriptorSetLayoutCache) device_table.vkDestroyDescriptorSetLayout(device_, pair.second, nullptr);
    descriptorSetLayoutCache.clear();

    for (auto& pair : pipelineLayoutCache) device_table.vkDestroyPipelineLayout(device_, pair.second, nullptr);
    pipelineLayoutCache.clear();

    for (auto& shaderMap : shaderCache) {
        for (auto& pair : shaderMap) delete pair.second;
        shaderMap.clear(); 
    }
}