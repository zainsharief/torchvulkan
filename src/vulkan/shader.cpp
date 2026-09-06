#include "shader.h"
#include "cache.h"
#include "allocator.h"
#include "vulkan_context.h"
#include <algorithm>
#include <cstring>

VulkanShaderManager::VulkanShaderManager(DeviceContext* device)
    : device(device)
{
    dispatcher = new DAGDispatcher(device);

    metadata_buffer = new VulkanBuffer(device->allocator, device->device);
    VkResult result = metadata_buffer->createBuffer(METADATA_BUFFER_SIZE, MemoryUsage::HOST_TO_DEVICE);
    TORCH_CHECK(result == VK_SUCCESS, "torchvulkan [ERROR]: Failed to allocate metadata buffer.");
    metadata_base_ptr = metadata_buffer->data();
    metadata_offset = 0;
}

uint64_t VulkanShaderManager::registerMetadata(const Metadata& metadata)
{
    metadata_offset = (metadata_offset + 7) & ~uint64_t(7);
    if (metadata_offset + metadata.size > METADATA_BUFFER_SIZE) flush();

    uint64_t block_offset = metadata_offset;
    uint8_t* block = static_cast<uint8_t*>(metadata_base_ptr) + block_offset;
    memcpy(block, metadata.data, metadata.size);
    metadata_offset += metadata.size;

    size_t table_offset = metadata.size - metadata.count * sizeof(uint64_t);
    const uint64_t* offsets = reinterpret_cast<const uint64_t*>(static_cast<const uint8_t*>(metadata.data) + table_offset);

    uint64_t addresses[MAX_METADATA_SECTIONS] = {0};
    for (uint32_t i = 0; i < metadata.count; i++) addresses[i] = metadata_buffer->bufferAddress() + block_offset + offsets[i];
    memcpy(block + table_offset, addresses, metadata.count * sizeof(uint64_t));

    return metadata_buffer->bufferAddress() + block_offset + table_offset;
}

void VulkanShaderManager::flush()
{
    if (metadata_offset > 0) metadata_buffer->flush();
    metadata_offset = 0;
    if (operations.empty()) return;

    dispatcher->dispatch(operations);

    for (OpInfo* op : operations)
    {
        for (MemoryRange* range : op->inputs) delete range;
        for (MemoryRange* range : op->outputs) delete range;
        delete op;
    }
    operations.clear();
}

void VulkanShaderManager::dispatchShader(
    torchvulkan::ShaderID shaderid, 
    SpecializationArgs specConstants,
    PushConstants pushConstants,
    at::TensorList readTensors,
    at::TensorList writeTensors,
    uint32_t groupX, uint32_t groupY, uint32_t groupZ)
{
    std::vector<MemoryRange*> inputs;
    inputs.reserve(readTensors.size());
    for (const at::Tensor& in : readTensors)
    {
        MemoryRange* range = new MemoryRange{
            reinterpret_cast<uint64_t>(in.storage().data_ptr().get_context()),
            static_cast<size_t>(in.storage_offset() * in.element_size()),
            in.nbytes()
        };
        inputs.push_back(range);
    }

    std::vector<MemoryRange*> outputs;
    outputs.reserve(writeTensors.size());
    for (const at::Tensor& out : writeTensors)
    {
        MemoryRange* range = new MemoryRange{
            reinterpret_cast<uint64_t>(out.storage().data_ptr().get_context()),
            static_cast<size_t>(out.storage_offset() * out.element_size()),
            out.nbytes()
        };
        outputs.push_back(range);
    }

    ShaderSubmitInfo* submit = allocateShader(shaderid, specConstants);

    OpInfo* info = new OpInfo{};
    info->type = OpType::COMPUTE;
    info->pipeline = submit->pipeline;
    info->pipelineLayout = submit->pipelineLayout;

    TORCH_CHECK(pushConstants.size <= MAX_PUSH_CONSTANT_BYTES, "torchvulkan [ERROR]: Push constants exceeded ", MAX_PUSH_CONSTANT_BYTES, " bytes!");
    
    info->pushConstantSize = pushConstants.size;
    if (pushConstants.size > 0) memcpy(info->pushConstantData.data(), pushConstants.data, pushConstants.size);
    
    info->groupX = groupX; info->groupY = groupY; info->groupZ = groupZ;
    info->inputs = std::move(inputs);
    info->outputs = std::move(outputs);
    operations.push_back(info);

    size_t dispatch_bytes = 0;
    for (const at::Tensor& t : readTensors) dispatch_bytes += t.storage().nbytes();
    for (const at::Tensor& t : writeTensors) dispatch_bytes += t.storage().nbytes();
    device->pending_bytes += dispatch_bytes;
    if (device->pending_bytes > DeviceContext::PENDING_BYTES_FLUSH_THRESHOLD) flush();
}

void VulkanShaderManager::dispatchCopy(
    VulkanBuffer* src, uint64_t srcOffset,
    VulkanBuffer* dst, uint64_t dstOffset,
    size_t count)
{
    OpInfo* info = new OpInfo{};
    info->type = OpType::COPY;
    info->srcBuffer = src->buffer();
    info->dstBuffer = dst->buffer();
    info->region.srcOffset = srcOffset;
    info->region.dstOffset = dstOffset;
    info->region.size = count;

    info->inputs.push_back(new MemoryRange{reinterpret_cast<uint64_t>(src), srcOffset, count});
    info->outputs.push_back(new MemoryRange{reinterpret_cast<uint64_t>(dst), dstOffset, count});
    operations.push_back(info);

    device->pending_bytes += 2 * count; // source + destination
    if (device->pending_bytes > DeviceContext::PENDING_BYTES_FLUSH_THRESHOLD) flush();
}

ShaderSubmitInfo* VulkanShaderManager::allocateShader(const torchvulkan::ShaderID shaderID, const SpecializationArgs spec)
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

VkShaderModule VulkanShaderManager::allocateShaderModule(const torchvulkan::Shader shader)
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
    if (device->device_table.vkCreateShaderModule(device->device, &shaderInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        TORCH_CHECK(false, "torchvulkan [ERROR]: Failed to create shader module.");
    }
    shaderModuleCache[static_cast<uint32_t>(shader.shaderId)] = shaderModule;
    return shaderModule;
}

VkPipelineLayout VulkanShaderManager::allocatePipelineLayout()
{
    // assume active mutex
    uint64_t layoutKey = MAX_PUSH_CONSTANT_BYTES;
    auto it = pipelineLayoutCache.find(layoutKey);
    if (it != pipelineLayoutCache.end()) return it->second;

    VkPushConstantRange pushConstant{};
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstant.offset = 0;
    pushConstant.size = MAX_PUSH_CONSTANT_BYTES;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 0;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstant;

    VkPipelineLayout pipelineLayout;
    if (device->device_table.vkCreatePipelineLayout(device->device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        TORCH_CHECK(false, "torchvulkan [ERROR]: Failed to create pipeline layout.");
    }

    pipelineLayoutCache[layoutKey] = pipelineLayout;
    return pipelineLayout;
}

ShaderSubmitInfo* VulkanShaderManager::allocatePipeline(const torchvulkan::Shader shader, const SpecializationArgs spec)
{
    // assume active mutex
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout = allocatePipelineLayout();
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
    
    if (device->device_table.vkCreateComputePipelines(device->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
        TORCH_CHECK(false, "torchvulkan [ERROR]: Failed to create compute pipeline.");
    }

    ShaderSubmitInfo* info = new ShaderSubmitInfo{ pipeline, pipelineLayout };
    return info;
}

void VulkanShaderManager::clearCache()
{
    for (auto& pair : shaderModuleCache) device->device_table.vkDestroyShaderModule(device->device, pair.second, nullptr);
    shaderModuleCache.clear();

    for (auto& pair : pipelineLayoutCache) device->device_table.vkDestroyPipelineLayout(device->device, pair.second, nullptr);
    pipelineLayoutCache.clear();

    for (auto& shaderMap : shaderCache) 
    {
        for (auto& pair : shaderMap) delete pair.second;
        shaderMap.clear();
    }
}