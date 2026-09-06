#pragma once
#include <volk.h>
#include <ATen/ATen.h>

#include "shaders/shader_registry.h"
#include "builders.h"
#include "dispatch.h"

class DeviceContext;
class VulkanBuffer;

struct ShaderSubmitInfo {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
};

struct SpecializationArgs {
    const void* data = nullptr;
    const size_t* offsets = nullptr;
    const size_t* sizes = nullptr;
    const uint32_t numConstants = 0;
    const uint64_t packedArgs = 0;
};

struct MemoryRange {
    uint64_t base_addr;
    size_t byte_offset;
    size_t byte_size;
    bool overlaps(const MemoryRange& other) const;
};

enum OpType {
    COMPUTE,
    COPY
};

struct OpInfo {
    OpType type;

    // compute
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    std::array<uint8_t, MAX_PUSH_CONSTANT_BYTES> pushConstantData{};
    size_t pushConstantSize = 0;
    uint32_t groupX, groupY, groupZ;

    // copy
    VkBuffer srcBuffer = VK_NULL_HANDLE;
    VkBuffer dstBuffer = VK_NULL_HANDLE;
    VkBufferCopy region{};

    std::vector<MemoryRange*> inputs;
    std::vector<MemoryRange*> outputs;
};

class VulkanShaderManager
{
public:
    VulkanShaderManager(
        DeviceContext* device
    );

    uint64_t registerMetadata(
        const Metadata& metadata
    );

    void dispatchShader(
        torchvulkan::ShaderID shaderid, 
        SpecializationArgs specConstants,
        PushConstants pushConstants,
        at::TensorList readTensors,
        at::TensorList writeTensors,
        uint32_t groupX, uint32_t groupY, uint32_t groupZ
    );

    void dispatchCopy(
        VulkanBuffer* src, uint64_t srcOffset,
        VulkanBuffer* dst, uint64_t dstOffset,
        size_t count
    );

    void flush();

    ~VulkanShaderManager() { clearCache(); delete metadata_buffer; delete dispatcher; };

private:
    std::mutex mutex_;
    DeviceContext* device = nullptr;
    std::vector<OpInfo*> operations;
    DAGDispatcher* dispatcher;

    static const uint64_t METADATA_BUFFER_SIZE = 1024 * 1024; // 1MB
    VulkanBuffer* metadata_buffer;
    void* metadata_base_ptr;
    uint64_t metadata_offset;

    void clearCache();
    ShaderSubmitInfo* allocateShader(const torchvulkan::ShaderID shaderID, const SpecializationArgs spec);
    VkShaderModule allocateShaderModule(const torchvulkan::Shader shader);
    VkPipelineLayout allocatePipelineLayout();
    ShaderSubmitInfo* allocatePipeline(const torchvulkan::Shader shader, const SpecializationArgs spec);
    std::array<std::unordered_map<uint64_t, ShaderSubmitInfo*>, static_cast<std::size_t>(torchvulkan::ShaderID::SHADER_COUNT)> shaderCache{};
    std::unordered_map<uint64_t, VkPipelineLayout> pipelineLayoutCache;
    std::unordered_map<uint64_t, VkShaderModule> shaderModuleCache;
};