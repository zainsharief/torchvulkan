#pragma once
#include <volk.h>
#include <ATen/ATen.h>

#include "generated/shader_kernels.h"
#include "builders.h"
#include "dispatch.h"

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct DeviceContext;
class VulkanBuffer;

// largest power of two under the maximum vulkan workgroup size of 65535 (must match indexing.slang)
constexpr uint32_t GROUP_ROW = 32768;

struct ShaderKey {
    torchvulkan::Kernel kernel;
    c10::ScalarType dtype;
    c10::ScalarType dtype2 = c10::ScalarType::Undefined;
};

struct ShaderSubmitInfo {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
};

struct SpecializationKeyHash {
    using is_transparent = void;
    size_t operator()(std::string_view key) const { return std::hash<std::string_view>{}(key); }
};

struct MemoryRange {
    uint64_t base_addr;
    size_t byte_offset;
    size_t byte_size;
};

enum OpType {
    COMPUTE,
    COPY
};

struct OpInfo {
    OpType type = OpType::COMPUTE;

    // compute
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    std::array<uint8_t, MAX_PUSH_CONSTANT_BYTES> pushConstantData{};
    size_t pushConstantSize = 0;
    uint32_t groupX = 1, groupY = 1, groupZ = 1;

    // copy
    VkBuffer srcBuffer = VK_NULL_HANDLE;
    VkBuffer dstBuffer = VK_NULL_HANDLE;
    VkBufferCopy region{};

    std::vector<MemoryRange> inputs;
    std::vector<MemoryRange> outputs;
};

class VulkanShaderManager
{
public:
    explicit VulkanShaderManager(DeviceContext* device);
    ~VulkanShaderManager();

    void dispatchShader(
        ShaderKey key,
        const SpecializationArgs& specConstants,
        const PushConstants& pushConstants,
        at::TensorList readTensors,
        at::TensorList writeTensors,
        uint32_t groupX, uint32_t groupY, uint32_t groupZ
    );

    void dispatchCopy(
        VulkanBuffer* src, uint64_t srcOffset,
        VulkanBuffer* dst, uint64_t dstOffset,
        size_t count,
        VulkanBuffer* release_after = nullptr
    );

    void flush();
    void savePipelineCache();

private:
    DeviceContext* device = nullptr;
    DAGDispatcher dispatcher;

    std::mutex record_mutex_;
    std::vector<std::unique_ptr<OpInfo>> operations;
    std::vector<VulkanBuffer*> release_after_flush;

    static const uint64_t METADATA_BUFFER_SIZE = 1024 * 1024; // 1MB
    std::unique_ptr<VulkanBuffer> metadata_buffer;
    uint64_t metadata_offset = 0;

    static constexpr size_t MIN_PENDING_BYTES_FLUSH_THRESHOLD = 256ull << 20;
    static constexpr size_t MAX_PENDING_OPS = 512;
    std::unordered_set<uint64_t> pending_buffers;
    size_t pending_bytes = 0;

    void submit();
    uint64_t registerMetadata(const Metadata& metadata);
    void trackPending(const MemoryRange& range);
    void flushIfPending();

    std::mutex cache_mutex_;
    std::vector<int32_t> binaryIndex;
    std::vector<std::unordered_map<std::string, ShaderSubmitInfo, SpecializationKeyHash, std::equal_to<>>> shaderCache;
    std::vector<VkShaderModule> shaderModuleCache;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    bool pipelineCacheDirty = false;

    uint32_t resolveBinary(ShaderKey key) const;
    ShaderSubmitInfo allocateShader(uint32_t binary, const SpecializationArgs& spec);
    VkShaderModule allocateShaderModule(uint32_t binary);
    
    VkPipeline createPipeline(uint32_t binary, const SpecializationArgs& spec);
    void displayPipelineStatistics(VkPipeline pipeline);
    void createPipelineCache();
    std::string pipelineCachePath() const;
    void clearCache();
};