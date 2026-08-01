#pragma once
#include <vector>
#include <mutex>
#include <stdexcept>
#include <map>
#include <array>

#include "volk.h"
#include "memory.h"
#include "shader.h"

#define VK_CHECK(x) \
    if (x != VK_SUCCESS) TORCH_CHECK(false, "torchvulkan [ERROR]: Vulkan error in Cache");

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
};

class VulkanCache {
public:
    void softClearCache();
    void clearCache();
    ~VulkanCache() { clearCache(); }

    VkCommandBuffer allocateCommandBuffer(VkCommandPool commandPool);
    void deleteCommandBuffer(VkCommandBuffer commandBuffer);

    VkFence allocateFence();
    void deleteFence(VkFence fence);

    VulkanBuffer* allocateBuffer(size_t size, MemoryUsage usage);
    void deleteBuffer(VulkanBuffer* buffer, MemoryUsage usage);

    ShaderSubmitInfo* allocateShader(const torchvulkan::ShaderID shaderID, const SpecializationArgs spec);

    void addCoopMatConfig(VkComponentTypeKHR aType, VkComponentTypeKHR bType, VkComponentTypeKHR cType, VkComponentTypeKHR resultType, CoopMatConfig config);
    std::vector<CoopMatConfig> getCoopMatConfig(c10::ScalarType aType, c10::ScalarType bType, c10::ScalarType cType, c10::ScalarType resultType) const;
    CoopMatParams* getCoopMatParams(c10::ScalarType dtype, const std::vector<uint32_t>& available_sizes, uint32_t M, uint32_t N);

    void setDevice(VkDevice device) { device_ = device; }
    void setDeviceTable(VolkDeviceTable& table) { device_table = table; }

private:
    VkShaderModule allocateShaderModule(const torchvulkan::Shader shader);
    VkDescriptorSetLayout allocateDescriptorSetLayout(const torchvulkan::Shader shader);
    VkPipelineLayout allocatePipelineLayout(const torchvulkan::Shader shader);
    ShaderSubmitInfo* allocatePipeline(const torchvulkan::Shader shader, const SpecializationArgs spec);

    VkDevice device_;
    VolkDeviceTable device_table;
    std::mutex mutex_;
    std::vector<VkCommandBuffer> commandBufferPool;
    std::vector<VkFence> fencePool;

    static constexpr size_t NUM_BINS = 64;
    std::vector<VulkanBuffer*> deviceBufferPool[NUM_BINS];
    std::vector<VulkanBuffer*> stagingBufferPool[NUM_BINS];
    size_t getBinIndex(size_t size);
    size_t nextPowerOf2(size_t n);

    std::array<std::unordered_map<uint64_t, ShaderSubmitInfo*>, static_cast<std::size_t>(torchvulkan::ShaderID::SHADER_COUNT)> shaderCache{};

    std::unordered_map<uint64_t, VkPipelineLayout> pipelineLayoutCache;
    std::unordered_map<uint64_t, VkDescriptorSetLayout> descriptorSetLayoutCache;
    std::unordered_map<uint64_t, VkShaderModule> shaderModuleCache;
    std::unordered_map<uint64_t, std::vector<CoopMatConfig>> coopMatCache;
    std::unordered_map<uint64_t, CoopMatParams> coopMatParamCache;
};