#pragma once
#include <volk.h>
#include <ATen/ATen.h>

#include "shaders/shader_registry.h"

class DeviceContext;

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

struct VulkanShader {
    ShaderSubmitInfo* submitInfo;
    DeviceContext* device = nullptr;
    
    VulkanShader(torchvulkan::ShaderID shaderid, SpecializationArgs spec, DeviceContext* device);

    void dispatch(
        const void* pushConstants, 
        size_t pushConstantsSize, 
        at::TensorList tensors,
        uint32_t groupX, uint32_t groupY, uint32_t groupZ
    );
};