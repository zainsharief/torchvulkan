#include "shader.h"
#include "cache.h"
#include "vulkan_context.h"

VulkanShader::VulkanShader(torchvulkan::ShaderID shaderId, SpecializationArgs spec, DeviceContext* device)
    : device(device)
{
    submitInfo = device->cache.allocateShader(shaderId, spec);
}

void VulkanShader::dispatch(
    const void* pushConstants, 
    size_t pushConstantsSize, 
    at::TensorList tensors,
    uint32_t groupX, uint32_t groupY, uint32_t groupZ) 
{
    size_t dispatch_bytes = 0;
    for (const at::Tensor& t : tensors) dispatch_bytes += t.storage().nbytes();
    if (device->pending_bytes > DeviceContext::PENDING_BYTES_FLUSH_THRESHOLD) device->flush();
    device->pending_bytes += dispatch_bytes;

    VkCommandBuffer cmd = device->getCommandBuffer();
    device->device_table.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, submitInfo->pipeline);
    
    if (pushConstantsSize > 0 && pushConstants != nullptr) 
    {
        device->device_table.vkCmdPushConstants(
            cmd, submitInfo->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 
            0, pushConstantsSize, pushConstants
        );
    }

    // we do not expect ops to have more than 8 tensors (we can change in the future)
    c10::SmallVector<VkDescriptorBufferInfo, 8> bufferInfos(tensors.size());
    c10::SmallVector<VkWriteDescriptorSet, 8> descriptorWrites(tensors.size());

    for (size_t i = 0; i < tensors.size(); i++) {
        VulkanBuffer* vkb = (VulkanBuffer*)tensors[i].storage().data_ptr().get_context();
        
        bufferInfos[i].buffer = vkb->buffer();
        bufferInfos[i].offset = tensors[i].storage_offset() * tensors[i].element_size();
        bufferInfos[i].range = VK_WHOLE_SIZE;

        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].dstSet = VK_NULL_HANDLE;
        descriptorWrites[i].dstBinding = i;
        descriptorWrites[i].dstArrayElement = 0;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pBufferInfo = &bufferInfos[i];
    }

    device->device_table.vkCmdPushDescriptorSetKHR(
        cmd, 
        VK_PIPELINE_BIND_POINT_COMPUTE, 
        submitInfo->pipelineLayout, 
        /* index = */ 0,
        static_cast<uint32_t>(descriptorWrites.size()), 
        descriptorWrites.data()
    );

    device->device_table.vkCmdDispatch(cmd, groupX, groupY, groupZ);

    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;

    device->device_table.vkCmdPipelineBarrier(
        cmd, 
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
        0, 1, &barrier, 0, nullptr, 0, nullptr
    );
}