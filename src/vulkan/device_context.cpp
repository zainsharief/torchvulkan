#include "device_context.h"

VkCommandBuffer DeviceContext::getCommandBuffer()
{
    if (cmd != VK_NULL_HANDLE) return cmd;

    cmd = cache.allocateCommandBuffer(commandPool);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    device_table.vkBeginCommandBuffer(cmd, &beginInfo);

    return cmd;
}

void DeviceContext::flush()
{
    pending_bytes = 0;
    if (cmd == VK_NULL_HANDLE) return;
        
    device_table.vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    VkFence fence = cache.allocateFence();

    std::unique_lock<std::mutex> lock(mutex_);
    VkResult submit_result = device_table.vkQueueSubmit(computeQueue, 1, &submitInfo, fence);
    lock.unlock();

    cache.deleteCommandBuffer(cmd);
    cmd = VK_NULL_HANDLE;

    TORCH_CHECK(submit_result == VK_SUCCESS, "torchvulkan [ERROR]: vkQueueSubmit failed with VkResult ", submit_result);

    VkResult wait_result = device_table.vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    if (wait_result != VK_SUCCESS) {
        TORCH_CHECK(false, "torchvulkan [ERROR]: Wait for fence failed with VkResult ", wait_result);
    }

    cache.deleteFence(fence);
}

DeviceContext::~DeviceContext() 
{
    cache.clearCache();

    if (commandPool != VK_NULL_HANDLE) device_table.vkDestroyCommandPool(device, commandPool, nullptr);
    if (allocator != VK_NULL_HANDLE) vmaDestroyAllocator(allocator);
    if (device != VK_NULL_HANDLE) device_table.vkDestroyDevice(device, nullptr);
}
