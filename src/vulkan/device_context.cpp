#include "device_context.h"

void DeviceContext::flush()
{
    shader_manager->flush();
}

DeviceContext::~DeviceContext() 
{
    cache.clearCache();

    delete shader_manager;
    shader_manager = nullptr;

    if (commandPool != VK_NULL_HANDLE) device_table.vkDestroyCommandPool(device, commandPool, nullptr);
    if (allocator != VK_NULL_HANDLE) vmaDestroyAllocator(allocator);
    if (device != VK_NULL_HANDLE) device_table.vkDestroyDevice(device, nullptr);
}
