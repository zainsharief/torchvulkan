#pragma once
#include <c10/core/Device.h>
#include <volk.h>
#include <vector>

#include "device_context.h"

class VulkanContext {
public:
    // singleton pattern
    static VulkanContext& Instance();
    
    // delete copy constructors
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;
    
    VkInstance instance = VK_NULL_HANDLE;
    std::vector<DeviceContext*> devices;

    DeviceContext* CurrentDeviceContext() { return devices[currentDeviceIndex]; }
    static c10::DeviceIndex CurrentDevice() { return currentDeviceIndex; }
    static void SetCurrentDevice(c10::DeviceIndex deviceIndex) { currentDeviceIndex = deviceIndex; }
    uint32_t getDeviceCount() { return devices.size(); }
    bool shouldFallback() { return shouldFallback_; }

private:
    VulkanContext();
    ~VulkanContext();
    
    void initVulkan();
    void createDeviceContexts();
    void createDeviceWithExtensions();
    void createDeviceAllocator();
    void createDeviceCommandPools();
    void validateDevices();

    void queryVulkanVersion();
    uint32_t apiVersion, major, minor, patch;
    bool shouldFallback_ = true;

    // each device creates a new currentDeviceIndex 
    static thread_local c10::DeviceIndex currentDeviceIndex;
};