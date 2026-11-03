#pragma once
#include <c10/core/Device.h>
#include <volk.h>
#include <atomic>
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
    DeviceContext* GetDeviceContext(c10::DeviceIndex index);
    static c10::DeviceIndex CurrentDevice() { return currentDeviceIndex; }
    static void SetCurrentDevice(c10::DeviceIndex deviceIndex) { currentDeviceIndex = deviceIndex; }
    uint32_t getDeviceCount() { return static_cast<uint32_t>(devices.size()); }

    bool isStrict() { return isStrict_; }
    bool profilingEnabled() { return enableProfiling_; }
    bool strictAllocationEnabled() { return strictAllocation_; }
    static uint64_t ValidationErrorCount();
    static bool IsInBadFork();
    static bool IsInitialized();

private:
    VulkanContext();
    ~VulkanContext();

    void initVulkan();
    void createDeviceContexts();
    void createDeviceWithExtensions();
    void createDeviceAllocator();
    void createDeviceCommandPools();
    void validateDevices();
    void createGenerators();

    // minimum Vulkan version
    static constexpr uint32_t apiVersion = VK_API_VERSION_1_3;
    bool isStrict_ = false;
    bool enableProfiling_ = false;
    bool strictAllocation_ = false;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;

    // each device creates a new currentDeviceIndex
    static thread_local c10::DeviceIndex currentDeviceIndex;
};
