#define VOLK_IMPLEMENTATION
#include "vulkan_context.h"
#include <iostream>

thread_local c10::DeviceIndex VulkanContext::currentDeviceIndex;

VulkanContext& VulkanContext::Instance() 
{
    static VulkanContext* vulkanContextInstance = new VulkanContext();
    return *vulkanContextInstance;
}

VulkanContext::VulkanContext()
{
    initVulkan();
    createDeviceContexts();
    createDeviceWithExtensions();
    createDeviceAllocator();
    createDeviceCommandPools();
    validateDevices();
}

void VulkanContext::queryVulkanVersion() 
{
    if (vkEnumerateInstanceVersion == nullptr) {
        apiVersion = VK_API_VERSION_1_0;
    } else {
        vkEnumerateInstanceVersion(&apiVersion);
    }

    major = VK_API_VERSION_MAJOR(apiVersion);
    minor = VK_API_VERSION_MINOR(apiVersion);
    patch = VK_API_VERSION_PATCH(apiVersion);
}

void VulkanContext::initVulkan()
{
    VkResult result = volkInitialize(); 
    if (result != VK_SUCCESS) {
        TORCH_CHECK(false, "torchvulkan [ERROR]: Failed to initialize volk with error code ", std::to_string(result), ". Vulkan loader cannot be found.");
    }

    queryVulkanVersion();

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "torchvulkan";
    appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    appInfo.pEngineName = "torchvulkan backend";
    appInfo.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    appInfo.apiVersion = apiVersion; // if changed, edit VmaAllocatorCreateInfo below

    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> supportedExtensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, supportedExtensions.data());

    auto hasExt = [&](const char* extName) {
        for (const auto& ext : supportedExtensions) if (strcmp(ext.extensionName, extName) == 0) return true;
        return false;
    };

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    std::vector<const char*> instanceExtensions;
    std::vector<const char*> instanceLayers;

    #ifdef __APPLE__
    if (hasExt("VK_KHR_portability_enumeration")) {
        instanceExtensions.push_back("VK_KHR_portability_enumeration");
        createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
    instanceExtensions.push_back("VK_KHR_get_physical_device_properties2"); 
    #endif

    #ifndef NDEBUG
    instanceLayers.push_back("VK_LAYER_KHRONOS_validation");
    #endif

    createInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
    createInfo.ppEnabledExtensionNames = instanceExtensions.data();

    createInfo.enabledLayerCount = static_cast<uint32_t>(instanceLayers.size());
    createInfo.ppEnabledLayerNames = instanceLayers.data();

    result = vkCreateInstance(&createInfo, nullptr, &instance);
    if (result != VK_SUCCESS) {
        TORCH_CHECK(false, "torchvulkan [ERROR]: Failed to initialize Vulkan with error code ", std::to_string(result), ".");
    }

    volkLoadInstance(instance);

    const char* env = std::getenv("TORCHVULKAN_STRICT");
    isStrict_ = env != nullptr && env[0] != '\0' && env[0] != '0';
}

void VulkanContext::createDeviceContexts()
{
    uint32_t physicalDeviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr);

    std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
    vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices.data());

    for (const VkPhysicalDevice& physicalDevice : physicalDevices) 
    {                
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
        if (queueFamilyCount == 0) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(physicalDevice, &props);
            TORCH_WARN("torchvulkan [WARNING]: Vulkan device '", props.deviceName, "' does not have any queue families and will be skipped.");
            continue;
        }
    
        std::vector<VkQueueFamilyProperties> families(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, families.data());
        
        // if there is a dedicated compute core, we pick that, otherwise we pick any compute & graphics core
        int32_t bestQueueFamily = -1;
        for (uint32_t i = 0; i < queueFamilyCount; i++) 
        {
            if (!(families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) continue;
            bestQueueFamily = i;
            if (!(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) break;   
        }
        // make sure there exists a compute core
        if (bestQueueFamily == -1) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(physicalDevice, &props);
            TORCH_WARN("torchvulkan [WARNING]: Vulkan device '", props.deviceName, "' does not have a compute queue family and will be skipped.");
            continue;
        }

        DeviceContext* context = new DeviceContext();
        context->physicalDevice = physicalDevice;
        context->computeQueueFamily = bestQueueFamily;

        VkPhysicalDeviceSubgroupProperties subgroupProps{};
        subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        subgroupProps.pNext = nullptr;

        VkPhysicalDeviceProperties2 props2{};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &subgroupProps;

        vkGetPhysicalDeviceProperties2(physicalDevice, &props2);

        context->properties = props2.properties;        
        context->subgroup_size = subgroupProps.subgroupSize;

        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryHeapCount; ++i) 
        {
            if (!(memProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) continue;
            context->vram_heap_index = i;
            break;
        }

        devices.push_back(context);
    }
}

void VulkanContext::createDeviceWithExtensions()
{
    float priority = 1.0f;
    
    for (DeviceContext* device : devices) 
    { 

        std::vector<const char*> deviceExtensions;

        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(device->physicalDevice, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> availableExts(extCount);
        vkEnumerateDeviceExtensionProperties(device->physicalDevice, nullptr, &extCount, availableExts.data());

        auto hasExt = [&](const char* name) {
            for (const auto& e : availableExts) if (strcmp(e.extensionName, name) == 0) return true;
            return false;
        };
        
        // chain of feature structs to query what the device supports
        VkPhysicalDeviceSubgroupSizeControlFeaturesEXT supportedSubgroupControl{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT};
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR supportedCoopMat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT supportedAtomicFloat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
        supportedAtomicFloat.pNext = &supportedCoopMat;
        VkPhysicalDeviceShaderIntegerDotProductFeatures supportedDotProduct{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
        supportedDotProduct.pNext = &supportedAtomicFloat;
        VkPhysicalDeviceVulkan12Features supported12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        supported12.pNext = &supportedDotProduct;
        VkPhysicalDeviceVulkan11Features supported11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        supported11.pNext = &supported12;
        VkPhysicalDeviceFeatures2 supportedFeatures2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        supportedFeatures2.pNext = &supported11;

        // query whch ones are supported
        vkGetPhysicalDeviceFeatures2(device->physicalDevice, &supportedFeatures2);

        if (!hasExt(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME) || !hasExt(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME)) {
            TORCH_WARN("torchvulkan [WARNING]: Vulkan device '", device->properties.deviceName, "' does not support required features and will be skipped.");
            device->valid = false;
            continue;
        }

        // chain of feature structs to enable the features we want (only the ones supported by the device)
        VkPhysicalDeviceSubgroupSizeControlFeaturesEXT enableSubgroupControl{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT};
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR enableCoopMatrices{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT enableAtomicFloat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
        enableAtomicFloat.shaderBufferFloat32Atomics = supportedAtomicFloat.shaderBufferFloat32Atomics;
        enableAtomicFloat.shaderBufferFloat32AtomicAdd = supportedAtomicFloat.shaderBufferFloat32AtomicAdd;
        VkPhysicalDeviceShaderIntegerDotProductFeatures enableDotProduct{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
        enableDotProduct.shaderIntegerDotProduct = supportedDotProduct.shaderIntegerDotProduct;
        enableDotProduct.pNext = &enableAtomicFloat;
        VkPhysicalDeviceVulkan12Features enable12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        enable12.shaderFloat16 = supported12.shaderFloat16; 
        enable12.shaderInt8 = supported12.shaderInt8;
        enable12.storageBuffer8BitAccess = supported12.storageBuffer8BitAccess;
        enable12.scalarBlockLayout = supported12.scalarBlockLayout;
        enable12.bufferDeviceAddress = supported12.bufferDeviceAddress;
        enable12.pNext = &enableDotProduct;
        VkPhysicalDeviceVulkan11Features enable11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        enable11.storageBuffer16BitAccess = supported11.storageBuffer16BitAccess;
        enable11.pNext = &enable12;
        VkPhysicalDeviceFeatures enable10{};
        enable10.shaderFloat64 = supportedFeatures2.features.shaderFloat64;
        enable10.shaderInt64 = supportedFeatures2.features.shaderInt64;
        enable10.shaderInt16 = supportedFeatures2.features.shaderInt16;

        device->support_float32 = true;
        device->support_int32 = true;
        device->support_float64 = supportedFeatures2.features.shaderFloat64;
        device->support_int64 = supportedFeatures2.features.shaderInt64;
        device->support_float16 = supported12.shaderFloat16 && enable11.storageBuffer16BitAccess;
        device->support_bfloat16 = false; // adding support when it comes out!
        device->support_int16 = supportedFeatures2.features.shaderInt16 && enable11.storageBuffer16BitAccess;
        device->support_int8 = supported12.shaderInt8 && supported12.storageBuffer8BitAccess;

        if (hasExt(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) && supportedCoopMat.cooperativeMatrix) 
        {
            device->support_coopmat = true;
            enableCoopMatrices.cooperativeMatrix = VK_TRUE;
            enableAtomicFloat.pNext = &enableCoopMatrices;
            deviceExtensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);

            uint32_t propertyCount = 0;
            vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(device->physicalDevice, &propertyCount, nullptr);
            std::vector<VkCooperativeMatrixPropertiesKHR> coopMatProperties;
            coopMatProperties.resize(propertyCount);
            vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(device->physicalDevice, &propertyCount, coopMatProperties.data());

            for (int i = 0; i < propertyCount; ++i)
            {
                VkCooperativeMatrixPropertiesKHR property = coopMatProperties[i];
                CoopMatConfig config{property.MSize, property.NSize, property.KSize};
                device->cache.addCoopMatConfig(property.AType, property.BType, property.CType, property.ResultType, config);
            }

            if (hasExt(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME) && supportedSubgroupControl.subgroupSizeControl) 
            {
                device->support_subgroup_control = true;
                enableSubgroupControl.subgroupSizeControl = VK_TRUE;
                enableSubgroupControl.computeFullSubgroups = supportedSubgroupControl.computeFullSubgroups;
                enableCoopMatrices.pNext = &enableSubgroupControl;
                deviceExtensions.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
            }
        }

        // creating the device 
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = device->computeQueueFamily;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &priority;

        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.features = enable10;
        features2.pNext = &enable11;

        deviceExtensions.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
        deviceExtensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
        deviceExtensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
        deviceExtensions.push_back(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);

        #ifdef __APPLE__
        if (hasExt("VK_KHR_portability_subset")) {
            deviceExtensions.push_back("VK_KHR_portability_subset");
        }
        #endif

        for (const char* ext: deviceExtensions) 
        {
            if (hasExt(ext)) continue;
            TORCH_WARN("torchvulkan [WARNING]: Vulkan device '", device->properties.deviceName, "' does not support required extension ", ext, " and will be skipped.");
            device->valid = false;
        }
        if (!device->valid) continue;

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = 1;
        createInfo.pQueueCreateInfos = &queueInfo;
        createInfo.pNext = &features2;
        createInfo.pEnabledFeatures = VK_NULL_HANDLE;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (vkCreateDevice(device->physicalDevice, &createInfo, nullptr, &device->device) == VK_SUCCESS) {
            volkLoadDeviceTable(&device->device_table, device->device);
            device->cache.setDevice(device->device);
            device->cache.setDeviceTable(device->device_table);
            device->device_table.vkGetDeviceQueue(device->device, device->computeQueueFamily, 0, &device->computeQueue);
            continue;
        }
            
        TORCH_WARN("torchvulkan [WARNING]: Failed to create a logical device for ", device->properties.deviceName, ". This device will be skipped.");
        device->valid = false;
    }
}
    
void VulkanContext::createDeviceAllocator()
{
    for (DeviceContext* device : devices) 
    {
        if (!device->valid) continue;

        VmaAllocatorCreateInfo allocatorInfo = {};
        allocatorInfo.physicalDevice = device->physicalDevice;
        allocatorInfo.device = device->device;
        allocatorInfo.instance = instance;
        allocatorInfo.vulkanApiVersion = apiVersion;
        allocatorInfo.flags = VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;

        VmaVulkanFunctions vmaFunctions = {};
        vmaFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
        vmaFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
        allocatorInfo.pVulkanFunctions = &vmaFunctions;

        if (vmaCreateAllocator(&allocatorInfo, &device->allocator) == VK_SUCCESS) continue;
        TORCH_WARN("torchvulkan [WARNING]: Failed to create VMA allocator for ", device->properties.deviceName, ". This device will be skipped.");
        device->valid = false;
    }
}

void VulkanContext::createDeviceCommandPools()
{
    for (DeviceContext* device : devices) 
    {
        if (!device->valid) continue;

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = device->computeQueueFamily;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        if (device->device_table.vkCreateCommandPool(device->device, &poolInfo, nullptr, &device->commandPool) == VK_SUCCESS) continue;
        TORCH_WARN("torchvulkan [WARNING]: Failed to create a command pool for ", device->properties.deviceName, ". This device will be skipped.");
        device->valid = false;
    }
}

void VulkanContext::validateDevices()
{
    std::vector<DeviceContext*> validDevices;
    validDevices.reserve(devices.size());

    for (DeviceContext* device : devices) {
        if (device->valid) validDevices.push_back(device);
        else delete device;
    }

    devices = std::move(validDevices);
    if (devices.empty()) TORCH_CHECK(false, "torchvulkan [WARNING]: No Vulkan devices could be initialized.");
}

VulkanContext::~VulkanContext() 
{
    for (const DeviceContext* device : devices) delete device;
    if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
}