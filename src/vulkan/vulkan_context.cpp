#define VOLK_IMPLEMENTATION
#include "vulkan_context.h"

#ifdef __APPLE__
#include <dlfcn.h>
#include <filesystem>
#endif

// VK_KHR_shader_bfloat16 is newer than some Vulkan SDKs still in use (1.4.309 lacks
// it). These are its definitions from vulkan_core.h, so whether bfloat16 runs on
// the GPU depends on the device, not on which headers the build happened to find.
#ifndef VK_KHR_shader_bfloat16
#define VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME "VK_KHR_shader_bfloat16"
static constexpr VkStructureType VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR = static_cast<VkStructureType>(1000141000);
typedef struct VkPhysicalDeviceShaderBfloat16FeaturesKHR {
    VkStructureType    sType;
    void*              pNext;
    VkBool32           shaderBFloat16Type;
    VkBool32           shaderBFloat16DotProduct;
    VkBool32           shaderBFloat16CooperativeMatrix;
} VkPhysicalDeviceShaderBfloat16FeaturesKHR;
#endif

thread_local c10::DeviceIndex VulkanContext::currentDeviceIndex;

static bool isEnvSet(const char* name)
{
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static VkResult loadVulkan()
{
#ifdef __APPLE__
    Dl_info info{};
    if (!isEnvSet("TORCHVULKAN_SYSTEM_VULKAN") && dladdr(reinterpret_cast<const void*>(&loadVulkan), &info) && info.dli_fname != nullptr) 
    {
        auto bundled = std::filesystem::path(info.dli_fname).parent_path() / "libMoltenVK.dylib";
        if (std::filesystem::exists(bundled)) {
            void* module = dlopen(bundled.c_str(), RTLD_NOW | RTLD_LOCAL);
            auto getInstanceProcAddr = module ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(module, "vkGetInstanceProcAddr")) : nullptr;
            if (getInstanceProcAddr != nullptr) {
                volkInitializeCustom(getInstanceProcAddr);
                return VK_SUCCESS;
            }
            const char* error = dlerror();
            TORCH_WARN("torchvulkan: failed to load the bundled MoltenVK (", error ? error : "vkGetInstanceProcAddr not found", "); falling back to the system Vulkan loader.");
        }
    }
#endif
    return volkInitialize();
}

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

void VulkanContext::initVulkan()
{
    VkResult result = loadVulkan();
    if (result != VK_SUCCESS) {
        TORCH_CHECK(false, "torchvulkan [ERROR]: Failed to initialize volk with error code ", std::to_string(result), ". Vulkan loader cannot be found.");
    }

    uint32_t instanceVersion = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion != nullptr) vkEnumerateInstanceVersion(&instanceVersion);
    TORCH_CHECK(instanceVersion >= apiVersion, "torchvulkan [ERROR]: Vulkan 1.3 or newer is required, but the Vulkan loader only supports ",
        VK_API_VERSION_MAJOR(instanceVersion), ".", VK_API_VERSION_MINOR(instanceVersion), "."
    );

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "torchvulkan";
    appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    appInfo.pEngineName = "torchvulkan backend";
    appInfo.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    appInfo.apiVersion = apiVersion;

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
    #endif

    #ifndef NDEBUG
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> supportedLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, supportedLayers.data());
    for (const auto& layer : supportedLayers) {
        if (strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0) instanceLayers.push_back("VK_LAYER_KHRONOS_validation");
    }
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

    isStrict_ = isEnvSet("TORCHVULKAN_STRICT");
    enableProfiling_ = isEnvSet("TORCHVULKAN_PROFILE");
}

void VulkanContext::createDeviceContexts()
{
    uint32_t physicalDeviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr);

    std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
    vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices.data());

    for (const VkPhysicalDevice& physicalDevice : physicalDevices) 
    {                
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        if (props.apiVersion < apiVersion) {
            TORCH_WARN("torchvulkan [WARNING]: Vulkan device '", props.deviceName, "' only supports Vulkan ", VK_API_VERSION_MAJOR(props.apiVersion), ".",
                VK_API_VERSION_MINOR(props.apiVersion), " (1.3 or newer is required) and will be skipped.");
            continue;
        }

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
        if (queueFamilyCount == 0) {
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

        // subgroup reductions need arithmetic + shuffle ops available in compute shaders
        const VkSubgroupFeatureFlags requiredSubgroupOps = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
        context->support_subgroup_arithmetic =
            (subgroupProps.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
            (subgroupProps.supportedOperations & requiredSubgroupOps) == requiredSubgroupOps &&
            subgroupProps.subgroupSize > 0;

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
        VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR supportedPipelineExec{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR supportedCoopMat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
        supportedCoopMat.pNext = &supportedPipelineExec;
        VkPhysicalDeviceVulkan13Features supported13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        supported13.pNext = &supportedCoopMat;
        VkPhysicalDeviceVulkan12Features supported12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        supported12.pNext = &supported13;
        VkPhysicalDeviceVulkan11Features supported11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        supported11.pNext = &supported12;
        VkPhysicalDeviceFeatures2 supportedFeatures2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        supportedFeatures2.pNext = &supported11;

        VkPhysicalDeviceShaderBfloat16FeaturesKHR supportedBfloat16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR};
        if (hasExt(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME)) {
            supportedBfloat16.pNext = supportedFeatures2.pNext;
            supportedFeatures2.pNext = &supportedBfloat16;
        }

        // query which ones are supported
        vkGetPhysicalDeviceFeatures2(device->physicalDevice, &supportedFeatures2);

        // chain of feature structs to enable the features we want (only the ones supported by the device)
        // synchronization2 lets the dispatcher scope its barriers to the stages and accesses it actually uses
        VkPhysicalDeviceVulkan13Features enable13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        enable13.synchronization2 = VK_TRUE;
        enable13.subgroupSizeControl = supported13.subgroupSizeControl;
        enable13.computeFullSubgroups = supported13.computeFullSubgroups;
        VkPhysicalDeviceVulkan12Features enable12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        enable12.pNext = &enable13;
        enable12.shaderFloat16 = supported12.shaderFloat16;
        enable12.shaderInt8 = supported12.shaderInt8;
        enable12.storageBuffer8BitAccess = supported12.storageBuffer8BitAccess;
        enable12.scalarBlockLayout = supported12.scalarBlockLayout;
        enable12.bufferDeviceAddress = supported12.bufferDeviceAddress;
        enable12.shaderSubgroupExtendedTypes = supported12.shaderSubgroupExtendedTypes; // subgroup ops on 8/16/64-bit types for reductions
        VkPhysicalDeviceVulkan11Features enable11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        enable11.storageBuffer16BitAccess = supported11.storageBuffer16BitAccess;
        enable11.pNext = &enable12;
        VkPhysicalDeviceFeatures enable10{};
        enable10.shaderFloat64 = supportedFeatures2.features.shaderFloat64;
        enable10.shaderInt64 = supportedFeatures2.features.shaderInt64;
        enable10.shaderInt16 = supportedFeatures2.features.shaderInt16;
        // no need to chain enable10 -> enable11 here as it is done in features2

        device->support_float32 = true;
        device->support_int32 = true;
        device->support_float64 = supportedFeatures2.features.shaderFloat64;
        device->support_int64 = supportedFeatures2.features.shaderInt64;
        device->support_float16 = supported12.shaderFloat16 && enable11.storageBuffer16BitAccess;
        device->support_bfloat16 = false;
        device->support_int16 = supportedFeatures2.features.shaderInt16 && enable11.storageBuffer16BitAccess;
        device->support_int8 = supported12.shaderInt8 && supported12.storageBuffer8BitAccess;
        device->support_subgroup_extended_types = supported12.shaderSubgroupExtendedTypes;
        device->support_subgroup_control = supported13.subgroupSizeControl;

        VkPhysicalDeviceShaderBfloat16FeaturesKHR enableBfloat16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR};
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR enableCoopMatrices{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
        VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR enablePipelineExec{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};

        if (supportedBfloat16.shaderBFloat16Type && enable11.storageBuffer16BitAccess)
        {
            device->support_bfloat16 = true;
            enableBfloat16.shaderBFloat16Type = VK_TRUE;

            if (supportedBfloat16.shaderBFloat16CooperativeMatrix && hasExt(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) && supportedCoopMat.cooperativeMatrix)
            {
                device->support_bfloat16_coopmat = true;
                enableBfloat16.shaderBFloat16CooperativeMatrix = VK_TRUE;
            }
            enableBfloat16.pNext = enable12.pNext;
            enable12.pNext = &enableBfloat16;
            deviceExtensions.push_back(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME);
        }

        if (hasExt(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) && supportedCoopMat.cooperativeMatrix)
        {
            device->support_coopmat = true;
            enableCoopMatrices.cooperativeMatrix = VK_TRUE;
            enableCoopMatrices.pNext = enable12.pNext;
            enable12.pNext = &enableCoopMatrices;
            deviceExtensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);

            uint32_t propertyCount = 0;
            vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(device->physicalDevice, &propertyCount, nullptr);
            std::vector<VkCooperativeMatrixPropertiesKHR> coopMatProperties;
            coopMatProperties.resize(propertyCount);
            vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(device->physicalDevice, &propertyCount, coopMatProperties.data());

            for (int i = 0; i < propertyCount; ++i)
            {
                VkCooperativeMatrixPropertiesKHR property = coopMatProperties[i];

                bool uses_bfloat16 = property.AType == COMPONENT_TYPE_BFLOAT16 || property.BType == COMPONENT_TYPE_BFLOAT16 ||
                                     property.CType == COMPONENT_TYPE_BFLOAT16 || property.ResultType == COMPONENT_TYPE_BFLOAT16;
                if (uses_bfloat16 && !device->support_bfloat16_coopmat) continue;

                CoopMatConfig config{property.MSize, property.NSize, property.KSize};
                device->cache.addCoopMatConfig(property.AType, property.BType, property.CType, property.ResultType, config);
            }
        }

        if (hasExt(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME) && supportedPipelineExec.pipelineExecutableInfo)
        {
            device->support_pipeline_statistics = true;
            enablePipelineExec.pipelineExecutableInfo = VK_TRUE;
            enablePipelineExec.pNext = enable12.pNext;
            enable12.pNext = &enablePipelineExec;
            deviceExtensions.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
        }
        else if (enableProfiling_)
        {
            TORCH_WARN("torchvulkan [WARNING]: Vulkan device '", device->properties.deviceName, "' does not support ", VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME, "; pipeline statistics will not be reported.");
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
        allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

        VmaVulkanFunctions vmaFunctions = {};
        vmaFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
        vmaFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
        allocatorInfo.pVulkanFunctions = &vmaFunctions;

        if (vmaCreateAllocator(&allocatorInfo, &device->allocator) == VK_SUCCESS) 
        {
            // the shader manager allocates its metadata buffer up front, so it needs the allocator
            device->shader_manager = new VulkanShaderManager(device);
            continue;
        }

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