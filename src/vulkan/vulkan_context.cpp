#define VOLK_IMPLEMENTATION
#include "vulkan_context.h"
#include "generator.h"
#include "shader.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>

#ifdef __APPLE__
#include <dlfcn.h>
#include <filesystem>
#endif

#ifndef _WIN32
#include <pthread.h>
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

static std::atomic<bool> contextInitialized{false};
static std::atomic<bool> inBadFork{false};
static std::atomic<uint64_t> validationErrors{0};

static VKAPI_ATTR VkBool32 VKAPI_CALL validationCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /* types */,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void* /* user_data */)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        validationErrors++;
        std::fprintf(stderr, "torchvulkan [VALIDATION]: %s\n", data->pMessage);
    }
    return VK_FALSE;
}

// device types in the order devices are numbered: a discrete GPU becomes device 0, a software rasterizer last
static int deviceTypeRank(VkPhysicalDeviceType type)
{
    switch (type)
    {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 0;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 1;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 2;
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return 4;
        default: return 3;
    }
}

#ifndef _WIN32
// a forked child shares the parent's driver state, which is not valid in the child
static void markForkedChild() { inBadFork = contextInitialized.load(); }
#endif

VulkanContext& VulkanContext::Instance() 
{
    TORCH_CHECK(!inBadFork, 
        "torchvulkan [ERROR]: Cannot use Vulkan in a forked subprocess after the parent process initialised it. "
        "Use the 'spawn' start method (e.g. multiprocessing_context='spawn' for a DataLoader)."
    );

    static VulkanContext* vulkanContextInstance = []() {
#ifndef _WIN32
        pthread_atfork(nullptr, nullptr, markForkedChild);
#endif
        VulkanContext* context = new VulkanContext();
        contextInitialized = true;
        return context;
    }();
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
    createGenerators();
}

void VulkanContext::initVulkan()
{
    VkResult result = loadVulkan();
    TORCH_CHECK(result == VK_SUCCESS, "torchvulkan [ERROR]: Failed to initialize volk with error code ", result, ". Vulkan loader cannot be found.");

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

    // the validation layer is on with TORCHVULKAN_VALIDATION=1 (=sync adds synchronization validation)
    const char* validationSetting = std::getenv("TORCHVULKAN_VALIDATION");
    bool wantValidation = isEnvSet("TORCHVULKAN_VALIDATION");
    bool wantSyncValidation = validationSetting != nullptr && std::strcmp(validationSetting, "sync") == 0;
    bool hasValidation = false;

    if (wantValidation) {
        uint32_t layerCount = 0;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> supportedLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, supportedLayers.data());
        for (const auto& layer : supportedLayers) {
            if (strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0) hasValidation = true;
        }
        if (hasValidation) instanceLayers.push_back("VK_LAYER_KHRONOS_validation");
        else TORCH_WARN("torchvulkan [WARNING]: Validation was requested, but VK_LAYER_KHRONOS_validation is not installed.");
    }

    bool hasDebugUtils = hasValidation && hasExt(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (hasDebugUtils) instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    VkValidationFeatureEnableEXT syncValidation = VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
    VkValidationFeaturesEXT validationFeatures{VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
    validationFeatures.enabledValidationFeatureCount = 1;
    validationFeatures.pEnabledValidationFeatures = &syncValidation;
    if (hasValidation && wantSyncValidation) {
        // VK_EXT_validation_features is provided by the layer itself
        instanceExtensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
        createInfo.pNext = &validationFeatures;
    }

    createInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
    createInfo.ppEnabledExtensionNames = instanceExtensions.data();

    createInfo.enabledLayerCount = static_cast<uint32_t>(instanceLayers.size());
    createInfo.ppEnabledLayerNames = instanceLayers.data();

    result = vkCreateInstance(&createInfo, nullptr, &instance);
    TORCH_CHECK(result == VK_SUCCESS, "torchvulkan [ERROR]: Failed to initialize Vulkan with error code ", result, ".");

    volkLoadInstance(instance);

    if (hasDebugUtils) {
        VkDebugUtilsMessengerCreateInfoEXT messengerInfo{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        messengerInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        messengerInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                    VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        messengerInfo.pfnUserCallback = validationCallback;
        if (vkCreateDebugUtilsMessengerEXT(instance, &messengerInfo, nullptr, &debugMessenger) != VK_SUCCESS) debugMessenger = VK_NULL_HANDLE;
    }

    isStrict_ = isEnvSet("TORCHVULKAN_STRICT");
    enableProfiling_ = isEnvSet("TORCHVULKAN_PROFILE");
    strictAllocation_ = isEnvSet("TORCHVULKAN_STRICT_ALLOCATION");
}

void VulkanContext::createDeviceContexts()
{
    uint32_t physicalDeviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr);

    std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
    vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices.data());

    std::stable_sort(physicalDevices.begin(), physicalDevices.end(), [](VkPhysicalDevice a, VkPhysicalDevice b) {
        VkPhysicalDeviceProperties pa, pb;
        vkGetPhysicalDeviceProperties(a, &pa);
        vkGetPhysicalDeviceProperties(b, &pb);
        return deviceTypeRank(pa.deviceType) < deviceTypeRank(pb.deviceType);
    });

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

        VkPhysicalDeviceVulkan13Properties props13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
        VkPhysicalDeviceVulkan12Properties props12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
        props12.pNext = &props13;
        VkPhysicalDeviceSubgroupProperties subgroupProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        subgroupProps.pNext = &props12;

        VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props2.pNext = &subgroupProps;

        vkGetPhysicalDeviceProperties2(physicalDevice, &props2);

        VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features2.pNext = &features13;
        vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);

        context->properties = props2.properties;
        context->driver_id = props12.driverID;

        // every pipeline is pinned to the largest subgroup size (see createPipeline); without control, kernels get the driver's default.
        // a pinned size caps a workgroup at size * maxComputeWorkgroupSubgroups, so devices where that is below the workgroups
        // we launch (lavapipe: 4 * 32) are left unpinned
        bool pinned_fits = props13.maxSubgroupSize * props13.maxComputeWorkgroupSubgroups >= props2.properties.limits.maxComputeWorkGroupInvocations;
        context->support_subgroup_control = features13.subgroupSizeControl && (props13.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) && pinned_fits;
        context->subgroup_size = context->support_subgroup_control ? props13.maxSubgroupSize : subgroupProps.subgroupSize;

        // subgroup reductions need arithmetic + shuffle ops available in compute shaders
        const VkSubgroupFeatureFlags requiredSubgroupOps = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
        context->support_subgroup_arithmetic =
            (subgroupProps.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
            (subgroupProps.supportedOperations & requiredSubgroupOps) == requiredSubgroupOps &&
            subgroupProps.subgroupSize > 0 &&
            context->support_subgroup_control;

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
        
        // chain of feature structs to query what the device supports; extension structs only when the extension exists
        VkPhysicalDeviceVulkan13Features supported13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceVulkan12Features supported12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        supported12.pNext = &supported13;
        VkPhysicalDeviceVulkan11Features supported11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        supported11.pNext = &supported12;
        VkPhysicalDeviceFeatures2 supportedFeatures2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        supportedFeatures2.pNext = &supported11;

        auto chainQuery = [&](auto& features) {
            features.pNext = supportedFeatures2.pNext;
            supportedFeatures2.pNext = &features;
        };

        VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR supportedPipelineExec{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
        if (hasExt(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME)) chainQuery(supportedPipelineExec);
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR supportedCoopMat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
        if (hasExt(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) chainQuery(supportedCoopMat);
        VkPhysicalDeviceShaderBfloat16FeaturesKHR supportedBfloat16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR};
        if (hasExt(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME)) chainQuery(supportedBfloat16);

        // query which ones are supported
        vkGetPhysicalDeviceFeatures2(device->physicalDevice, &supportedFeatures2);

        // chain of feature structs to enable the features we want (only the ones supported by the device)
        // synchronization2 lets the dispatcher scope its barriers to the stages and accesses it actually uses
        VkPhysicalDeviceVulkan13Features enable13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        enable13.synchronization2 = VK_TRUE;
        enable13.subgroupSizeControl = supported13.subgroupSizeControl;
        enable13.computeFullSubgroups = supported13.computeFullSubgroups;
        enable13.maintenance4 = supported13.maintenance4; // Slang sizes workgroups with LocalSizeId
        VkPhysicalDeviceVulkan12Features enable12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        enable12.pNext = &enable13;
        enable12.shaderFloat16 = supported12.shaderFloat16;
        enable12.shaderInt8 = supported12.shaderInt8;
        enable12.storageBuffer8BitAccess = supported12.storageBuffer8BitAccess;
        enable12.storagePushConstant8 = supported12.storagePushConstant8; // 8-bit kernels take T-typed scalars as push constants
        enable12.scalarBlockLayout = supported12.scalarBlockLayout;
        enable12.bufferDeviceAddress = supported12.bufferDeviceAddress;
        enable12.shaderSubgroupExtendedTypes = supported12.shaderSubgroupExtendedTypes; // subgroup ops on 8/16/64-bit types for reductions
        enable12.vulkanMemoryModel = supported12.vulkanMemoryModel; // Slang's SPIR-V declares the Vulkan memory model
        enable12.vulkanMemoryModelDeviceScope = supported12.vulkanMemoryModelDeviceScope;
        VkPhysicalDeviceVulkan11Features enable11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        enable11.storageBuffer16BitAccess = supported11.storageBuffer16BitAccess;
        enable11.storagePushConstant16 = supported11.storagePushConstant16;
        enable11.pNext = &enable12;
        VkPhysicalDeviceFeatures enable10{};
        enable10.shaderFloat64 = supportedFeatures2.features.shaderFloat64;
        enable10.shaderInt64 = supportedFeatures2.features.shaderInt64;
        enable10.shaderInt16 = supportedFeatures2.features.shaderInt16;
        // no need to chain enable10 -> enable11 here as it is done in features2

        // every kernel takes its tensors as 64-bit buffer device addresses
        if (!supported12.bufferDeviceAddress || !supportedFeatures2.features.shaderInt64) {
            TORCH_WARN("torchvulkan [WARNING]: Vulkan device '", device->properties.deviceName, "' lacks bufferDeviceAddress or shaderInt64 and will be skipped.");
            device->valid = false;
            continue;
        }

        device->support_float32 = true;
        device->support_int32 = true;
        device->support_float64 = supportedFeatures2.features.shaderFloat64;
        device->support_int64 = true;
        device->support_float16 = supported12.shaderFloat16 && enable11.storageBuffer16BitAccess;
        device->support_bfloat16 = false;
        device->support_int16 = supportedFeatures2.features.shaderInt16 && enable11.storageBuffer16BitAccess;
        device->support_int8 = supported12.shaderInt8 && supported12.storageBuffer8BitAccess;
        device->support_subgroup_extended_types = supported12.shaderSubgroupExtendedTypes;

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
            std::vector<VkCooperativeMatrixPropertiesKHR> coopMatProperties(propertyCount, {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
            vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(device->physicalDevice, &propertyCount, coopMatProperties.data());

            for (const VkCooperativeMatrixPropertiesKHR& property : coopMatProperties)
            {
                // matmul_coop's fragments are subgroup-scoped
                if (property.scope != VK_SCOPE_SUBGROUP_KHR) continue;

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

        // without memory budget, VMA estimates the budget as 80% of the heap size
        if (hasExt(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME))
        {
            device->support_memory_budget = true;
            deviceExtensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
        }

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
        allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
        if (device->support_memory_budget) allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;

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
    TORCH_CHECK(!devices.empty(), "torchvulkan [ERROR]: No Vulkan devices could be initialized.");
    for (size_t i = 0; i < devices.size(); i++) devices[i]->index = static_cast<c10::DeviceIndex>(i);
}

DeviceContext* VulkanContext::GetDeviceContext(c10::DeviceIndex index)
{
    if (index < 0) index = currentDeviceIndex;
    TORCH_CHECK(index >= 0 && index < static_cast<c10::DeviceIndex>(devices.size()), "torchvulkan [ERROR]: Invalid Vulkan device index ", index, ".");
    return devices[index];
}

uint64_t VulkanContext::ValidationErrorCount()
{
    return validationErrors.load();
}

bool VulkanContext::IsInBadFork()
{
    return inBadFork;
}

bool VulkanContext::IsInitialized()
{
    return contextInitialized;
}

void VulkanContext::createGenerators()
{
    std::vector<at::Generator> generators = VulkanGeneratorImpl::create_default_generators(devices.size());
    for (size_t i = 0; i < devices.size(); i++) devices[i]->generator = std::move(generators[i]);
}

VulkanContext::~VulkanContext() 
{
    for (const DeviceContext* device : devices) delete device;
    if (debugMessenger != VK_NULL_HANDLE) vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
    if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
}