#include <torch/extension.h>
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "torchvulkan backend";

    m.def("is_available", []() -> bool {
        try {
            return VulkanContext::Instance().getDeviceCount() > 0;
        } catch (...) {
            return false;
        }
    });

    m.def("device_count", []() -> int64_t {
        try {
            return static_cast<int64_t>(VulkanContext::Instance().getDeviceCount());
        } catch (...) {
            return 0;
        }
    });

    m.def("synchronize", []() {
        VulkanContext::Instance().CurrentDeviceContext()->flush();
    });

    m.def("empty_cache", []() {
        DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
        device->flush();
        globalVulkanAllocator.clearResources();
        device->cache.clearCache();
    });
}