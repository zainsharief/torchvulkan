#pragma once
#include <c10/util/Exception.h>
#include "vulkan/vulkan_context.h"

// every CPU fallback, for unimplemented ops and for registered ops with an unsupported dtype or shape;
// an error under TORCHVULKAN_STRICT=1
#define TORCHVULKAN_FALLBACK(...) \
    do { \
        if (VulkanContext::Instance().isStrict()) TORCH_CHECK(false, "torchvulkan [NOT IMPLEMENTED]: ", __VA_ARGS__, " Set TORCHVULKAN_STRICT=0 to disable strict mode."); \
        TORCH_WARN_ONCE("torchvulkan [WARNING]: ", __VA_ARGS__, " Set TORCHVULKAN_STRICT=1 to enable strict mode."); \
    } while (0)
