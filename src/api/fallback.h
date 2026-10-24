#pragma once
#include <c10/util/Exception.h>
#include "vulkan/vulkan_context.h"

// Every path that computes on the CPU instead of the Vulkan device reports itself through this.
// With TORCHVULKAN_STRICT=1 it is an error (the same "[NOT IMPLEMENTED]" prefix as an unimplemented
// operator), otherwise a warning: once per call site, or every time after torch.set_warn_always(True).
#define TORCHVULKAN_CPU_FALLBACK(...)                                                                   \
    do {                                                                                                \
        if (VulkanContext::Instance().isStrict()) {                                                     \
            TORCH_CHECK(false, "torchvulkan [NOT IMPLEMENTED]: ", __VA_ARGS__,                          \
                " Set TORCHVULKAN_STRICT=0 to allow CPU fallbacks.");                                   \
        }                                                                                               \
        TORCH_WARN_ONCE("torchvulkan [WARNING]: ", __VA_ARGS__, " Set TORCHVULKAN_STRICT=1 to make this an error."); \
    } while (0)
