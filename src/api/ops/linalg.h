#include <iostream>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "shaders/shader_registry.h"
#include "helpers.h"

#include <c10/core/MemoryFormat.h>

namespace torchvulkan {

at::Tensor addr_vulkan(
    const at::Tensor& self,
    const at::Tensor& vec1,
    const at::Tensor& vec2,
    const at::Scalar& beta,
    const at::Scalar& alpha
);

at::Tensor bilinear_vulkan(
    const at::Tensor& input1,
    const at::Tensor& input2,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias
);

} // namespace torchvulkan
