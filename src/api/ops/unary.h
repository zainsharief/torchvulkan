#include <iostream>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "shaders/shader_registry.h"
#include "helpers.h"

#include <c10/core/MemoryFormat.h>

enum class UnaryOp {
    RELU = 0
};

namespace torchvulkan {

at::Tensor& fill_scalar_vulkan(
    at::Tensor& self,
    const at::Scalar& value
);

at::Tensor& fill_tensor_vulkan(
    at::Tensor& self,
    const at::Tensor& value
);

at::Tensor& zero_vulkan(
    at::Tensor& self
);

at::Tensor relu_vulkan(const at::Tensor& self);

void dispatch_unary_shader(
    const at::Tensor& src,
    const at::Tensor& dst,
    UnaryOp operation
);

} // namespace torchvulkan

