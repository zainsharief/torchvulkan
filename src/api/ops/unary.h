#include <iostream>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "shaders/shader_registry.h"
#include "helpers.h"

#include <c10/core/MemoryFormat.h>
#include <functional>

enum class UnaryOp {
    RELU = 0,
    EXP = 1,
    LOG = 2,
    SQRT = 3,
    NEG = 4,
    RECIPROCAL = 5
};

namespace torchvulkan {

at::Tensor& fill_scalar_vulkan(at::Tensor& self, const at::Scalar& value);
at::Tensor& fill_tensor_vulkan(at::Tensor& self, const at::Tensor& value);
at::Tensor& zero_vulkan(at::Tensor& self);
at::Tensor relu_vulkan(const at::Tensor& self);
at::Tensor exp_vulkan(const at::Tensor& self);
at::Tensor log_vulkan(const at::Tensor& self);
at::Tensor sqrt_vulkan(const at::Tensor& self);
at::Tensor neg_vulkan(const at::Tensor& self);
at::Tensor reciprocal_vulkan(const at::Tensor& self);

at::Tensor unary_op_vulkan(
    const at::Tensor& self,
    UnaryOp operation,
    const std::function<at::Tensor(const at::Tensor&)>& fallback
);

} // namespace torchvulkan

