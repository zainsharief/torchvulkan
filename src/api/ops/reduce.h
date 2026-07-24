#include <iostream>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "shaders/shader_registry.h"
#include "helpers.h"

#include <c10/core/MemoryFormat.h>

enum class ReduceOp {
    SUM = 0,
    AMAX = 1
};

namespace torchvulkan {

at::Tensor dispatch_reduce_shader(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    ReduceOp operation
);

at::Tensor reduce_dims_vulkan(
    const at::Tensor& self,
    at::OptionalIntArrayRef dims,
    bool keepdim,
    ReduceOp operation
);

at::Tensor sum_dim_vulkan(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype
);

at::Tensor sum_vulkan(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype
);

at::Tensor amax_vulkan(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim
);

at::Tensor mean_dim_vulkan(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype
);

at::Tensor mean_vulkan(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype
);

} // namespace torchvulkan
