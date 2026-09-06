#pragma once
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "shaders/shader_registry.h"
#include "helpers.h"

#include <c10/core/MemoryFormat.h>
#include <tuple>

enum class ReduceOp {
    SUM = 0,
    AMAX = 1,
    AMIN = 2,
    PROD = 3
};

namespace torchvulkan {

at::Tensor dispatch_reduce_shader(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    ReduceOp operation,
    double empty_value,
    bool has_subgroup
);

at::Tensor reduce_dims_vulkan(
    const at::Tensor& self,
    at::OptionalIntArrayRef dims,
    bool keepdim,
    ReduceOp operation,
    double empty_value,
    bool has_subgroup
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

at::Tensor amin_vulkan(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim
);

at::Tensor prod_dim_vulkan(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype
);

at::Tensor prod_vulkan(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype
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

// composite reductions (no dedicated shader)
at::Tensor logsumexp_vulkan(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim
);

at::Tensor var_correction_vulkan(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<at::Scalar>& correction,
    bool keepdim
);

at::Tensor std_correction_vulkan(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<at::Scalar>& correction,
    bool keepdim
);

std::tuple<at::Tensor, at::Tensor> var_mean_correction_vulkan(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<at::Scalar>& correction,
    bool keepdim
);

std::tuple<at::Tensor, at::Tensor> std_mean_correction_vulkan(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<at::Scalar>& correction,
    bool keepdim
);

at::Tensor& cumsum_out_vulkan(const at::Tensor& self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor& out);
at::Tensor& cumprod_out_vulkan(const at::Tensor& self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor& out);
at::Tensor& argmax_out_vulkan(const at::Tensor& self, c10::optional<int64_t> dim, bool keepdim, at::Tensor& out);
at::Tensor& argmin_out_vulkan(const at::Tensor& self, c10::optional<int64_t> dim, bool keepdim, at::Tensor& out);
void cummax_helper_vulkan(const at::Tensor& self, at::Tensor& values, at::Tensor& indices, int64_t dim);
void cummin_helper_vulkan(const at::Tensor& self, at::Tensor& values, at::Tensor& indices, int64_t dim);

} // namespace torchvulkan
