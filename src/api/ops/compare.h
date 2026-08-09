#include <torch/extension.h>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "shaders/shader_registry.h"
#include "helpers.h"

enum class CompareOp {
    EQ = 0,
    NE = 1,
    LT = 2,
    LE = 3,
    GT = 4,
    GE = 5,
    LAND = 6,
    LOR = 7,
    LXOR = 8
};

namespace torchvulkan {

at::Tensor& compare_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out, CompareOp op);
at::Tensor& compare_scalar_out_vulkan(const at::Tensor& self, const at::Scalar& other, at::Tensor& out, CompareOp op);
at::Tensor& eq_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out);
at::Tensor& eq_scalar_out_vulkan(const at::Tensor& self, const at::Scalar& other, at::Tensor& out);
at::Tensor& ne_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out);
at::Tensor& ne_scalar_out_vulkan(const at::Tensor& self, const at::Scalar& other, at::Tensor& out);
at::Tensor& lt_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out);
at::Tensor& lt_scalar_out_vulkan(const at::Tensor& self, const at::Scalar& other, at::Tensor& out);
at::Tensor& le_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out);
at::Tensor& le_scalar_out_vulkan(const at::Tensor& self, const at::Scalar& other, at::Tensor& out);
at::Tensor& gt_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out);
at::Tensor& gt_scalar_out_vulkan(const at::Tensor& self, const at::Scalar& other, at::Tensor& out);
at::Tensor& ge_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out);
at::Tensor& ge_scalar_out_vulkan(const at::Tensor& self, const at::Scalar& other, at::Tensor& out);
at::Tensor& logical_and_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out);
at::Tensor& logical_or_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out);
at::Tensor& logical_xor_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out);
at::Tensor& logical_not_out_vulkan(const at::Tensor& self, at::Tensor& out);
at::Tensor where_vulkan(const at::Tensor& condition, const at::Tensor& self, const at::Tensor& other);
at::Tensor where_scalar_other_vulkan(const at::Tensor& condition, const at::Tensor& self, const at::Scalar& other);
at::Tensor where_scalar_self_vulkan(const at::Tensor& condition, const at::Scalar& self, const at::Tensor& other);
at::Tensor where_scalar_vulkan(const at::Tensor& condition, const at::Scalar& self, const at::Scalar& other);
at::Tensor isnan_vulkan(const at::Tensor& self);
at::Tensor& isinf_out_vulkan(const at::Tensor& self, at::Tensor& out);
at::Tensor& isposinf_out_vulkan(const at::Tensor& self, at::Tensor& out);
at::Tensor& isneginf_out_vulkan(const at::Tensor& self, at::Tensor& out);

} // namespace torchvulkan
