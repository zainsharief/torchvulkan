#include <iostream>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "shaders/shader_registry.h"
#include "helpers.h"

#include <c10/core/MemoryFormat.h>
#include <ATen/TensorIterator.h>
#include <functional>

enum class BinaryOp {
    ADD = 0,
    SUB = 1,
    RSUB = 2,
    MUL = 3,
    DIV = 4,
    MAX = 5,
    MIN = 6,
    POW = 7,
    RPOW = 8,
    ATAN2 = 9,
    THRESHOLD_BACKWARD = 10
};

namespace torchvulkan {

at::Tensor add_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha);
at::Tensor add_scalar_vulkan(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha);
at::Tensor subtract_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha);
at::Tensor subtract_scalar_vulkan(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha);
at::Tensor rsub_scalar_vulkan(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha);
at::Tensor multiply_vulkan(const at::Tensor& self, const at::Tensor& other);
at::Tensor multiply_scalar_vulkan(const at::Tensor& self, const at::Scalar& other);
at::Tensor divide_vulkan(const at::Tensor& self, const at::Tensor& other);
at::Tensor divide_scalar_vulkan(const at::Tensor& self, const at::Scalar& other);
at::Tensor maximum_vulkan(const at::Tensor& self, const at::Tensor& other);
at::Tensor minimum_vulkan(const at::Tensor& self, const at::Tensor& other);
at::Tensor pow_vulkan(const at::Tensor& self, const at::Tensor& other);
at::Tensor pow_tensor_scalar_vulkan(const at::Tensor& self, const at::Scalar& other);
at::Tensor pow_scalar_vulkan(const at::Scalar& self, const at::Tensor& other);
at::Tensor atan2_vulkan(const at::Tensor& self, const at::Tensor& other);
at::Tensor threshold_backward_vulkan(const at::Tensor& grad_output, const at::Tensor& self, const at::Scalar& threshold);

at::Tensor& add_vulkan_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha);
at::Tensor& add_scalar_vulkan_(at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha);
at::Tensor& subtract_vulkan_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha);
at::Tensor& subtract_scalar_vulkan_(at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha);
at::Tensor& multiply_vulkan_(at::Tensor& self, const at::Tensor& other);
at::Tensor& multiply_scalar_vulkan_(at::Tensor& self, const at::Scalar& other);
at::Tensor& divide_vulkan_(at::Tensor& self, const at::Tensor& other);
at::Tensor& divide_scalar_vulkan_(at::Tensor& self, const at::Scalar& other);
at::Tensor& addcmul_vulkan_(at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2, const at::Scalar& value);
at::Tensor& addcdiv_vulkan_(at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2, const at::Scalar& value);

at::Tensor lerp_scalar_vulkan(const at::Tensor& self, const at::Tensor& end, const at::Scalar& weight);
at::Tensor& lerp_scalar_vulkan_(at::Tensor& self, const at::Tensor& end, const at::Scalar& weight);
at::Tensor& lerp_scalar_vulkan_out(const at::Tensor& self, const at::Tensor& end, const at::Scalar& weight, at::Tensor& out);

at::Tensor binary_op_vulkan(
    const at::Tensor& self, 
    const at::Tensor& other, 
    const at::Scalar& alpha, 
    BinaryOp operation, 
    const std::function<at::Tensor(const at::Tensor&, const at::Tensor&)>& fallback
);

at::Tensor binary_op_vulkan(
    const at::Tensor& self, 
    const at::Scalar& other, 
    const at::Scalar& alpha, 
    BinaryOp operation, 
    const std::function<at::Tensor(const at::Tensor&, const at::Scalar&)>& fallback
);

} // namespace torchvulkan