#include <torch/extension.h>
#include <ATen/native/CPUFallback.h>
#include "api/ops/factory.h"
#include "api/ops/binary.h"
#include "api/ops/unary.h"
#include "api/ops/matmul.h"
#include "api/ops/linalg.h"

using namespace torchvulkan;

void vulkan_cpu_fallback_warning(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    TORCH_CHECK(false, "torchvulkan [NOT IMPLEMENTED]: Silent fallback detected for operation: ", op.schema().operator_name());
    at::native::cpu_fallback(op, stack);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&vulkan_cpu_fallback_warning>());
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    // factory
    m.impl("empty.memory_format", &empty_memory_format_vulkan);
    m.impl("empty_strided", &empty_strided_vulkan);
    m.impl("as_strided", &as_strided_vulkan);
    m.impl("resize_", &resize_vulkan);
    m.impl("view", &view_vulkan);
    m.impl("contiguous", &contiguous_vulkan);
    m.impl("clone", &clone_vulkan);
    m.impl("_copy_from", &copy_from_vulkan);
    m.impl("_copy_from_and_resize", &copy_from_and_resize_vulkan);
    m.impl("copy_", &copy_vulkan_);

    // unary
    m.impl("fill_.Scalar", &fill_scalar_vulkan);
    m.impl("fill_.Tensor", &fill_tensor_vulkan);
    m.impl("zero_", &zero_vulkan);

    // binary - Add
    m.impl("add.Tensor", &add_vulkan);
    m.impl("add.Scalar", &add_scalar_vulkan);

    // binary - Subtract
    m.impl("sub.Tensor", &subtract_vulkan);
    m.impl("sub.Scalar", &subtract_scalar_vulkan);
    m.impl("rsub.Scalar", &rsub_scalar_vulkan);

    // binary - Multiply
    m.impl("mul.Tensor", &multiply_vulkan);
    m.impl("mul.Scalar", &multiply_scalar_vulkan);

    // binary - Divide
    m.impl("div.Tensor", &divide_vulkan);
    m.impl("div.Scalar", &divide_scalar_vulkan);

    // binary - Min/Max
    m.impl("maximum", &maximum_vulkan);
    m.impl("minimum", &minimum_vulkan);

    // binary - Pow
    m.impl("pow.Tensor_Tensor", &pow_vulkan);
    m.impl("pow.Tensor_Scalar", &pow_tensor_scalar_vulkan);
    m.impl("pow.Scalar", &pow_scalar_vulkan);

    // binary - Atan2
    m.impl("atan2", &atan2_vulkan);

    // matmul
    m.impl("mm", &mm_vulkan);
    m.impl("bmm", &bmm_vulkan);
    m.impl("matmul", &matmul_vulkan);
    m.impl("addmm", &addmm_vulkan);
    m.impl("baddbmm", &baddbmm_vulkan);
    m.impl("addr", &addr_vulkan);
    m.impl("bilinear", &bilinear_vulkan);
}