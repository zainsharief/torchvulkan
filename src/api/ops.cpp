#include <torch/extension.h>
#include <ATen/native/CPUFallback.h>
#include "api/ops/factory.h"
#include "api/ops/binary.h"
#include "api/ops/unary.h"
#include "api/ops/matmul.h"
#include "api/ops/linalg.h"
#include "api/ops/reduce.h"
#include "api/ops/softmax.h"
#include "api/ops/nllloss.h"

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
    m.impl("_reshape_alias", &reshape_alias_vulkan);
    m.impl("view", &view_vulkan);
    m.impl("t", &t_vulkan);
    m.impl("transpose.int", &transpose_int_vulkan);
    m.impl("permute", &permute_vulkan);
    m.impl("contiguous", &contiguous_vulkan);
    m.impl("clone", &clone_vulkan);
    m.impl("_copy_from", &copy_from_vulkan);
    m.impl("_copy_from_and_resize", &copy_from_and_resize_vulkan);
    m.impl("copy_", &copy_vulkan_);
    m.impl("_local_scalar_dense", &local_scalar_dense_vulkan);

    // unary
    m.impl("fill_.Scalar", &fill_scalar_vulkan);
    m.impl("fill_.Tensor", &fill_tensor_vulkan);
    m.impl("zero_", &zero_vulkan);
    m.impl("relu", &relu_vulkan);
    m.impl("exp", &exp_vulkan);
    m.impl("log", &log_vulkan);
    m.impl("sqrt", &sqrt_vulkan);
    m.impl("neg", &neg_vulkan);
    m.impl("reciprocal", &reciprocal_vulkan);

    // binary - Add
    m.impl("add.Tensor", &add_vulkan);
    m.impl("add.Scalar", &add_scalar_vulkan);
    m.impl("add_.Tensor", &add_vulkan_);
    m.impl("add_.Scalar", &add_scalar_vulkan_);

    // binary - Subtract
    m.impl("sub.Tensor", &subtract_vulkan);
    m.impl("sub.Scalar", &subtract_scalar_vulkan);
    m.impl("sub_.Tensor", &subtract_vulkan_);
    m.impl("sub_.Scalar", &subtract_scalar_vulkan_);
    m.impl("rsub.Scalar", &rsub_scalar_vulkan);

    // binary - Multiply
    m.impl("mul.Tensor", &multiply_vulkan);
    m.impl("mul.Scalar", &multiply_scalar_vulkan);
    m.impl("mul_.Tensor", &multiply_vulkan_);
    m.impl("mul_.Scalar", &multiply_scalar_vulkan_);

    // binary - Divide
    m.impl("div.Tensor", &divide_vulkan);
    m.impl("div.Scalar", &divide_scalar_vulkan);
    m.impl("div_.Tensor", &divide_vulkan_);
    m.impl("div_.Scalar", &divide_scalar_vulkan_);

    // binary - Min/Max
    m.impl("maximum", &maximum_vulkan);
    m.impl("minimum", &minimum_vulkan);

    // binary - Pow
    m.impl("pow.Tensor_Tensor", &pow_vulkan);
    m.impl("pow.Tensor_Scalar", &pow_tensor_scalar_vulkan);
    m.impl("pow.Scalar", &pow_scalar_vulkan);

    // binary - Atan2
    m.impl("atan2", &atan2_vulkan);

    // binary - fused
    m.impl("addcmul_", &addcmul_vulkan_);
    m.impl("addcdiv_", &addcdiv_vulkan_);
    m.impl("lerp.Scalar", &lerp_scalar_vulkan);
    m.impl("lerp_.Scalar", &lerp_scalar_vulkan_);
    m.impl("lerp.Scalar_out", &lerp_scalar_vulkan_out);

    // autograd support
    m.impl("threshold_backward", &threshold_backward_vulkan);

    // reductions
    m.impl("sum.dim_IntList", &sum_dim_vulkan);
    m.impl("sum", &sum_vulkan);
    m.impl("amax", &amax_vulkan);
    m.impl("mean.dim", &mean_dim_vulkan);
    m.impl("mean", &mean_vulkan);

    // softmax
    m.impl("_log_softmax", &log_softmax_vulkan);
    m.impl("_log_softmax_backward_data", &log_softmax_backward_vulkan);

    // nll_loss
    m.impl("nll_loss_forward", &nll_loss_forward_vulkan);
    m.impl("nll_loss_backward", &nll_loss_backward_vulkan);

    // matmul
    m.impl("mm", &mm_vulkan);
    m.impl("bmm", &bmm_vulkan);
    m.impl("matmul", &matmul_vulkan);
    m.impl("addmm", &addmm_vulkan);
    m.impl("baddbmm", &baddbmm_vulkan);
    m.impl("addr", &addr_vulkan);
    m.impl("bilinear", &bilinear_vulkan);
}