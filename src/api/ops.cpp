#include <torch/extension.h>
#include <ATen/native/CPUFallback.h>
#include "api/ops/factory.h"
#include "api/ops/binary.h"
#include "api/ops/unary.h"
#include "api/ops/matmul.h"
#include "api/ops/reduce.h"
#include "api/ops/softmax.h"
#include "api/ops/nllloss.h"
#include "api/ops/composite.h"
#include "api/ops/compare.h"

using namespace torchvulkan;

void vulkan_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) 
{
    VulkanContext& context = VulkanContext::Instance();
    if (context.isStrict()) TORCH_CHECK(false, "torchvulkan [NOT IMPLEMENTED]: Silent fallback detected for operation: ", op.schema().operator_name(), ". Set TORCHVULKAN_STRICT=0 to disable strict mode.");
    TORCH_WARN_ONCE("torchvulkan [NOT IMPLEMENTED]: Silent fallback detected for operation: ", op.schema().operator_name(), ". Set TORCHVULKAN_STRICT=1 to enable strict mode.");
    at::native::cpu_fallback(op, stack);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&vulkan_cpu_fallback>());
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

    // unary 
    m.impl("sin", &sin_vulkan);
    m.impl("cos", &cos_vulkan);
    m.impl("tan", &tan_vulkan);
    m.impl("asin", &asin_vulkan);
    m.impl("acos", &acos_vulkan);
    m.impl("atan", &atan_vulkan);
    m.impl("sinh", &sinh_vulkan);
    m.impl("cosh", &cosh_vulkan);
    m.impl("tanh", &tanh_vulkan);
    m.impl("asinh", &asinh_vulkan);
    m.impl("acosh", &acosh_vulkan);
    m.impl("atanh", &atanh_vulkan);
    m.impl("exp2", &exp2_vulkan);
    m.impl("log2", &log2_vulkan);
    m.impl("log10", &log10_vulkan);
    m.impl("expm1", &expm1_vulkan);
    m.impl("log1p", &log1p_vulkan);
    m.impl("rsqrt", &rsqrt_vulkan);
    m.impl("sigmoid", &sigmoid_vulkan);
    m.impl("deg2rad", &deg2rad_vulkan);
    m.impl("rad2deg", &rad2deg_vulkan);

    m.impl("floor", &floor_vulkan);
    m.impl("ceil", &ceil_vulkan);
    m.impl("trunc", &trunc_vulkan);
    m.impl("round", &round_vulkan);
    m.impl("frac", &frac_vulkan);
    m.impl("abs", &abs_vulkan);
    m.impl("sign", &sign_vulkan);
    m.impl("sgn", &sign_vulkan);
    m.impl("logit", &logit_vulkan);

    m.impl("erf", &erf_vulkan);
    m.impl("erfinv", &erfinv_vulkan);
    m.impl("sinc", &sinc_vulkan);
    m.impl("special_entr", &entr_vulkan);
    m.impl("lgamma", &lgamma_vulkan);
    m.impl("digamma", &digamma_vulkan);
    m.impl("i0", &i0_vulkan);
    m.impl("special_i1", &i1_vulkan);
    m.impl("lgamma.out", &lgamma_out_vulkan);
    m.impl("gelu.out", &gelu_out_vulkan);
    m.impl("erfc.out", &erfc_out_vulkan);
    m.impl("special_i0e.out", &i0e_out_vulkan);
    m.impl("special_i1e.out", &i1e_out_vulkan);
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

    // binary - elementwise math
    m.impl("fmax", &fmax_vulkan);
    m.impl("fmin", &fmin_vulkan);
    m.impl("fmod.Tensor", &fmod_vulkan);
    m.impl("fmod.Scalar", &fmod_scalar_vulkan);
    m.impl("remainder.Tensor", &remainder_vulkan);
    m.impl("remainder.Scalar", &remainder_scalar_vulkan);
    m.impl("hypot", &hypot_vulkan);
    m.impl("xlogy.Tensor", &xlogy_vulkan);
    m.impl("logaddexp", &logaddexp_vulkan);
    m.impl("logaddexp2", &logaddexp2_vulkan);

    // binary - clamp (composed from MAX/MIN)
    m.impl("clamp", &clamp_scalar_vulkan);
    m.impl("clamp.Tensor", &clamp_tensor_vulkan);
    m.impl("clamp_min", &clamp_min_scalar_vulkan);
    m.impl("clamp_min.Tensor", &clamp_min_tensor_vulkan);
    m.impl("clamp_max", &clamp_max_scalar_vulkan);
    m.impl("clamp_max.Tensor", &clamp_max_tensor_vulkan);

    // composite activations (no dedicated shader)
    m.impl("silu", &silu_vulkan);
    m.impl("hardtanh", &hardtanh_vulkan);
    m.impl("hardsigmoid", &hardsigmoid_vulkan);
    m.impl("hardswish", &hardswish_vulkan);
    m.impl("leaky_relu", &leaky_relu_vulkan);
    m.impl("elu", &elu_vulkan);
    m.impl("softplus", &softplus_vulkan);
    m.impl("mish", &mish_vulkan);
    m.impl("tanhshrink", &tanhshrink_vulkan);
    m.impl("square", &square_vulkan);
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
    m.impl("amin", &amin_vulkan);
    m.impl("prod", &prod_vulkan);
    m.impl("prod.dim_int", &prod_dim_vulkan);
    m.impl("mean.dim", &mean_dim_vulkan);
    m.impl("mean", &mean_vulkan);
    m.impl("logsumexp", &logsumexp_vulkan);
    m.impl("cumsum.out", &cumsum_out_vulkan);
    m.impl("cumprod.out", &cumprod_out_vulkan);
    m.impl("argmax.out", &argmax_out_vulkan);
    m.impl("argmin.out", &argmin_out_vulkan);
    m.impl("_cummax_helper", &cummax_helper_vulkan);
    m.impl("_cummin_helper", &cummin_helper_vulkan);
    m.impl("var.correction", &var_correction_vulkan);
    m.impl("std.correction", &std_correction_vulkan);
    m.impl("var_mean.correction", &var_mean_correction_vulkan);
    m.impl("std_mean.correction", &std_mean_correction_vulkan);

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

    // composite ops
    m.impl("dot", &dot_vulkan);
    m.impl("vdot", &dot_vulkan);
    m.impl("addmv.out", &addmv_out_vulkan);
    m.impl("linalg_vector_norm.out", &linalg_vector_norm_out_vulkan);
    m.impl("norm.out", &norm_scalaropt_dim_out_vulkan);
    m.impl("renorm.out", &renorm_out_vulkan);
    m.impl("softshrink.out", &softshrink_out_vulkan);
    m.impl("hardshrink.out", &hardshrink_out_vulkan);
    m.impl("threshold.out", &threshold_out_vulkan);
    m.impl("glu.out", &glu_out_vulkan);
    m.impl("_prelu_kernel", &prelu_vulkan);
    m.impl("log_sigmoid_forward", &log_sigmoid_forward_vulkan);
    m.impl("huber_loss", &huber_loss_vulkan);
    m.impl("smooth_l1_loss.out", &smooth_l1_loss_out_vulkan);
    m.impl("mse_loss.out", &mse_loss_out_vulkan);
    m.impl("binary_cross_entropy", &binary_cross_entropy_vulkan);
    m.impl("soft_margin_loss", &soft_margin_loss_vulkan);
    m.impl("special_xlog1py.out", &special_xlog1py_out_vulkan);
    m.impl("trace", &trace_vulkan);
    m.impl("cat.out", &cat_out_vulkan);
    m.impl("roll", &roll_vulkan);
    m.impl("masked_fill_.Scalar", &masked_fill_scalar_vulkan_);
    m.impl("masked_fill_.Tensor", &masked_fill_tensor_vulkan_);
    m.impl("nan_to_num.out", &nan_to_num_out_vulkan);
    m.impl("heaviside.out", &heaviside_out_vulkan);

    // clamp .out variants 
    m.impl("clamp_min.out", &clamp_min_out_vulkan);
    m.impl("clamp_max.out", &clamp_max_out_vulkan);
    m.impl("clamp_min.Tensor_out", &clamp_min_tensor_out_vulkan);
    m.impl("clamp_max.Tensor_out", &clamp_max_tensor_out_vulkan);
    m.impl("clamp.out", &clamp_out_vulkan);
    m.impl("clamp.Tensor_out", &clamp_tensor_out_vulkan);
    // comparison / logical
    m.impl("eq.Tensor_out", &eq_tensor_out_vulkan);
    m.impl("eq.Scalar_out", &eq_scalar_out_vulkan);
    m.impl("ne.Tensor_out", &ne_tensor_out_vulkan);
    m.impl("ne.Scalar_out", &ne_scalar_out_vulkan);
    m.impl("lt.Tensor_out", &lt_tensor_out_vulkan);
    m.impl("lt.Scalar_out", &lt_scalar_out_vulkan);
    m.impl("le.Tensor_out", &le_tensor_out_vulkan);
    m.impl("le.Scalar_out", &le_scalar_out_vulkan);
    m.impl("gt.Tensor_out", &gt_tensor_out_vulkan);
    m.impl("gt.Scalar_out", &gt_scalar_out_vulkan);
    m.impl("ge.Tensor_out", &ge_tensor_out_vulkan);
    m.impl("ge.Scalar_out", &ge_scalar_out_vulkan);
    m.impl("logical_and.out", &logical_and_out_vulkan);
    m.impl("logical_or.out", &logical_or_out_vulkan);
    m.impl("logical_xor.out", &logical_xor_out_vulkan);
    m.impl("logical_not.out", &logical_not_out_vulkan);
    m.impl("isnan", &isnan_vulkan);
    m.impl("isinf.out", &isinf_out_vulkan);
    m.impl("isposinf.out", &isposinf_out_vulkan);
    m.impl("isneginf.out", &isneginf_out_vulkan);

    // where / select
    m.impl("where.self", &where_vulkan);
    m.impl("where.ScalarOther", &where_scalar_other_vulkan);
    m.impl("where.ScalarSelf", &where_scalar_self_vulkan);
    m.impl("where.Scalar", &where_scalar_vulkan);
}