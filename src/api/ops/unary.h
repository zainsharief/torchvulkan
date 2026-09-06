#pragma once
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
    RECIPROCAL = 5,
    SIN = 6,
    COS = 7,
    TAN = 8,
    ASIN = 9,
    ACOS = 10,
    ATAN = 11,
    SINH = 12,
    COSH = 13,
    TANH = 14,
    EXP2 = 15,
    LOG2 = 16,
    LOG10 = 17,
    EXPM1 = 18,
    LOG1P = 19,
    RSQRT = 20,
    SIGMOID = 21,
    ASINH = 22,
    ACOSH = 23,
    ATANH = 24,
    DEG2RAD = 25,
    RAD2DEG = 26,
    FLOOR = 27,
    CEIL = 28,
    TRUNC = 29,
    ROUND = 30,
    FRAC = 31,
    ABS = 32,
    SIGN = 33,
    ERF = 34,
    ERFINV = 35,
    SINC = 36,
    ENTR = 37,
    LGAMMA = 38,
    DIGAMMA = 39,
    I0 = 40,
    I1 = 41
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
at::Tensor sin_vulkan(const at::Tensor& self);
at::Tensor cos_vulkan(const at::Tensor& self);
at::Tensor tan_vulkan(const at::Tensor& self);
at::Tensor asin_vulkan(const at::Tensor& self);
at::Tensor acos_vulkan(const at::Tensor& self);
at::Tensor atan_vulkan(const at::Tensor& self);
at::Tensor sinh_vulkan(const at::Tensor& self);
at::Tensor cosh_vulkan(const at::Tensor& self);
at::Tensor tanh_vulkan(const at::Tensor& self);
at::Tensor exp2_vulkan(const at::Tensor& self);
at::Tensor log2_vulkan(const at::Tensor& self);
at::Tensor log10_vulkan(const at::Tensor& self);
at::Tensor expm1_vulkan(const at::Tensor& self);
at::Tensor log1p_vulkan(const at::Tensor& self);
at::Tensor rsqrt_vulkan(const at::Tensor& self);
at::Tensor sigmoid_vulkan(const at::Tensor& self);
at::Tensor asinh_vulkan(const at::Tensor& self);
at::Tensor acosh_vulkan(const at::Tensor& self);
at::Tensor atanh_vulkan(const at::Tensor& self);
at::Tensor deg2rad_vulkan(const at::Tensor& self);
at::Tensor rad2deg_vulkan(const at::Tensor& self);
at::Tensor floor_vulkan(const at::Tensor& self);
at::Tensor ceil_vulkan(const at::Tensor& self);
at::Tensor trunc_vulkan(const at::Tensor& self);
at::Tensor round_vulkan(const at::Tensor& self);
at::Tensor frac_vulkan(const at::Tensor& self);
at::Tensor abs_vulkan(const at::Tensor& self);
at::Tensor sign_vulkan(const at::Tensor& self);
at::Tensor erf_vulkan(const at::Tensor& self);
at::Tensor erfinv_vulkan(const at::Tensor& self);
at::Tensor sinc_vulkan(const at::Tensor& self);
at::Tensor entr_vulkan(const at::Tensor& self);
at::Tensor lgamma_vulkan(const at::Tensor& self);
at::Tensor digamma_vulkan(const at::Tensor& self);
at::Tensor i0_vulkan(const at::Tensor& self);
at::Tensor i1_vulkan(const at::Tensor& self);
at::Tensor silu_vulkan(const at::Tensor& self);
at::Tensor hardtanh_vulkan(const at::Tensor& self, const at::Scalar& min_val, const at::Scalar& max_val);
at::Tensor hardsigmoid_vulkan(const at::Tensor& self);
at::Tensor hardswish_vulkan(const at::Tensor& self);
at::Tensor leaky_relu_vulkan(const at::Tensor& self, const at::Scalar& negative_slope);
at::Tensor elu_vulkan(const at::Tensor& self, const at::Scalar& alpha, const at::Scalar& scale, const at::Scalar& input_scale);
at::Tensor softplus_vulkan(const at::Tensor& self, const at::Scalar& beta, const at::Scalar& threshold);
at::Tensor mish_vulkan(const at::Tensor& self);
at::Tensor tanhshrink_vulkan(const at::Tensor& self);
at::Tensor square_vulkan(const at::Tensor& self);
at::Tensor logit_vulkan(const at::Tensor& self, c10::optional<double> eps);

at::Tensor unary_op_vulkan(
    const at::Tensor& self,
    UnaryOp operation,
    const std::function<at::Tensor(const at::Tensor&)>& fallback
);

} // namespace torchvulkan

