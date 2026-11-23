#pragma once
#include <torch/extension.h>
#include <c10/core/MemoryFormat.h>
#include <array>
#include <optional>
#include <tuple>

namespace torchvulkan {

// ops/binary.cpp
at::Tensor add_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha);
at::Tensor multiply_vulkan(const at::Tensor& self, const at::Tensor& other);
at::Tensor divide_vulkan(const at::Tensor& self, const at::Tensor& other);
at::Tensor divide_scalar_vulkan(const at::Tensor& self, const at::Scalar& other);

struct ComplexParts {
    at::Tensor re;
    at::Tensor im;
};
ComplexParts complex_parts(const at::Tensor& z);
at::Tensor complex_from(const ComplexParts& z, at::ScalarType dtype);
ComplexParts complex_scalar_parts(const at::Scalar& s, const at::Tensor& like);
ComplexParts complex_mul(const ComplexParts& a, const ComplexParts& b);
ComplexParts complex_div(const ComplexParts& a, const ComplexParts& b);
at::Tensor complex_abs(const ComplexParts& z);
ComplexParts complex_square(const ComplexParts& z);
ComplexParts complex_exp(const ComplexParts& z);
ComplexParts complex_log(const ComplexParts& z);
ComplexParts complex_sqrt(const ComplexParts& z);
ComplexParts complex_sin(const ComplexParts& z);
ComplexParts complex_cos(const ComplexParts& z);
ComplexParts complex_tan(const ComplexParts& z);
ComplexParts complex_sinh(const ComplexParts& z);
ComplexParts complex_cosh(const ComplexParts& z);
ComplexParts complex_tanh(const ComplexParts& z);
ComplexParts complex_asin(const ComplexParts& z);
ComplexParts complex_acos(const ComplexParts& z);
ComplexParts complex_atan(const ComplexParts& z);
ComplexParts complex_asinh(const ComplexParts& z);
ComplexParts complex_acosh(const ComplexParts& z);
ComplexParts complex_atanh(const ComplexParts& z);
ComplexParts complex_expm1(const ComplexParts& z);
ComplexParts complex_log1p(const ComplexParts& z);
ComplexParts complex_logaddexp(const ComplexParts& x, const ComplexParts& y);

// ops/norm.cpp
std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> batch_norm_shader(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, const at::Tensor& running_mean,
    const at::Tensor& running_var, bool training, double momentum, double eps);
std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> batch_norm_backward_shader(
    const at::Tensor& grad_out, const at::Tensor& input, const at::Tensor& weight, const at::Tensor& running_mean,
    const at::Tensor& running_var, const at::Tensor& save_mean, const at::Tensor& save_invstd, bool train, double eps,
    std::array<bool, 3> output_mask);
std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> layer_norm_shader(
    const at::Tensor& input, size_t normalized_dims, const at::Tensor& weight, const at::Tensor& bias, double eps);
std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> layer_norm_backward_shader(
    const at::Tensor& grad_out, const at::Tensor& input, size_t normalized_dims, const at::Tensor& mean,
    const at::Tensor& rstd, const at::Tensor& weight, const at::Tensor& bias, std::array<bool, 3> output_mask);

// ops/reduce.cpp
at::Tensor sum_vulkan(const at::Tensor& self, c10::optional<at::ScalarType> dtype);

// tensor/allocation.cpp
void resize_bytes_vulkan(c10::StorageImpl* storage, size_t nbytes);
const at::Tensor& resize_vulkan(const at::Tensor& self, c10::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format);

// tensor/transfer.cpp
at::Tensor copy_vulkan(const at::Tensor& self, const at::Tensor& dst, bool non_blocking);

// tensor/cast.cpp
void dispatch_cast_shader(const at::Tensor& src, const at::Tensor& dst);

} // namespace torchvulkan
