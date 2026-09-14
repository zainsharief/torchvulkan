#pragma once
#include <torch/extension.h>
#include <array>
#include <tuple>

namespace torchvulkan {

at::Tensor dot_vulkan(const at::Tensor& self, const at::Tensor& other);
at::Tensor& addmv_out_vulkan(const at::Tensor& self, const at::Tensor& mat, const at::Tensor& vec, const at::Scalar& beta, const at::Scalar& alpha, at::Tensor& out);
at::Tensor& linalg_vector_norm_out_vulkan(const at::Tensor& self, const at::Scalar& ord, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor& out);
at::Tensor& norm_scalaropt_dim_out_vulkan(const at::Tensor& self, const c10::optional<at::Scalar>& p, at::IntArrayRef dim, bool keepdim, at::Tensor& out);
at::Tensor& renorm_out_vulkan(const at::Tensor& self, const at::Scalar& p, int64_t dim, const at::Scalar& maxnorm, at::Tensor& out);
at::Tensor& softshrink_out_vulkan(const at::Tensor& self, const at::Scalar& lambd, at::Tensor& out);
at::Tensor& hardshrink_out_vulkan(const at::Tensor& self, const at::Scalar& lambd, at::Tensor& out);
at::Tensor& threshold_out_vulkan(const at::Tensor& self, const at::Scalar& threshold, const at::Scalar& value, at::Tensor& out);
at::Tensor& glu_out_vulkan(const at::Tensor& self, int64_t dim, at::Tensor& out);
at::Tensor prelu_vulkan(const at::Tensor& self, const at::Tensor& weight);
std::tuple<at::Tensor, at::Tensor> log_sigmoid_forward_vulkan(const at::Tensor& self);
at::Tensor huber_loss_vulkan(const at::Tensor& self, const at::Tensor& target, int64_t reduction, double delta);
at::Tensor& smooth_l1_loss_out_vulkan(const at::Tensor& self, const at::Tensor& target, int64_t reduction, double beta, at::Tensor& out);
at::Tensor& mse_loss_out_vulkan(const at::Tensor& self, const at::Tensor& target, int64_t reduction, at::Tensor& out);
at::Tensor binary_cross_entropy_vulkan(const at::Tensor& self, const at::Tensor& target, const c10::optional<at::Tensor>& weight, int64_t reduction);
at::Tensor soft_margin_loss_vulkan(const at::Tensor& self, const at::Tensor& target, int64_t reduction);
at::Tensor& special_xlog1py_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out);
at::Tensor trace_vulkan(const at::Tensor& self);
at::Tensor& gelu_out_vulkan(const at::Tensor& self, c10::string_view approximate, at::Tensor& out);
at::Tensor gelu_backward_vulkan(const at::Tensor& grad_output, const at::Tensor& self, c10::string_view approximate);
at::Tensor& erfc_out_vulkan(const at::Tensor& self, at::Tensor& out);
at::Tensor& log_ndtr_out_vulkan(const at::Tensor& self, at::Tensor& out);
at::Tensor& i0e_out_vulkan(const at::Tensor& self, at::Tensor& out);
at::Tensor& i1e_out_vulkan(const at::Tensor& self, at::Tensor& out);
at::Tensor& erfcx_out_vulkan(const at::Tensor& self, at::Tensor& out);
at::Tensor& lgamma_out_vulkan(const at::Tensor& self, at::Tensor& out);
at::Tensor& masked_fill_scalar_vulkan_(at::Tensor& self, const at::Tensor& mask, const at::Scalar& value);
at::Tensor& masked_fill_tensor_vulkan_(at::Tensor& self, const at::Tensor& mask, const at::Tensor& value);
at::Tensor& nan_to_num_out_vulkan(const at::Tensor& self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf, at::Tensor& out);
at::Tensor& heaviside_out_vulkan(const at::Tensor& self, const at::Tensor& values, at::Tensor& out);
at::Tensor& cat_out_vulkan(const at::ITensorListRef& tensors, int64_t dim, at::Tensor& out);
at::Tensor roll_vulkan(const at::Tensor& self, at::IntArrayRef shifts, at::IntArrayRef dims);
at::Tensor& clamp_min_out_vulkan(const at::Tensor& self, const at::Scalar& min, at::Tensor& out);
at::Tensor& clamp_max_out_vulkan(const at::Tensor& self, const at::Scalar& max, at::Tensor& out);
at::Tensor& clamp_min_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& min, at::Tensor& out);
at::Tensor& clamp_max_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& max, at::Tensor& out);
at::Tensor& clamp_out_vulkan(const at::Tensor& self, const c10::optional<at::Scalar>& min, const c10::optional<at::Scalar>& max, at::Tensor& out);
at::Tensor& clamp_tensor_out_vulkan(const at::Tensor& self, const c10::optional<at::Tensor>& min, const c10::optional<at::Tensor>& max, at::Tensor& out);

int64_t fused_sdp_choice_vulkan(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const c10::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    c10::optional<double> scale,
    bool enable_gqa);

at::Tensor embedding_vulkan(
    const at::Tensor& weight,
    const at::Tensor& indices,
    c10::SymInt padding_idx,
    bool scale_grad_by_freq,
    bool sparse);

at::Tensor embedding_dense_backward_vulkan(
    const at::Tensor& grad_output,
    const at::Tensor& indices,
    c10::SymInt num_weights,
    c10::SymInt padding_idx,
    bool scale_grad_by_freq);

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_layer_norm_vulkan(
    const at::Tensor& input,
    c10::SymIntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    double eps);

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_layer_norm_backward_vulkan(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    c10::SymIntArrayRef normalized_shape,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    std::array<bool, 3> output_mask);

at::Tensor& normal_vulkan_(at::Tensor& self, double mean, double stddev, c10::optional<at::Generator> generator);
at::Tensor& uniform_vulkan_(at::Tensor& self, double from, double to, c10::optional<at::Generator> generator);

} // namespace torchvulkan
