#include <torch/extension.h>
#include "api/ops/softmax.h"

at::Tensor torchvulkan::log_softmax_vulkan(const at::Tensor& self, int64_t dim, bool half_to_float)
{
    at::Tensor input = half_to_float ? self.to(at::kFloat) : self;
    if (input.numel() == 0) return input.clone();

    at::Tensor self_max = at::amax(input, {dim}, /*keepdim=*/true);
    at::Tensor shifted = at::sub(input, self_max);
    at::Tensor sum_exp = at::sum(at::exp(shifted), {dim}, /*keepdim=*/true);
    return at::sub(shifted, at::log(sum_exp));
}

at::Tensor torchvulkan::log_softmax_backward_vulkan(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    at::ScalarType /* input_dtype */)
{
    if (grad_output.numel() == 0) return grad_output.clone();

    at::Tensor sum_grad = at::sum(grad_output, {dim}, /*keepdim=*/true);
    at::Tensor softmax = at::exp(output);
    return at::sub(grad_output, at::mul(softmax, sum_grad));
}

at::Tensor torchvulkan::softmax_vulkan(const at::Tensor& self, int64_t dim, bool half_to_float)
{
    return at::exp(log_softmax_vulkan(self, dim, half_to_float));
}

at::Tensor torchvulkan::softmax_backward_vulkan(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    at::ScalarType /* input_dtype */)
{
    if (grad_output.numel() == 0) return grad_output.clone();

    at::ScalarType out_type = grad_output.scalar_type();
    bool upcast = out_type == at::kHalf || out_type == at::kBFloat16;
    at::Tensor go = upcast ? grad_output.to(at::kFloat) : grad_output;
    at::Tensor out = upcast ? output.to(at::kFloat) : output;

    at::Tensor inner = at::sum(at::mul(go, out), {dim}, /*keepdim=*/true);
    at::Tensor res = at::mul(out, at::sub(go, inner));
    return upcast ? res.to(out_type) : res;
}
