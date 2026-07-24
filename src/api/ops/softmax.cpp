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
