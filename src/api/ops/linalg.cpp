#include <torch/extension.h>
#include "api/ops/linalg.h"

at::Tensor torchvulkan::addr_vulkan(
    const at::Tensor& self,
    const at::Tensor& vec1,
    const at::Tensor& vec2,
    const at::Scalar& beta,
    const at::Scalar& alpha)
{
    // TODO: implement this properly
    return at::addr(self.to(at::kCPU), vec1.to(at::kCPU), vec2.to(at::kCPU), beta, alpha).to(self.device());
}

at::Tensor torchvulkan::bilinear_vulkan(
    const at::Tensor& input1,
    const at::Tensor& input2,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias)
{
    // TODO: implement this properly
    c10::optional<at::Tensor> bias_cpu = bias.has_value() ? c10::optional<at::Tensor>(bias->to(at::kCPU)) : c10::nullopt;
    at::Tensor result = at::bilinear(input1.to(at::kCPU), input2.to(at::kCPU), weight.to(at::kCPU), bias_cpu);
    return result.to(input1.device());
}
