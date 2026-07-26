#include <torch/extension.h>

namespace torchvulkan {

at::Tensor log_softmax_vulkan(const at::Tensor& self, int64_t dim, bool half_to_float);

at::Tensor log_softmax_backward_vulkan(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    at::ScalarType input_dtype
);

} // namespace torchvulkan
