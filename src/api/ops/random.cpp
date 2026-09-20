#include <torch/extension.h>

namespace {

at::Tensor& normal_vulkan_(at::Tensor& self, double mean, double stddev, c10::optional<at::Generator> generator)
{
    at::Tensor cpu = at::empty(self.sizes(), self.options().device(at::kCPU));
    cpu.normal_(mean, stddev, generator);
    self.copy_(cpu);
    return self;
}

at::Tensor& uniform_vulkan_(at::Tensor& self, double from, double to, c10::optional<at::Generator> generator)
{
    at::Tensor cpu = at::empty(self.sizes(), self.options().device(at::kCPU));
    cpu.uniform_(from, to, generator);
    self.copy_(cpu);
    return self;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("normal_", &normal_vulkan_);
    m.impl("uniform_", &uniform_vulkan_);
}
