#include <torch/extension.h>
#include <ATen/native/CPUFallback.h>
#include "api/fallback.h"

void vulkan_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack)
{
    TORCHVULKAN_FALLBACK("Silent fallback detected for operation: ", op.schema().operator_name(), ".");
    at::native::cpu_fallback(op, stack);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&vulkan_cpu_fallback>());
}
