#include <torch/extension.h>
#include <ATen/native/CPUFallback.h>
#include "vulkan/vulkan_context.h"

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
