#include <iostream>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "shaders/shader_registry.h"
#include "helpers.h"

#include <c10/core/MemoryFormat.h>

namespace torchvulkan {

at::Tensor dispatch_matmul_shader(
    const at::Tensor& self, 
    const at::Tensor& other,
    const at::Tensor& bias,
    const at::Scalar& alpha,
    const at::Scalar& beta,
    std::function<at::Tensor()> cpu_fallback
);

at::Tensor dispatch_matmul_coop_shader(
    const at::Tensor& self, 
    const at::Tensor& other,
    const at::Tensor& bias_,
    const at::Scalar& alpha,
    const at::Scalar& beta,
    std::function<at::Tensor()> cpu_fallback
);

at::Tensor dispatch_matmul_simd_shader(
    const at::Tensor& self, 
    const at::Tensor& other,
    const at::Tensor& bias_,
    const at::Scalar& alpha,
    const at::Scalar& beta,
    std::function<at::Tensor()> cpu_fallback
);

at::Tensor mm_vulkan(
    const at::Tensor& self, 
    const at::Tensor& other
);

at::Tensor bmm_vulkan(
    const at::Tensor& self, 
    const at::Tensor& other
);

at::Tensor matmul_vulkan(
    const at::Tensor& self, 
    const at::Tensor& other
);

at::Tensor addmm_vulkan(
    const at::Tensor& input, 
    const at::Tensor& mat1, 
    const at::Tensor& mat2, 
    const at::Scalar& beta, 
    const at::Scalar& alpha
);

at::Tensor baddbmm_vulkan(
    const at::Tensor& input, 
    const at::Tensor& mat1, 
    const at::Tensor& mat2, 
    const at::Scalar& beta, 
    const at::Scalar& alpha
);

} // namespace torchvulkan