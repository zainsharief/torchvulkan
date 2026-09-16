#pragma once
#include <torch/extension.h>
#include <c10/core/MemoryFormat.h>

namespace torchvulkan {

// ops/binary.cpp
at::Tensor add_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha);
at::Tensor multiply_vulkan(const at::Tensor& self, const at::Tensor& other);
at::Tensor divide_vulkan(const at::Tensor& self, const at::Tensor& other);
at::Tensor divide_scalar_vulkan(const at::Tensor& self, const at::Scalar& other);

// ops/reduce.cpp
at::Tensor sum_vulkan(const at::Tensor& self, c10::optional<at::ScalarType> dtype);

// tensor/allocation.cpp
const at::Tensor& resize_vulkan(const at::Tensor& self, c10::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format);

// tensor/transfer.cpp
at::Tensor copy_vulkan(const at::Tensor& self, const at::Tensor& dst, bool non_blocking);

// tensor/cast.cpp
void dispatch_cast_shader(const at::Tensor& src, const at::Tensor& dst);

} // namespace torchvulkan
