#include <iostream>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "shaders/shader_registry.h"
#include "helpers.h"

#include <c10/core/MemoryFormat.h>

namespace torchvulkan {

at::Tensor empty_memory_format_vulkan(
    c10::SymIntArrayRef size,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> /* device */,
    c10::optional<bool> /* pin_memory */,
    c10::optional<at::MemoryFormat> memory_format
);

at::Tensor empty_strided_vulkan(
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> /* layout */,
    c10::optional<at::Device> /* device */,
    c10::optional<bool> /* pin_memory */
);

at::Tensor copy_from_vulkan(
    const at::Tensor& self, 
    const at::Tensor& dst, 
    bool non_blocking
); 

at::Tensor copy_from_and_resize_vulkan(
    const at::Tensor& self, 
    const at::Tensor& dst
);

at::Tensor& copy_vulkan_(
    at::Tensor& self, 
    const at::Tensor& src, 
    bool non_blocking
);

at::Tensor as_strided_vulkan(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    c10::optional<c10::SymInt> storage_offset
);

const at::Tensor& resize_vulkan(
    const at::Tensor& self, 
    c10::IntArrayRef size, 
    c10::optional<at::MemoryFormat> memory_format
);

at::Tensor view_vulkan(
    const at::Tensor& self,
    c10::SymIntArrayRef size
);

at::Tensor contiguous_vulkan(const at::Tensor& self, at::MemoryFormat memory_format);
at::Tensor clone_vulkan(const at::Tensor& self, c10::optional<at::MemoryFormat> memory_format);

void dispatch_copy_shader(const at::Tensor& src, const at::Tensor& dst);
void dispatch_cast_shader(const at::Tensor& src, const at::Tensor& dst);
at::Tensor copy_vulkan(const at::Tensor& self, const at::Tensor& dst, bool non_blocking);

} // namespace torchvulkan