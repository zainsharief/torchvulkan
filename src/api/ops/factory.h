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

at::Tensor reshape_alias_vulkan(
    const at::Tensor& self, 
    c10::SymIntArrayRef sizes, 
    c10::SymIntArrayRef strides
); 

at::Tensor view_vulkan(
    const at::Tensor& self,
    c10::SymIntArrayRef size
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

at::Tensor tril_indices_vulkan(
    int64_t row, int64_t col, int64_t offset, 
    c10::optional<at::ScalarType> dtype, 
    c10::optional<at::Layout> layout, 
    c10::optional<at::Device> device, 
    c10::optional<bool> pin_memory
);

at::Tensor triu_indices_vulkan(
    int64_t row, int64_t col, int64_t offset, 
    c10::optional<at::ScalarType> dtype, 
    c10::optional<at::Layout> layout, 
    c10::optional<at::Device> device, 
    c10::optional<bool> pin_memory
);

at::Tensor t_vulkan(const at::Tensor& self);
at::Tensor transpose_int_vulkan(const at::Tensor& self, int64_t dim0, int64_t dim1);
at::Tensor permute_vulkan(const at::Tensor& self, c10::IntArrayRef dims);
at::Scalar local_scalar_dense_vulkan(const at::Tensor& self);
at::Tensor contiguous_vulkan(const at::Tensor& self, at::MemoryFormat memory_format);
at::Tensor clone_vulkan(const at::Tensor& self, c10::optional<at::MemoryFormat> memory_format);
at::Tensor copy_vulkan(const at::Tensor& self, const at::Tensor& dst, bool non_blocking);

at::Tensor& arange_start_out_vulkan(
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step,
    at::Tensor& out
);

at::Tensor& linspace_out_vulkan(
    const at::Scalar& start,
    const at::Scalar& end,
    int64_t steps,
    at::Tensor& out
);

at::Tensor& logspace_out_vulkan(
    const at::Scalar& start,
    const at::Scalar& end,
    int64_t steps,
    double base,
    at::Tensor& out
);

at::Tensor& eye_m_out_vulkan(
    c10::SymInt n,
    c10::SymInt m,
    at::Tensor& out
);

void dispatch_copy_shader(
    const at::Tensor& src, 
    const at::Tensor& dst
);

void dispatch_cast_shader(
    const at::Tensor& src, 
    const at::Tensor& dst
);

} // namespace torchvulkan