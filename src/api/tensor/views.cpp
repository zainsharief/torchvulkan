#include <torch/extension.h>
#include <ATen/InferSize.h>
#include <ATen/WrapDimUtils.h>
#include <c10/core/MemoryFormat.h>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "api/ops/helpers.h"
#include "api/ops/internal.h"

namespace {

at::Tensor as_strided_vulkan(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    c10::optional<c10::SymInt> storage_offset)
{
    std::vector<int64_t> concrete_sizes;
    std::vector<int64_t> concrete_strides;
    concrete_sizes.reserve(size.size());
    concrete_strides.reserve(stride.size());
    
    for (const auto& s : size) {
        concrete_sizes.push_back(s.guard_int(__FILE__, __LINE__));
    }
    for (const auto& s : stride) {
        concrete_strides.push_back(s.guard_int(__FILE__, __LINE__));
    }
    
    int64_t offset = storage_offset.has_value() 
        ? storage_offset.value().guard_int(__FILE__, __LINE__) 
        : self.storage_offset();

    at::Tensor result = at::detail::make_tensor<at::TensorImpl>(
        c10::TensorImpl::VIEW,
        c10::Storage(self.storage()),
        self.key_set(),
        self.dtype()
    );
    
    // update the sizes, strides, and offset in the view
    result.unsafeGetTensorImpl()->set_sizes_and_strides(concrete_sizes, concrete_strides);
    result.unsafeGetTensorImpl()->set_storage_offset(offset);
    
    return result;
}

at::Tensor reshape_alias_vulkan(
    const at::Tensor& self, 
    c10::SymIntArrayRef sizes, 
    c10::SymIntArrayRef strides) 
{
    auto result = at::detail::make_tensor<c10::TensorImpl>(
        c10::TensorImpl::VIEW,
        c10::Storage(self.storage()),
        self.key_set(),
        self.dtype()
    );
    
    auto* result_impl = result.unsafeGetTensorImpl();
    result_impl->set_storage_offset(self.storage_offset());
    result_impl->set_sizes_and_strides(sizes, strides);
    
    return result;
}

at::Tensor view_vulkan(
    const at::Tensor& self,
    c10::SymIntArrayRef size)
{
    std::vector<int64_t> concrete_sizes;
    concrete_sizes.reserve(size.size());
    for (const auto& s : size) {
        concrete_sizes.push_back(s.guard_int(__FILE__, __LINE__));
    }

    at::DimVector inferred_sizes = at::infer_size_dv(concrete_sizes, self.numel());
    concrete_sizes.assign(inferred_sizes.begin(), inferred_sizes.end());

    auto stride = at::detail::computeStride(self.sizes(), self.strides(), concrete_sizes);
    TORCH_CHECK(stride.has_value(), "torchvulkan [ERROR]: View size is not compatible with input tensor's size and stride.");
    
    at::Tensor result = self.alias();
    result.unsafeGetTensorImpl()->set_sizes_and_strides(concrete_sizes, stride.value());
    result.unsafeGetTensorImpl()->set_storage_offset(self.storage_offset());
    
    return result;
}

at::Tensor transpose_int_vulkan(const at::Tensor& self, int64_t dim0, int64_t dim1)
{
    int64_t ndim = self.dim();
    dim0 = c10::maybe_wrap_dim(dim0, ndim);
    dim1 = c10::maybe_wrap_dim(dim1, ndim);

    if (dim0 == dim1) return self.alias();

    std::vector<int64_t> sizes(self.sizes().begin(), self.sizes().end());
    std::vector<int64_t> strides(self.strides().begin(), self.strides().end());
    std::swap(sizes[dim0], sizes[dim1]);
    std::swap(strides[dim0], strides[dim1]);

    return self.as_strided(sizes, strides, self.storage_offset());
}

at::Tensor t_vulkan(const at::Tensor& self)
{
    TORCH_CHECK(self.dim() <= 2, "torchvulkan [ERROR]: t() expects a tensor with <= 2 dimensions.");
    if (self.dim() < 2) return self.alias();
    return transpose_int_vulkan(self, 0, 1);
}

at::Tensor permute_vulkan(const at::Tensor& self, c10::IntArrayRef dims)
{
    int64_t ndim = self.dim();
    TORCH_CHECK((int64_t)dims.size() == ndim, "torchvulkan [ERROR]: permute dims size does not match tensor dimensions.");

    std::vector<int64_t> sizes(ndim);
    std::vector<int64_t> strides(ndim);
    for (int64_t i = 0; i < ndim; i++) {
        int64_t d = c10::maybe_wrap_dim(dims[i], ndim);
        sizes[i] = self.size(d);
        strides[i] = self.stride(d);
    }

    return self.as_strided(sizes, strides, self.storage_offset());
}

at::Tensor contiguous_vulkan(const at::Tensor& self, at::MemoryFormat memory_format)
{
    if (self.is_contiguous(memory_format)) return self;
    at::Tensor result = at::empty_like(self, self.options().memory_format(memory_format));

    // call directly rather than through dispatcher to avoid weird fallbacks that trigger infinite recursive loops
    return torchvulkan::copy_vulkan(self, result, false); 
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("as_strided", &as_strided_vulkan);
    m.impl("_reshape_alias", &reshape_alias_vulkan);
    m.impl("view", &view_vulkan);
    m.impl("t", &t_vulkan);
    m.impl("transpose.int", &transpose_int_vulkan);
    m.impl("permute", &permute_vulkan);
    m.impl("contiguous", &contiguous_vulkan);
}
