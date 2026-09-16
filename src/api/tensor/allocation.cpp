#include <torch/extension.h>
#include <c10/core/MemoryFormat.h>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "api/ops/helpers.h"
#include "api/ops/internal.h"

namespace {

at::Tensor empty_memory_format_vulkan(
    c10::SymIntArrayRef size,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> /* device */,
    c10::optional<bool> /* pin_memory */,
    c10::optional<at::MemoryFormat> memory_format) 
{
    at::ScalarType dtype_ = dtype.value_or(at::kFloat);
    at::Layout layout_ = layout.value_or(at::kStrided);
    at::MemoryFormat memory_format_ = memory_format.value_or(at::MemoryFormat::Contiguous);

    TORCH_CHECK(layout_ == at::kStrided, "torchvulkan [ERROR]: Only Strided layout is supported");
    
    std::vector<int64_t> concrete_sizes;
    concrete_sizes.reserve(size.size());
    for (const auto& s : size) {
        concrete_sizes.push_back(s.guard_int(__FILE__, __LINE__));
    }
    
    int64_t numel = c10::multiply_integers(concrete_sizes);
    size_t nbytes = numel * c10::elementSize(dtype_);

    std::vector<int64_t> strides = compute_strides(concrete_sizes, memory_format_);

    auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        nbytes,
        c10::GetAllocator(c10::DeviceType::PrivateUse1),
        /* resizeable = */ true 
    );
    
    at::Tensor tensor = at::detail::make_tensor<at::TensorImpl>(
        std::move(storage_impl),
        c10::DispatchKey::PrivateUse1,
        c10::scalarTypeToTypeMeta(dtype_)
    );
    
    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(concrete_sizes, strides);

    return tensor;
}

at::Tensor empty_strided_vulkan(
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> /* layout */, // assumed to be strided, so ignored
    c10::optional<at::Device> /* device */,
    c10::optional<bool> /* pin_memory */) 
{
    at::ScalarType dtype_ = dtype.value_or(at::kFloat);
    
    std::vector<int64_t> concrete_sizes;
    std::vector<int64_t> concrete_strides;
    concrete_sizes.reserve(size.size());
    concrete_strides.reserve(stride.size());
    
    for (const auto& s : size) {
        int64_t val = s.guard_int(__FILE__, __LINE__);
        concrete_sizes.push_back(val);
    }
    for (const auto& s : stride) {
        int64_t val = s.guard_int(__FILE__, __LINE__);
        concrete_strides.push_back(val);
    }
    
    size_t nbytes = at::detail::computeStorageNbytes(
        concrete_sizes, 
        concrete_strides, 
        c10::elementSize(dtype_)
    );

    auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        nbytes,
        c10::GetAllocator(c10::DeviceType::PrivateUse1),
        /* resizeable = */ true 
    );
    
    at::Tensor tensor = at::detail::make_tensor<at::TensorImpl>(
        std::move(storage_impl),
        c10::DispatchKey::PrivateUse1,
        c10::scalarTypeToTypeMeta(dtype_)
    );
    
    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(concrete_sizes, concrete_strides);

    return tensor;
}

} // namespace

const at::Tensor& torchvulkan::resize_vulkan(
    const at::Tensor& self, 
    c10::IntArrayRef size, 
    c10::optional<at::MemoryFormat> memory_format) 
{
    at::TensorImpl* impl = self.unsafeGetTensorImpl();

    int64_t numel = 1;
    for (auto s : size) numel *= s;
    size_t new_bytes = numel * self.itemsize();

    if (impl->storage().nbytes() < new_bytes) {
        auto new_storage = c10::make_intrusive<c10::StorageImpl>(
            c10::StorageImpl::use_byte_size_t(),
            new_bytes,
            c10::GetAllocator(c10::DeviceType::PrivateUse1),
            /* resizeable = */ true
        );
        impl->set_storage_keep_dtype(std::move(new_storage));
    }

    at::MemoryFormat format = memory_format.value_or(at::MemoryFormat::Contiguous);
    std::vector<int64_t> strides = compute_strides(size, format);
    impl->set_sizes_and_strides(size, strides);

    return self;
}

namespace {

at::Tensor clone_vulkan(const at::Tensor& self, c10::optional<at::MemoryFormat> memory_format) 
{
    c10::TensorOptions options = self.options();
    if (memory_format.has_value()) options = options.memory_format(memory_format.value());
    at::Tensor result = at::empty_like(self, options);

    // call directly rather than through dispatcher to avoid weird fallbacks that trigger infinite recursive loops
    return torchvulkan::copy_vulkan(self, result, false);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("empty.memory_format", &empty_memory_format_vulkan);
    m.impl("empty_strided", &empty_strided_vulkan);
    m.impl("resize_", &torchvulkan::resize_vulkan);
    m.impl("clone", &clone_vulkan);
}
