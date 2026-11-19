#include <torch/extension.h>
#include <c10/core/MemoryFormat.h>
#include <ATen/native/Resize.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/ResizeCommon.h>
#include <c10/core/DeviceGuard.h>
#include "vulkan/memory.h"
#include "vulkan/allocator.h"
#include "vulkan/vulkan_context.h"
#include "api/ops/helpers.h"
#include "api/ops/internal.h"

namespace {

namespace {

void fill_deterministic_if_enabled(const at::Tensor& tensor)
{
    if (!at::globalContext().deterministicAlgorithms() || !at::globalContext().deterministicFillUninitializedMemory()) return;
    int64_t numel = static_cast<int64_t>(tensor.storage().nbytes()) / static_cast<int64_t>(tensor.itemsize());
    if (numel == 0) return;
    at::Tensor flat = at::empty({0}, tensor.options());
    flat.set_(tensor.storage(), 0, {numel}, {1});
    at::native::fill_empty_deterministic_(flat);
}

void warn_complex_half(at::ScalarType dtype)
{
    if (dtype == at::kComplexHalf) TORCH_WARN_ONCE("ComplexHalf support is experimental and many operators don't support it yet.");
}

} // namespace

at::Tensor empty_memory_format_vulkan(
    c10::SymIntArrayRef size,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> /* pin_memory */,
    c10::optional<at::MemoryFormat> memory_format) 
{
    // the allocator places memory on the current device, so make that the requested one
    c10::OptionalDeviceGuard guard;
    if (device.has_value() && device->has_index()) guard.reset_device(*device);

    at::ScalarType dtype_ = dtype.value_or(c10::typeMetaToScalarType(at::get_default_dtype()));  // torch.set_default_dtype
    at::Layout layout_ = layout.value_or(at::kStrided);
    at::MemoryFormat memory_format_ = memory_format.value_or(at::MemoryFormat::Contiguous);
    warn_complex_half(dtype_);

    TORCH_CHECK(layout_ == at::kStrided, "torchvulkan [ERROR]: Only Strided layout is supported");
    
    std::vector<int64_t> concrete_sizes;
    concrete_sizes.reserve(size.size());
    for (const auto& s : size) {
        concrete_sizes.push_back(s.guard_int(__FILE__, __LINE__));
    }
    
    at::detail::check_size_nonnegative(concrete_sizes);
    size_t nbytes = at::detail::computeStorageNbytesContiguous(concrete_sizes, c10::elementSize(dtype_));

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
    fill_deterministic_if_enabled(tensor);

    return tensor;
}

at::Tensor empty_strided_vulkan(
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> /* layout */, // assumed to be strided, so ignored
    c10::optional<at::Device> device,
    c10::optional<bool> /* pin_memory */) 
{
    c10::OptionalDeviceGuard guard;
    if (device.has_value() && device->has_index()) guard.reset_device(*device);

    at::ScalarType dtype_ = dtype.value_or(c10::typeMetaToScalarType(at::get_default_dtype()));  // torch.set_default_dtype
    
    std::vector<int64_t> concrete_sizes;
    std::vector<int64_t> concrete_strides;
    warn_complex_half(dtype_);
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
    
    at::detail::check_size_nonnegative(concrete_sizes);
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
    fill_deterministic_if_enabled(tensor);

    return tensor;
}

} // namespace

void torchvulkan::resize_bytes_vulkan(c10::StorageImpl* storage, size_t nbytes)
{
    TORCH_CHECK(storage->resizable(), "torchvulkan [ERROR]: Trying to resize storage that is not resizable.");

    // allocate on the storage's own device, not the current one
    c10::OptionalDeviceGuard guard;
    const c10::Device device = storage->data_ptr().device();
    if (device.type() == c10::DeviceType::PrivateUse1 && device.has_index()) guard.reset_device(device);

    at::DataPtr new_data = storage->allocator()->allocate(nbytes);

    size_t copy_bytes = std::min(nbytes, storage->nbytes());
    if (copy_bytes > 0 && storage->data() != nullptr) {
        globalVulkanAllocator.copy_device_to_device(new_data.get(), /* dest_offset = */ 0, storage->data(), /* src_offset = */ 0, copy_bytes);
    }

    storage->set_data_ptr_noswap(std::move(new_data));
    storage->set_nbytes(nbytes);
}

const at::Tensor& torchvulkan::resize_vulkan(
    const at::Tensor& self, 
    c10::IntArrayRef size, 
    c10::optional<at::MemoryFormat> memory_format) 
{
    at::TensorImpl* impl = self.unsafeGetTensorImpl();

    at::detail::check_size_nonnegative(size);
    if (impl->sizes() == size && !memory_format.has_value()) return self;
    size_t new_bytes = at::detail::computeStorageNbytesContiguous(size, self.itemsize(), impl->storage_offset());

    if (impl->storage().nbytes() < new_bytes) {
        int64_t old_bytes = static_cast<int64_t>(impl->storage().nbytes());
        torchvulkan::resize_bytes_vulkan(impl->storage().unsafeGetStorageImpl(), new_bytes);
        if (at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory()) {
            at::native::fill_resize_deterministic_(self, old_bytes);
        }
    }

    at::MemoryFormat format = memory_format.value_or(at::MemoryFormat::Contiguous);
    std::vector<int64_t> strides = compute_strides(size, format);
    impl->set_sizes_and_strides(size, strides);

    return self;
}

namespace {

at::Tensor& set_source_storage_offset_vulkan(
    at::Tensor& self,
    c10::Storage source,
    int64_t storage_offset,
    c10::IntArrayRef size,
    c10::IntArrayRef stride)
{
    at::native::checkSetStorage(self, std::move(source), storage_offset, size, stride);

    std::vector<int64_t> strides = stride.empty() ? compute_strides(size, at::MemoryFormat::Contiguous) : stride.vec();
    size_t needed = at::detail::computeStorageNbytes(size, strides, self.itemsize(), storage_offset);
    TORCH_CHECK(needed <= self.storage().nbytes(), 
        "torchvulkan [ERROR]: set_ needs ", needed, " bytes, but the storage only has ", self.storage().nbytes(), "."
    );

    self.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
    self.unsafeGetTensorImpl()->set_sizes_and_strides(size, strides);
    return self;
}

at::Tensor& set_source_storage_vulkan(at::Tensor& self, c10::Storage source)
{
    int64_t numel = static_cast<int64_t>(source.nbytes() / self.itemsize());
    return set_source_storage_offset_vulkan(self, std::move(source), 0, {numel}, {});
}

at::Tensor& set_source_tensor_vulkan(at::Tensor& self, const at::Tensor& source)
{
    if (self.unsafeGetTensorImpl() == source.unsafeGetTensorImpl()) return self;
    return set_source_storage_offset_vulkan(self, source.storage(), source.storage_offset(), source.sizes(), source.strides());
}

at::Tensor& set_empty_vulkan(at::Tensor& self)
{
    c10::Storage storage(c10::Storage::use_byte_size_t(), 0, c10::GetAllocator(c10::DeviceType::PrivateUse1), /* resizable = */ true);
    return set_source_storage_offset_vulkan(self, std::move(storage), 0, {0}, {});
}

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
    m.impl("set_.source_Storage_storage_offset", &set_source_storage_offset_vulkan);
    m.impl("set_.source_Storage", &set_source_storage_vulkan);
    m.impl("set_.source_Tensor", &set_source_tensor_vulkan);
    m.impl("set_", &set_empty_vulkan);
}
