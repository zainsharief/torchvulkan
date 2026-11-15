#pragma once
#include <ATen/TensorIterator.h>
#include "vulkan/vulkan_context.h"
#include "vulkan/builders.h"
#include "vulkan/shader.h"
#include <c10/core/Device.h>
#include <c10/core/impl/COW.h>
#include <c10/core/DefaultDtype.h>
#include <c10/core/MemoryFormat.h>
#include <c10/util/strides.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <vector>
#include <array>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <utility>

inline uint64_t get_tensor_address(const at::Tensor& t) 
{
    VulkanBuffer* buffer = extract_buffer(t.storage());
    return buffer->bufferAddress() + extract_buffer_offset(t.storage()) + t.storage_offset() * t.element_size();
}

inline void materialize_cow(const at::Tensor& t)
{
    if (t.defined() && c10::impl::cow::is_cow_data_ptr(t.storage().data_ptr())) {
        c10::impl::cow::materialize_cow(t.storage().unsafeGetStorageImpl());
    }
}

inline uint64_t get_output_address(const at::Tensor& t)
{
    materialize_cow(t);
    return get_tensor_address(t);
}

// the device a vulkan tensor lives on: ops dispatch there, whatever the current device is
inline DeviceContext* get_device_context(const at::Tensor& t)
{
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1, "torchvulkan [ERROR]: Expected a vulkan tensor, but got one on ", t.device(), ".");
    return VulkanContext::Instance().GetDeviceContext(t.device().index());
}

inline bool is_dtype_supported(const DeviceContext* device, at::ScalarType dtype) 
{
    switch (dtype) 
    {
        case at::kDouble: return device->support_float64;
        case at::kLong: return device->support_int64;
        case at::kUInt64: return device->support_int64;
        
        case at::kFloat: return device->support_float32;
        case at::kInt: return device->support_int32;
        case at::kUInt32: return device->support_int32;
        
        case at::kHalf: return device->support_float16;
        case at::kBFloat16: return device->support_bfloat16;
        case at::kShort: return device->support_int16;
        case at::kUInt16: return device->support_int16;
        
        case at::kChar: return device->support_int8; 
        case at::kByte: return device->support_int8; 
        case at::kBool: return device->support_int8; 
        
        default: return false;
    }
}

inline bool is_dtype_supported(const at::Tensor& on, at::ScalarType dtype)
{
    return is_dtype_supported(get_device_context(on), dtype);
}

// the dtype whose byte-width variant moves `dtype`'s elements, for kernels that only copy (copy, index_gather)
inline c10::ScalarType get_dtype_width_variant(c10::ScalarType dtype)
{
    switch (c10::elementSize(dtype))
    {
        case 1: return c10::ScalarType::Byte;
        case 2: return c10::ScalarType::UInt16;
        case 4: return c10::ScalarType::UInt32;
        case 8: return c10::ScalarType::UInt64;
        case 16: return c10::ScalarType::ComplexDouble;
        default: TORCH_CHECK(false, "torchvulkan [ERROR]: no byte-width kernel for ", dtype, " elements.");
    }
}

// copy-only kernels move bytes, so a dtype is usable when its byte-width variant is (complex128 has its own)
inline bool is_dtype_copyable(const at::Tensor& on, c10::ScalarType dtype)
{
    return c10::elementSize(dtype) == 16 || is_dtype_supported(on, get_dtype_width_variant(dtype));
}

// set by scripts/config.py, which compiles the shaders with the same table
inline int get_dtype_vec_size(at::ScalarType dtype) 
{
    size_t bytes = c10::elementSize(dtype);
    return bytes <= 16 ? torchvulkan::VEC_SIZE_FOR_BYTES[bytes] : 1;
}

// enough threads per workgroup to move 1KB, within what the device allows
inline uint32_t get_dtype_workgroup_size(at::ScalarType dtype, uint32_t vecSize, const DeviceContext* device) 
{
    size_t bytes = c10::elementSize(dtype);
    if (bytes == 0 || vecSize == 0) return 1;
    uint32_t size = static_cast<uint32_t>(1024 / (bytes * vecSize));
    size = std::min(size, device->properties.limits.maxComputeWorkGroupInvocations);
    size = std::min(size, device->properties.limits.maxComputeWorkGroupSize[0]);
    return std::max(size, 1u);
}

inline uint32_t get_group_count(uint64_t threads, uint32_t workgroupSize)
{
    uint64_t groups = (threads + workgroupSize - 1) / workgroupSize;
    TORCH_CHECK(groups <= std::numeric_limits<uint32_t>::max(), "torchvulkan [ERROR]: ", threads, " threads is too many for one dispatch.");
    return static_cast<uint32_t>(groups);
}

// Kernels index in uint32 and IntDivider is exact below 2^31, so every element of every operand must be
// reachable with a 31-bit element count and a 32-bit element offset. Larger tensors run on the CPU.
inline bool fits_32bit_indexing(std::initializer_list<at::Tensor> tensors)
{
    for (const at::Tensor& t : tensors)
    {
        if (!t.defined() || t.numel() == 0) continue;
        if (t.numel() > std::numeric_limits<int32_t>::max()) return false;
        uint64_t max_offset = 0;
        for (int64_t d = 0; d < t.dim(); d++)
        {
            if (t.stride(d) < 0) return false;
            max_offset += static_cast<uint64_t>(t.size(d) - 1) * static_cast<uint64_t>(t.stride(d));
        }
        if (max_offset > std::numeric_limits<uint32_t>::max()) return false;
    }
    return true;
}

inline bool fits_32bit_indexing(const at::TensorIterator& iter)
{
    for (int i = 0; i < iter.ntensors(); i++) {
        if (!fits_32bit_indexing({iter.tensor(i)})) return false;
    }
    return true;
}

// The metadata for a kernel that walks a TensorIterator's shape: the shape's sizes and fast-division
// constants, then one array of strides (in elements) per operand listed. An operand of -1 gets zero strides.
inline MetadataBuilder get_iter_metadata(const at::TensorIterator& iter, std::initializer_list<int> operands)
{
    uint32_t ndim = static_cast<uint32_t>(iter.ndim());
    IntDivider sizes;
    for (uint32_t i = 0; i < ndim; i++) sizes.set(i, static_cast<uint32_t>(iter.shape()[i]));

    MetadataBuilder metadata{};
    metadata.push(sizes, ndim);
    for (int operand : operands)
    {
        uint32_t strides[MAX_DIMS] = {0};
        if (operand >= 0) {
            int64_t element_size = iter.element_size(operand);
            for (uint32_t i = 0; i < ndim; i++) strides[i] = static_cast<uint32_t>(iter.strides(operand)[i] / element_size);
        }
        metadata.push_array(strides, ndim);
    }
    return metadata;
}

// The same for a kernel that walks `sizes` directly, innermost dimension first, with each tensor's own strides.
inline MetadataBuilder get_shape_metadata(c10::IntArrayRef sizes, std::initializer_list<c10::IntArrayRef> strides_per_tensor)
{
    uint32_t ndim = static_cast<uint32_t>(sizes.size());
    IntDivider divider;
    for (uint32_t i = 0; i < ndim; i++) divider.set(i, static_cast<uint32_t>(sizes[ndim - 1 - i]));

    MetadataBuilder metadata{};
    metadata.push(divider, ndim);
    for (c10::IntArrayRef tensor_strides : strides_per_tensor)
    {
        uint32_t strides[MAX_DIMS] = {0};
        for (uint32_t i = 0; i < ndim; i++) strides[i] = static_cast<uint32_t>(tensor_strides[ndim - 1 - i]);
        metadata.push_array(strides, ndim);
    }
    return metadata;
}

inline int get_dtype_digits(at::ScalarType dtype)
{
    switch (dtype)
    {
        case at::kDouble: return std::numeric_limits<double>::digits;
        case at::kFloat: return std::numeric_limits<float>::digits;
        case at::kHalf: return std::numeric_limits<c10::Half>::digits;
        case at::kBFloat16: return std::numeric_limits<c10::BFloat16>::digits;

        case at::kBool: return std::numeric_limits<bool>::digits;
        case at::kByte: return std::numeric_limits<uint8_t>::digits;
        case at::kChar: return std::numeric_limits<int8_t>::digits;
        case at::kShort: return std::numeric_limits<int16_t>::digits;
        case at::kUInt16: return std::numeric_limits<uint16_t>::digits;
        case at::kInt: return std::numeric_limits<int32_t>::digits;
        case at::kUInt32: return std::numeric_limits<uint32_t>::digits;
        case at::kLong: return std::numeric_limits<int64_t>::digits;
        case at::kUInt64: return std::numeric_limits<uint64_t>::digits;

        default: TORCH_CHECK(false, "torchvulkan [ERROR]: Unsupported dtype ", dtype, ".");
    }
}

inline std::pair<double, double> get_dtype_limits(at::ScalarType dtype)
{
    switch (dtype)
    {
        case at::kDouble: return {std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max()};
        case at::kFloat: return {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max()};
        case at::kHalf: return {static_cast<double>(std::numeric_limits<c10::Half>::lowest()), static_cast<double>(std::numeric_limits<c10::Half>::max())};
        case at::kBFloat16: return {static_cast<double>(std::numeric_limits<c10::BFloat16>::lowest()), static_cast<double>(std::numeric_limits<c10::BFloat16>::max())};

        case at::kBool: return {0, 1};
        case at::kByte: return {std::numeric_limits<uint8_t>::lowest(), std::numeric_limits<uint8_t>::max()};
        case at::kChar: return {std::numeric_limits<int8_t>::lowest(), std::numeric_limits<int8_t>::max()};
        case at::kShort: return {std::numeric_limits<int16_t>::lowest(), std::numeric_limits<int16_t>::max()};
        case at::kUInt16: return {std::numeric_limits<uint16_t>::lowest(), std::numeric_limits<uint16_t>::max()};
        case at::kInt: return {std::numeric_limits<int32_t>::lowest(), std::numeric_limits<int32_t>::max()};
        case at::kUInt32: return {std::numeric_limits<uint32_t>::lowest(), std::numeric_limits<uint32_t>::max()};
        case at::kLong: return {static_cast<double>(std::numeric_limits<int64_t>::lowest()), static_cast<double>(std::numeric_limits<int64_t>::max())};
        case at::kUInt64: return {0, static_cast<double>(std::numeric_limits<uint64_t>::max())};

        default: TORCH_CHECK(false, "torchvulkan [ERROR]: Unsupported dtype ", dtype, ".");
    }
}

inline double get_dtype_epsilon(at::ScalarType dtype)
{
    switch (dtype)
    {
        case at::kDouble: return std::numeric_limits<double>::epsilon();
        case at::kHalf: return static_cast<double>(std::numeric_limits<c10::Half>::epsilon());
        case at::kBFloat16: return static_cast<double>(std::numeric_limits<c10::BFloat16>::epsilon());
        default: return std::numeric_limits<float>::epsilon();
    }
}

inline double get_dtype_smallest_normal(at::ScalarType dtype)
{
    switch (dtype)
    {
        case at::kDouble: return std::numeric_limits<double>::min();
        case at::kHalf: return static_cast<double>(std::numeric_limits<c10::Half>::min());
        case at::kBFloat16: return static_cast<double>(std::numeric_limits<c10::BFloat16>::min());
        default: return std::numeric_limits<float>::min();
    }
}

inline int64_t round_to_dtype(int64_t value, at::ScalarType dtype)
{
    switch (dtype)
    {
        case at::kDouble: return static_cast<int64_t>(static_cast<double>(value));
        case at::kHalf: return static_cast<int64_t>(static_cast<float>(c10::Half(static_cast<float>(value))));
        case at::kBFloat16: return static_cast<int64_t>(static_cast<float>(c10::BFloat16(static_cast<float>(value))));
        default: return static_cast<int64_t>(static_cast<float>(value));
    }
}

inline void check_floating(const at::Tensor& self, const char* name)
{
    TORCH_CHECK_NOT_IMPLEMENTED(at::isFloatingType(self.scalar_type()), name, " is not implemented for '", self.scalar_type(), "'.");
}

inline std::vector<int64_t> compute_strides(
    c10::IntArrayRef sizes, 
    at::MemoryFormat format) 
{
    if (format == at::MemoryFormat::Contiguous) {
        auto strides = c10::contiguous_strides(sizes);
        return std::vector<int64_t>(strides.begin(), strides.end());
    }

    if (format == at::MemoryFormat::ChannelsLast && sizes.size() == 4) {
        auto strides = c10::get_channels_last_strides_2d(sizes);
        return std::vector<int64_t>(strides.begin(), strides.end());
    }

    if (format == at::MemoryFormat::ChannelsLast3d && sizes.size() == 5) {
        auto strides = c10::get_channels_last_strides_3d(sizes);
        return std::vector<int64_t>(strides.begin(), strides.end());
    }

    TORCH_CHECK(false, "torchvulkan [ERROR]: Unsupported memory format for sizes.");
}

inline at::Tensor promote_to_float(const at::Tensor& self)
{
    if (self.is_floating_point() || self.is_complex()) return self;
    return self.to(c10::typeMetaToScalarType(at::get_default_dtype()));
}

// copies an op result into an out= tensor, resizing it first when the shapes differ
inline at::Tensor& fill_out(at::Tensor& out, const at::Tensor& res)
{
    TORCH_CHECK(at::canCast(res.scalar_type(), out.scalar_type()),
        "result type ", res.scalar_type(), " can't be cast to the desired output type ", out.scalar_type()
    );
    if (out.sizes() != res.sizes()) out.resize_(res.sizes());
    out.copy_(res);
    return out;
}

// writes an in-place op's result back into self, with the checks torch makes: the result must keep
// self's shape (no broadcasting self up) and be castable to self's dtype (no float into an int tensor)
inline at::Tensor& copy_inplace_result(at::Tensor& self, const at::Tensor& result)
{
    TORCH_CHECK(at::canCast(result.scalar_type(), self.scalar_type()),
        "result type ", result.scalar_type(), " can't be cast to the desired output type ", self.scalar_type()
    );
    TORCH_CHECK(result.sizes() == self.sizes(),
        "output with shape ", self.sizes(), " doesn't match the broadcast shape ", result.sizes()
    );
    return self.copy_(result);
}

// integer reductions accumulate in int64 to avoid overflow
inline c10::ScalarType reduce_compute_dtype(const at::Tensor& self, c10::optional<at::ScalarType> dtype)
{
    if (dtype.has_value()) return *dtype;
    if (at::isIntegralType(self.scalar_type(), /*includeBool=*/true) && self.scalar_type() != at::kLong) {
        return at::kLong;
    }
    return self.scalar_type();
}

// the dtype a reduction accumulates in is not always the dtype it returns
inline c10::ScalarType reduce_accumulate_dtype(c10::ScalarType dtype)
{
    return (dtype == at::kHalf || dtype == at::kBFloat16) ? at::kFloat : dtype;
}

inline bool is_low_prec(at::ScalarType dtype)
{
    return dtype == at::kHalf || dtype == at::kBFloat16;
}

// the complex dtype a real one pairs into: complex128 for float64, complex64 otherwise
inline c10::ScalarType get_dtype_complex(c10::ScalarType real)
{
    return real == at::kDouble ? at::kComplexDouble : at::kComplexFloat;
}

// 0 = none, 1 = mean, 2 = sum, as torch's Reduction enum
inline at::Tensor reduce_loss(const at::Tensor& loss, int64_t reduction)
{
    if (reduction == 1) return at::mean(loss);
    if (reduction == 2) return at::sum(loss);
    return loss;
}

inline std::vector<int64_t> to_ints(c10::SymIntArrayRef sizes)
{
    std::vector<int64_t> values;
    for (const c10::SymInt& s : sizes) values.push_back(s.guard_int(__FILE__, __LINE__));
    return values;
}

// the dtype a random kernel samples in: float64 for float64 outputs, float32 otherwise (T.Real in philox.slang)
inline c10::ScalarType random_real_dtype(c10::ScalarType dtype)
{
    return dtype == at::kDouble ? at::kDouble : at::kFloat;
}

// the dtype a random kernel generates in when the device cannot write `dtype`
inline c10::ScalarType random_proxy_dtype(c10::ScalarType dtype)
{
    return at::isFloatingType(dtype) ? at::kFloat : at::kLong;
}

// a parameter packed for IRandomReal.fromBits: a float in the low 32 bits, or a whole double
inline uint64_t random_real_bits(double value, c10::ScalarType real)
{
    if (real == at::kDouble) {
        uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(double));
        return bits;
    }

    float narrowed = static_cast<float>(value);
    uint32_t bits = 0;
    std::memcpy(&bits, &narrowed, sizeof(float));
    return bits;
}
