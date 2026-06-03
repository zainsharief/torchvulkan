#pragma once
#include "vulkan/vulkan_context.h"
#include <c10/core/Device.h>
#include <c10/core/MemoryFormat.h>
#include <c10/util/strides.h>
#include <vector>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#define MAX_DIMS 4
#define MAX_VEC_SIZE 4
#define MAX_WORKGROUP_BYTES 1024

class SpecializationBuilder {
public:
    // pushes any value and automatically tracks its size and byte offset
    template <typename T>
    SpecializationBuilder& push(const T& value) {
        offsets_array.push_back(data_buffer.size());
        sizes_array.push_back(sizeof(T));

        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&value);
        data_buffer.insert(data_buffer.end(), ptr, ptr + sizeof(T));
        
        return *this;
    }

    const void* data() const { return data_buffer.data(); }
    const size_t* offsets() const { return offsets_array.data(); }
    const size_t* sizes() const { return sizes_array.data(); }
    uint32_t numConstants() const { return static_cast<uint32_t>(sizes_array.size()); }
    
private:
    std::vector<uint8_t> data_buffer;
    std::vector<size_t> offsets_array;
    std::vector<size_t> sizes_array;
};

class PushConstantBuilder {
public:
    // push a single value
    template <typename T>
    PushConstantBuilder& push(const T& value) 
    {
        size_t size = sizeof(T);
        TORCH_CHECK(current_size + size <= 128, "torchvulkan [ERROR]: Push constants exceeded 128 bytes!");
        
        std::memcpy(buffer.data() + current_size, &value, size);
        current_size += size;
        return *this;
    }

    // push an array
    template <typename T, size_t N>
    PushConstantBuilder& push_array(const T (&arr)[N]) 
    {
        size_t size = sizeof(T) * N;
        TORCH_CHECK(current_size + size <= 128, "torchvulkan [ERROR]: Push constants exceeded 128 bytes!");
        
        std::memcpy(buffer.data() + current_size, arr, size);
        current_size += size;
        return *this;
    }

    PushConstantBuilder& push_scalar(const at::Scalar& value, at::ScalarType dtype)
    {
        switch (dtype) {
            case at::kByte: push(value.to<uint8_t>()); break;
            case at::kChar: push(value.to<int8_t>()); break;
            case at::kShort: push(value.to<int16_t>()); break;
            case at::kInt: push(value.to<int32_t>()); break;
            case at::kLong: push(value.to<int64_t>()); break;
            case at::kBool: push(value.to<bool>()); break;
            case at::kHalf: push(value.to<c10::Half>()); break;
            case at::ScalarType::UInt16: push(value.to<uint16_t>()); break;
            case at::ScalarType::UInt32: push(value.to<uint32_t>()); break;
            case at::ScalarType::UInt64: push(value.to<uint64_t>()); break;
            case at::kDouble: push(value.to<double>()); break;
            case at::kFloat: push(value.to<float>()); break;
            default: TORCH_CHECK(false, "torchvulkan [ERROR]: Unsupported dtype for scalar value."); break;
        }

        return *this;
    }

    const void* data() const { return buffer.data(); }
    size_t size() const { return current_size; }

private:
    std::array<uint8_t, 128> buffer{}; // vulkan guarantees at least 128 bytes of push constant space on all devices
    size_t current_size = 0;
};

struct IntDivider {
    uint32_t divisor;
    uint32_t multiplier;
    uint32_t shift_val;
    uint32_t pad;

    IntDivider() : divisor(1), multiplier(1), shift_val(0), pad(0) {}

    IntDivider(uint32_t d) 
        : divisor(d) 
    {        
        for (shift_val = 0; shift_val < 32; shift_val++) {
            if ((1U << shift_val) >= d) break;
        }
        
        uint64_t one = 1;
        uint64_t magic = ((one << 32) * ((one << shift_val) - d)) / d + 1;
        multiplier = static_cast<uint32_t>(magic);
    }
};

inline bool is_dtype_supported(at::ScalarType dtype) 
{
    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();

    switch (dtype) 
    {
        case at::kDouble: return device->support_float64;
        case at::kLong: return device->support_int64;
        case at::kUInt64: return device->support_int64;
        
        case at::kFloat: return device->support_float32;
        case at::kInt: return device->support_int32;
        case at::kUInt32: return device->support_int32;
        
        case at::kHalf: return device->support_float16;
        case at::kBFloat16: return false; // we cannot support it yet
        case at::kShort: return device->support_int16;
        case at::kUInt16: return device->support_int16;
        
        case at::kChar: return device->support_int8; 
        case at::kByte: return device->support_int8; 
        case at::kBool: return device->support_int8; 
        
        default: return false;
    }
}

inline int get_dtype_vec_size(at::ScalarType dtype) 
{
    size_t bytes = c10::elementSize(dtype);
    if (bytes == 0) return 1;
    int size = (int)(16 / bytes);
    size = size <= 4 ? size : 4; // max vector size is 4
    return size; 
}

inline int get_dtype_workgroup_size(at::ScalarType dtype, uint32_t vecSize) 
{
    size_t bytes = c10::elementSize(dtype);
    if (bytes == 0 || vecSize == 0) return 1;
    int size = (int)(1024 / (bytes * vecSize)); // max out memory bandwidth
    return size;
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