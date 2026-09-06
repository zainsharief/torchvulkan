#pragma once
#include "vulkan/vulkan_context.h"
#include "vulkan/builders.h"
#include <c10/core/Device.h>
#include <c10/core/MemoryFormat.h>
#include <c10/util/strides.h>
#include <vector>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>

inline uint64_t get_tensor_address(const at::Tensor& t) 
{
    VulkanBuffer* buffer = static_cast<VulkanBuffer*>(t.storage().data_ptr().get_context());
    return buffer->bufferAddress() + t.storage_offset() * t.element_size();
}

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