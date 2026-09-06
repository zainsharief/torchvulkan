#pragma once
#include <ATen/core/Scalar.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <array>
#include <cstring>

#define MAX_PUSH_CONSTANT_BYTES 128
#define MAX_SPEC_CONSTANTS 32
#define MAX_SPEC_DATA_BYTES 256
#define MAX_DIMS 64

class SpecializationBuilder {
public:
    // pushes any value and automatically tracks its size and byte offset
    template <typename T>
    SpecializationBuilder& push(const T& value) {
        TORCH_CHECK(numConstants_ < MAX_SPEC_CONSTANTS, "torchvulkan [ERROR]: Too many specialization constants!");
        TORCH_CHECK(data_size_ + sizeof(T) <= MAX_SPEC_DATA_BYTES, "torchvulkan [ERROR]: Specialization constant data exceeded buffer!");

        offsets_array[numConstants_] = data_size_;
        sizes_array[numConstants_] = sizeof(T);
        numConstants_++;

        std::memcpy(data_buffer.data() + data_size_, &value, sizeof(T));
        data_size_ += sizeof(T);

        return *this;
    }

    const void* data() const { return data_buffer.data(); }
    const size_t* offsets() const { return offsets_array.data(); }
    const size_t* sizes() const { return sizes_array.data(); }
    uint32_t numConstants() const { return numConstants_; }

private:
    std::array<uint8_t, MAX_SPEC_DATA_BYTES> data_buffer{};
    std::array<size_t, MAX_SPEC_CONSTANTS> offsets_array{};
    std::array<size_t, MAX_SPEC_CONSTANTS> sizes_array{};
    uint32_t numConstants_ = 0;
    size_t data_size_ = 0;
};

struct PushConstants {
    void* data;
    size_t size;
};

class PushConstantBuilder {
public:
    // push a single value
    template <typename T>
    PushConstantBuilder& push(const T& value) 
    {
        size_t size = sizeof(T);
        TORCH_CHECK(current_size + size <= MAX_PUSH_CONSTANT_BYTES, "torchvulkan [ERROR]: Push constants exceeded ", MAX_PUSH_CONSTANT_BYTES, " bytes!");
        
        std::memcpy(buffer.data() + current_size, &value, size);
        current_size += size;
        return *this;
    }

    // push an array
    template <typename T, size_t N>
    PushConstantBuilder& push_array(const T (&arr)[N]) 
    {
        size_t size = sizeof(T) * N;
        TORCH_CHECK(current_size + size <= MAX_PUSH_CONSTANT_BYTES, "torchvulkan [ERROR]: Push constants exceeded ", MAX_PUSH_CONSTANT_BYTES, " bytes!");
        
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

    // vkCmdPushConstants requires a multiple of 4 bytes; the padding bytes are zero
    size_t size() const { return (current_size + 3) & ~size_t(3); }

private:
    std::array<uint8_t, MAX_PUSH_CONSTANT_BYTES> buffer{}; // vulkan guarantees at least 128 bytes of push constant space on all devices
    size_t current_size = 0;
};

struct IntDivider 
{
    uint32_t divisor[MAX_DIMS];
    uint32_t multiplier[MAX_DIMS];
    uint32_t shift_val[MAX_DIMS];

    IntDivider() 
    {
        for (int i = 0; i < MAX_DIMS; ++i) {
            divisor[i] = 1;
            multiplier[i] = 1;
            shift_val[i] = 0;
        }
    }

    void set(int idx, uint32_t d) {
        divisor[idx] = d;
        
        uint32_t shift = 0;
        for (shift = 0; shift < 32; shift++) {
            if ((1U << shift) >= d) break;
        }
        shift_val[idx] = shift;
        
        uint64_t one = 1;
        uint64_t magic = ((one << 32) * ((one << shift) - d)) / d + 1;
        multiplier[idx] = static_cast<uint32_t>(magic);
    }
};
