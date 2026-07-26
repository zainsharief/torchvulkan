#pragma once
#include <torch/extension.h>
#include <c10/core/Allocator.h>
#include <vector>
#include <mutex>

#include "vulkan_context.h"
#include "memory.h"

class VulkanAllocator : public c10::Allocator {
public:
    /**
     * @brief Allocates Vulkan device memory using the OOM (Out-Of-Memory) allocator.
     * * @note To satisfy Vulkan alignment requirements, the underlying allocation is 
     * padded to 128 bytes, though the tracked and requested logical size remains `nbytes`.
     * * @warning **ATen POINTER HACK:** PyTorch expects `c10::DataPtr` to hold a raw byte 
     * pointer. Because Vulkan memory is opaque to the CPU, we store the `VulkanBuffer*` 
     * instance in BOTH the `data` and `ctx` fields. When extracting the buffer elsewhere, 
     * you MUST use `(VulkanBuffer*)tensor.data_ptr().get_context()`.
     * * @param nbytes The logical number of bytes requested by the PyTorch tensor.
     * @returns A `c10::DataPtr` managing the VulkanBuffer. Ownership is transferred to 
     * PyTorch, which will automatically invoke `VulkanAllocator::deleter` upon destruction.
     */
    c10::DataPtr allocate(size_t nbytes) override;
    c10::DataPtr allocate(size_t nbytes) const {
        return const_cast<VulkanAllocator*>(this)->allocate(nbytes);
    }

    /**
     * @brief "Deletes" a buffer by returning it back to the cache.
     * * @param ptr Opaque pointer to the buffer (resolves to a VulkanBuffer*).
     */
    static void deleter(void* ptr);

    /**
     * @brief Performs an asynchronous Device-to-Device memory copy.
     * * @note This override exists strictly to satisfy the PyTorch `c10::Allocator` 
     * interface. For internal backend usage, we highly recommend using 
     * `VulkanAllocator::copy_device_to_device` directly, as it exposes necessary 
     * offset controls.
     * * @param dest Opaque pointer to the destination (resolves to a VulkanBuffer*).
     * @param src Opaque pointer to the source (resolves to a VulkanBuffer*).
     * @param count The number of bytes to copy.
     */
    void copy_data(void* dest, const void* src, std::size_t count) const override;

    /**
     * @brief Performs an asynchronous Host-to-Device memory copy.
     * * @param dest Opaque pointer to the destination (resolves to a VulkanBuffer*).
     * @param src Opaque pointer to the source CPU memory.
     * @param count The number of bytes to copy.
     */
    void copy_host_to_device(void* dest, uint64_t dest_offset, const void* src, std::size_t count) const;

    /**
     * @brief Performs a synchronous Device-to-Host memory copy.
     * * @warning Right now, this completely flushes the pipeline and stalls the CPU 
     * until all operations are complete. Future updates should improve this.
     * * @param dest Opaque pointer to the destination CPU memory.
     * @param src Opaque pointer to the source (resolves to a VulkanBuffer*).
     * @param count The number of bytes to copy.
     */
    void copy_device_to_host(void* dest, const void* src, uint64_t src_offset, std::size_t count) const;

    /**
     * @brief Performs a synchronous Device-to-Host memory copy from a pre-allocated Vulkan staging buffer.
     * * @note This is a highly specialized override designed specifically for non-contiguous 
     * tensor transfers. It prevents the overhead of a double-copy by allowing the compute 
     * shader to write directly into the staging buffer before this function is called.
     * * @warning **OWNERSHIP TRANSFER:** This function takes ownership of `stagingBuffer`. 
     * It will automatically invalidate, read, and then return the buffer to the cache 
     * for deletion. Do NOT access `stagingBuffer` after passing it to this function.
     * @warning Right now, this completely flushes the pipeline and stalls the CPU 
     * until all operations are complete. Future updates should improve this.
     * * @param dest Opaque pointer to the destination CPU memory.
     * @param stagingBuffer Opaque pointer to the source (resolves to a VulkanBuffer*).
     * @param count The number of bytes to copy.
     */
    void copy_device_to_host(void* dest, VulkanBuffer* stagingBuffer, std::size_t count) const;

    /**
     * @brief Performs an asynchronous Device-to-Device memory copy.
     * * @param dest Opaque pointer to the destination (resolves to a VulkanBuffer*).
     * @param src Opaque pointer to the source (resolves to a VulkanBuffer*).
     * @param count The number of bytes to copy.
     */
    void copy_device_to_device(void* dest, uint64_t dest_offset, const void* src, uint64_t src_offset, std::size_t count) const;

    /**
     * @brief Drains the pending-deletion queue of staging buffers (from
     * copy_host_to_device calls) back into the reusable pool.
     * * @warning The caller MUST have already flushed the device (a full submit +
     * wait) before calling this - staging buffers are queued here as soon as their
     * copy command is recorded, not once it has actually executed, so reclaiming
     * them without a preceding flush() can race a still-in-flight GPU read.
     */
    void clearResources() const;

private:
    mutable std::vector<VulkanBuffer*> deleteQueue;
    mutable std::mutex mutex_; // thread safety

    static const uint32_t SYNC_THRESHOLD = 200;
    static constexpr const float VRAM_USAGE_LIMIT = 0.95; // leave headroom for driver allocations
    mutable std::atomic<size_t> allocated_bytes{0};
    mutable std::atomic<uint32_t> num_allocations_since_sync{SYNC_THRESHOLD};
    mutable std::atomic<size_t> vram_limit{0};

    VulkanBuffer* out_of_memory_buffer(size_t size, MemoryUsage usage) const;
    void sync_memory_budget(size_t size) const;
};

extern VulkanAllocator globalVulkanAllocator;