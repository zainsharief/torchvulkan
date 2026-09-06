#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#include "memory.h" 

VkResult VulkanBuffer::createBuffer(size_t size, MemoryUsage usage)
{
    size_ = size;
    usage_ = usage;
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = (size + 127) & ~127; // pad to 128 bytes for safety
    bufferInfo.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |   // allow buffer device address support
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT |   // allow copy from this buffer
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT;    // allow copy to this buffer

    if (usage == MemoryUsage::DEVICE_ONLY) {
        bufferInfo.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // allow shader read/write this buffer
    }
    
    VmaAllocationCreateInfo allocInfo = {};

    switch (usage) 
    {
        case MemoryUsage::DEVICE_ONLY:
            allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            break;

        case MemoryUsage::HOST_TO_DEVICE:
            allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | 
                              VMA_ALLOCATION_CREATE_MAPPED_BIT;
            break;

        case MemoryUsage::DEVICE_TO_HOST:
            allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | 
                              VMA_ALLOCATION_CREATE_MAPPED_BIT;
            break;
    }

    VkResult result = vmaCreateBuffer(allocator_, &bufferInfo, &allocInfo, &buffer_, &allocation_, &allocInfo_);
    if (result != VK_SUCCESS) return result;

    if (usage != MemoryUsage::DEVICE_ONLY) mappedData_ = allocInfo_.pMappedData;

    // The device address is fixed for the buffer's lifetime, so query it once, here.
    VkBufferDeviceAddressInfo bdaInfo{};
    bdaInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    bdaInfo.buffer = buffer_;
    bufferAddress_ = vkGetBufferDeviceAddress(device, &bdaInfo);

    return result;
}

void VulkanBuffer::invalidate() { vmaInvalidateAllocation(allocator_, allocation_, 0, VK_WHOLE_SIZE); }
void VulkanBuffer::flush() { vmaFlushAllocation(allocator_, allocation_, 0, VK_WHOLE_SIZE); }
VulkanBuffer::~VulkanBuffer() { if (buffer_ != VK_NULL_HANDLE) vmaDestroyBuffer(allocator_, buffer_, allocation_); }