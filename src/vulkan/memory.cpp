#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#include "memory.h"
#include "device_context.h"

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

    VkResult result = vmaCreateBuffer(owner_->allocator, &bufferInfo, &allocInfo, &buffer_, &allocation_, &allocInfo_);
    if (result != VK_SUCCESS) return result;

    // unified memory reports every heap as device-local, so only a discrete GPU's spill into host memory clears this
    VkMemoryPropertyFlags memoryFlags = 0;
    vmaGetAllocationMemoryProperties(owner_->allocator, allocation_, &memoryFlags);
    deviceLocal_ = (memoryFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;

    if (usage != MemoryUsage::DEVICE_ONLY) mappedData_ = allocInfo_.pMappedData;

    // The device address is fixed for the buffer's lifetime, so query it once, here.
    VkBufferDeviceAddressInfo bdaInfo{};
    bdaInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    bdaInfo.buffer = buffer_;
    bufferAddress_ = owner_->device_table.vkGetBufferDeviceAddress(owner_->device, &bdaInfo);

    owner_->trackBuffer(this);
    return result;
}

VkResult VulkanBuffer::importHostPointer(void* base, size_t size)
{
    size_ = size;
    usage_ = MemoryUsage::DEVICE_TO_HOST;
    deviceLocal_ = false;
    const VolkDeviceTable& vk = owner_->device_table;
    const VkExternalMemoryHandleTypeFlagBits handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;

    VkMemoryHostPointerPropertiesEXT hostProps{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    VkResult result = vk.vkGetMemoryHostPointerPropertiesEXT(owner_->device, handleType, base, &hostProps);
    if (result != VK_SUCCESS) return result;

    VkExternalMemoryBufferCreateInfo externalInfo{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    externalInfo.handleTypes = handleType;
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.pNext = &externalInfo;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    result = vk.vkCreateBuffer(owner_->device, &bufferInfo, nullptr, &buffer_);
    if (result != VK_SUCCESS) return result;

    VkMemoryRequirements requirements{};
    vk.vkGetBufferMemoryRequirements(owner_->device, buffer_, &requirements);
    // a host-coherent type, so the CPU sees the GPU's writes through its own mapping without an invalidate
    const VkPhysicalDeviceMemoryProperties* props = nullptr;
    vmaGetMemoryProperties(owner_->allocator, &props);
    uint32_t typeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < props->memoryTypeCount; i++) {
        bool allowed = (requirements.memoryTypeBits & hostProps.memoryTypeBits & (1u << i)) != 0;
        if (allowed && (props->memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { typeIndex = i; break; }
    }
    if (typeIndex == UINT32_MAX || requirements.size > size) {
        vk.vkDestroyBuffer(owner_->device, buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    VkImportMemoryHostPointerInfoEXT importInfo{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
    importInfo.handleType = handleType;
    importInfo.pHostPointer = base;
    VkMemoryAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocateInfo.pNext = &importInfo;
    allocateInfo.allocationSize = size;
    allocateInfo.memoryTypeIndex = typeIndex;
    result = vk.vkAllocateMemory(owner_->device, &allocateInfo, nullptr, &importedMemory_);
    if (result == VK_SUCCESS) result = vk.vkBindBufferMemory(owner_->device, buffer_, importedMemory_, 0);
    if (result != VK_SUCCESS) {
        vk.vkDestroyBuffer(owner_->device, buffer_, nullptr);
        if (importedMemory_ != VK_NULL_HANDLE) vk.vkFreeMemory(owner_->device, importedMemory_, nullptr);
        buffer_ = VK_NULL_HANDLE;
        importedMemory_ = VK_NULL_HANDLE;
        return result;
    }
    owner_->trackBuffer(this);
    return VK_SUCCESS;
}

void VulkanBuffer::release()
{
    if (buffer_ == VK_NULL_HANDLE) return;
    if (importedMemory_ != VK_NULL_HANDLE) {
        owner_->device_table.vkDestroyBuffer(owner_->device, buffer_, nullptr);
        owner_->device_table.vkFreeMemory(owner_->device, importedMemory_, nullptr);
        buffer_ = VK_NULL_HANDLE;
        importedMemory_ = VK_NULL_HANDLE;
        return;
    }
    vmaDestroyBuffer(owner_->allocator, buffer_, allocation_);
    buffer_ = VK_NULL_HANDLE;
    allocation_ = VK_NULL_HANDLE;
    mappedData_ = nullptr;
}

VulkanBuffer::~VulkanBuffer()
{
    if (buffer_ == VK_NULL_HANDLE) return;
    owner_->untrackBuffer(this);
    release();
}

void VulkanBuffer::invalidate() { vmaInvalidateAllocation(owner_->allocator, allocation_, 0, VK_WHOLE_SIZE); }
void VulkanBuffer::flush() { vmaFlushAllocation(owner_->allocator, allocation_, 0, VK_WHOLE_SIZE); }
