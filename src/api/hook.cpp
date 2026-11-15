#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/Device.h>
#include <c10/core/impl/alloc_cpu.h>
#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>

#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h" 
#include "vulkan/generator.h"
#include "api/ops/internal.h"

class VulkanPinnedAllocator : public c10::Allocator {
public:
    c10::DataPtr allocate(size_t nbytes) override {
        void* data = nbytes == 0 ? nullptr : c10::alloc_cpu(nbytes);
        if (data != nullptr) {
            std::lock_guard<std::mutex> lock(mutex());
            blocks()[reinterpret_cast<uintptr_t>(data)] = nbytes;
        }
        return {data, data, &VulkanPinnedAllocator::deleter, c10::Device(c10::DeviceType::CPU)};
    }

    c10::DeleterFnPtr raw_deleter() const override { return &VulkanPinnedAllocator::deleter; }

    void copy_data(void* dest, const void* src, std::size_t count) const override { std::memcpy(dest, src, count); }

    static bool contains(const void* ptr) {
        std::lock_guard<std::mutex> lock(mutex());
        auto address = reinterpret_cast<uintptr_t>(ptr);
        auto it = blocks().upper_bound(address);
        if (it == blocks().begin()) return false;
        --it;
        return address < it->first + std::max<size_t>(it->second, 1);
    }

private:
    static void deleter(void* data) {
        if (data == nullptr) return;
        {
            std::lock_guard<std::mutex> lock(mutex());
            blocks().erase(reinterpret_cast<uintptr_t>(data));
        }
        c10::free_cpu(data);
    }
    static std::mutex& mutex() { static std::mutex m; return m; }
    static std::map<uintptr_t, size_t>& blocks() { static auto* b = new std::map<uintptr_t, size_t>(); return *b; }
};

static VulkanPinnedAllocator pinned_allocator;

struct VulkanHooksInterface : public at::PrivateUse1HooksInterface 
{
    bool isBuilt() const override { return true; }
    bool isAvailable() const override { return VulkanContext::Instance().getDeviceCount() > 0; }
    bool hasPrimaryContext(c10::DeviceIndex device_index) const override { return device_index < VulkanContext::Instance().getDeviceCount(); }
    at::Device getDeviceFromPtr(void* data) const override {
        if (data == nullptr) return c10::Device(c10::DeviceType::PrivateUse1, VulkanContext::CurrentDevice());
        return c10::Device(c10::DeviceType::PrivateUse1, static_cast<VulkanBuffer*>(data)->owner()->index);
    }

    c10::Allocator* getPinnedMemoryAllocator() const override { return &pinned_allocator; }
    bool isPinnedPtr(const void* data) const override { return VulkanPinnedAllocator::contains(data); }

    void resizePrivateUse1Bytes(const c10::Storage& storage, size_t newsize) const override {
        torchvulkan::resize_bytes_vulkan(storage.unsafeGetStorageImpl(), newsize);
    }

    const at::Generator& getDefaultGenerator(c10::DeviceIndex device_index) const override {
        return VulkanGeneratorImpl::default_generator(device_index);
    }

    at::Generator getNewGenerator(c10::DeviceIndex device_index = -1) const override {
        return VulkanGeneratorImpl::create(device_index);
    }
};

static VulkanHooksInterface vulkan_hooks;
static bool is_hooks_registered = []() {
    at::RegisterPrivateUse1HooksInterface(&vulkan_hooks);
    return true;
}();