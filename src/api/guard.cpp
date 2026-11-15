#include <torch/extension.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include <atomic>
#include <chrono>
#include <unordered_map>

#include "vulkan/vulkan_context.h"
#include "vulkan/shader.h"

namespace c10 {
namespace impl {

struct VulkanEvent {
    bool recorded = false;
    std::chrono::steady_clock::time_point time;
};

class VulkanGuardImpl : public DeviceGuardImplInterface {
public:
    VulkanGuardImpl() {}
    DeviceType type() const override { return DeviceType::PrivateUse1; }

    Stream getStream(Device d) const override { d = indexed(d); return Stream(Stream::UNSAFE, d, currentStreams()[d.index()]); }
    Stream getDefaultStream(Device d) const override { return Stream(Stream::UNSAFE, indexed(d), 0); }
    Stream getStreamFromGlobalPool(Device d, bool /* isHighPriority */) const override { return getNewStream(d, 0); }
    Stream getNewStream(Device d, int /* priority */) const override { return Stream(Stream::UNSAFE, indexed(d), ++nextStreamId()); }
    Stream exchangeStream(Stream s) const override {
        Device d = indexed(s.device());
        StreamId& current = currentStreams()[d.index()];
        Stream previous(Stream::UNSAFE, d, current);
        current = s.id();
        return previous;
    }
    bool queryStream(const Stream& s) const override {
        synchronizeStream(s);
        return true;
    }
    void synchronizeStream(const Stream& s) const override { synchronizeDevice(s.device_index()); }

    void destroyEvent(void* event, const DeviceIndex /* device_index */) const noexcept override {
        delete static_cast<VulkanEvent*>(event);
    }
    void record(void** event, const Stream& stream, const DeviceIndex device_index, const EventFlag /* flag */) const override {
        TORCH_CHECK(device_index == -1 || device_index == stream.device_index(), "Event device index ", device_index,
                    " does not match recording stream's device index ", stream.device_index(), ".");
        if (*event == nullptr) *event = new VulkanEvent();
        synchronizeStream(stream);
        auto* e = static_cast<VulkanEvent*>(*event);
        e->recorded = true;
        e->time = std::chrono::steady_clock::now();
    }
    void block(void* /* event */, const Stream& /* stream */) const override {}
    bool queryEvent(void* /* event */) const override { return true; }
    void synchronizeEvent(void* /* event */) const override {}
    double elapsedTime(void* event1, void* event2, const DeviceIndex /* device_index */) const override {
        auto* start = static_cast<VulkanEvent*>(event1);
        auto* end = static_cast<VulkanEvent*>(event2);
        TORCH_CHECK(start != nullptr && end != nullptr && start->recorded && end->recorded,
                    "Both events must be recorded before calculating elapsed time.");
        return std::chrono::duration<double, std::milli>(end->time - start->time).count();
    }

    Device exchangeDevice(Device d) const override {  
        Device oldDevice = getDevice();
        setDevice(d);
        return oldDevice;
    }

    void setDevice(c10::Device d) const override { 
        VulkanContext::SetCurrentDevice(VulkanContext::Instance().GetDeviceContext(d.index())->index);
    }
    void uncheckedSetDevice(Device d) const noexcept override { VulkanContext::SetCurrentDevice(d.index()); }
    
    c10::Device getDevice() const override { return c10::Device(c10::DeviceType::PrivateUse1, VulkanContext::CurrentDevice()); }
    DeviceIndex deviceCount() const noexcept override {
        try {
            return static_cast<DeviceIndex>(VulkanContext::Instance().getDeviceCount());
        } catch (...) {
            return 0;
        }
    }

    void synchronizeDevice(const DeviceIndex device_index) const override {
        VulkanContext::Instance().GetDeviceContext(device_index)->flush();
    }

private:
    Device indexed(Device d) const { return d.has_index() ? d : getDevice(); }
    static std::unordered_map<DeviceIndex, StreamId>& currentStreams() {
        thread_local std::unordered_map<DeviceIndex, StreamId> streams;
        return streams;
    }
    static std::atomic<StreamId>& nextStreamId() {
        static std::atomic<StreamId> id{0};
        return id;
    }
};

C10_REGISTER_GUARD_IMPL(PrivateUse1, VulkanGuardImpl);

}
}