#pragma once
#include <ATen/core/Generator.h>
#include <c10/core/GeneratorImpl.h>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

struct VulkanGeneratorState : public c10::intrusive_ptr_target {
    uint64_t seed = c10::default_rng_seed_val;
    uint64_t offset = 0;
};

class VulkanGeneratorImpl : public c10::GeneratorImpl {
public:
    VulkanGeneratorImpl(c10::DeviceIndex device_index);
    VulkanGeneratorImpl(c10::DeviceIndex device_index, c10::intrusive_ptr<VulkanGeneratorState> state);
    ~VulkanGeneratorImpl() override = default;

    void set_current_seed(uint64_t seed) override;
    uint64_t seed() override;
    uint64_t current_seed() const override { return state_->seed; };

    void set_offset(uint64_t offset) override { state_->offset = offset; };
    uint64_t get_offset() const override { return state_->offset; };

    void set_state(const c10::TensorImpl& new_state) override;
    c10::intrusive_ptr<c10::TensorImpl> get_state() const override;

    void graphsafe_set_state(const c10::intrusive_ptr<c10::GeneratorImpl>& new_state) override;
    c10::intrusive_ptr<c10::GeneratorImpl> graphsafe_get_state() const override;

    static std::vector<at::Generator> create_default_generators(size_t count);
    static const at::Generator& default_generator(c10::DeviceIndex device_index = -1);
    static at::Generator create(c10::DeviceIndex device_index = -1);
    static std::pair<uint64_t, uint64_t> reserve(const std::optional<at::Generator>& generator, c10::DeviceIndex device_index);

    static void manual_seed(c10::DeviceIndex device_index, uint64_t seed);
    static void manual_seed_all(uint64_t seed);

    static c10::DeviceType device_type() { return c10::DeviceType::PrivateUse1; }

private:
    VulkanGeneratorImpl* clone_impl() const override;
    std::pair<uint64_t, uint64_t> advance();

    static c10::DeviceIndex resolve_device(c10::DeviceIndex device_index);
    static void seed_generator(at::Generator generator, uint64_t seed);

    static constexpr int64_t STATE_BYTES = 2 * sizeof(uint64_t);

    c10::intrusive_ptr<VulkanGeneratorState> state_;

    static std::mutex pending_seeds_mutex_;
    static bool default_generators_created_;
    static std::optional<uint64_t> pending_seed_all_;
    static std::unordered_map<c10::DeviceIndex, uint64_t> pending_seeds_;
};
