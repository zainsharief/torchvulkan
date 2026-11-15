#include "generator.h"

#include <ATen/core/Tensor.h>
#include <ATen/EmptyTensor.h>
#include <cstring>

#include "vulkan_context.h"

std::mutex VulkanGeneratorImpl::pending_seeds_mutex_;
bool VulkanGeneratorImpl::default_generators_created_ = false;
std::optional<uint64_t> VulkanGeneratorImpl::pending_seed_all_;
std::unordered_map<c10::DeviceIndex, uint64_t> VulkanGeneratorImpl::pending_seeds_;

VulkanGeneratorImpl::VulkanGeneratorImpl(c10::DeviceIndex device_index)
    : VulkanGeneratorImpl(device_index, c10::make_intrusive<VulkanGeneratorState>())
{
}

VulkanGeneratorImpl::VulkanGeneratorImpl(c10::DeviceIndex device_index, c10::intrusive_ptr<VulkanGeneratorState> state)
    : c10::GeneratorImpl{
        c10::Device(c10::DeviceType::PrivateUse1, device_index),
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1)
    },
      state_(std::move(state))
{
}

void VulkanGeneratorImpl::set_current_seed(uint64_t seed)
{
    state_->seed = seed;
    state_->offset = 0;
}

uint64_t VulkanGeneratorImpl::seed()
{
    uint64_t random = c10::detail::getNonDeterministicRandom();
    set_current_seed(random);
    return random;
}

void VulkanGeneratorImpl::set_state(const c10::TensorImpl& new_state)
{
    at::detail::check_rng_state(new_state);
    TORCH_CHECK(new_state.numel() == STATE_BYTES, 
        "torchvulkan [ERROR]: Expected an RNG state of ", STATE_BYTES, " bytes, but got ", new_state.numel(), "."
    );

    const uint8_t* data = new_state.data_dtype_initialized<uint8_t>();
    std::memcpy(&state_->seed, data, sizeof(uint64_t));
    std::memcpy(&state_->offset, data + sizeof(uint64_t), sizeof(uint64_t));
}

c10::intrusive_ptr<c10::TensorImpl> VulkanGeneratorImpl::get_state() const
{
    at::Tensor state = at::detail::empty_cpu({STATE_BYTES}, at::ScalarType::Byte, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
    uint8_t* data = state.data_ptr<uint8_t>();
    std::memcpy(data, &state_->seed, sizeof(uint64_t));
    std::memcpy(data + sizeof(uint64_t), &state_->offset, sizeof(uint64_t));
    return state.getIntrusivePtr();
}

void VulkanGeneratorImpl::graphsafe_set_state(const c10::intrusive_ptr<c10::GeneratorImpl>& new_state)
{
    auto* other = dynamic_cast<VulkanGeneratorImpl*>(new_state.get());
    TORCH_CHECK(other != nullptr, "torchvulkan [ERROR]: graphsafe_set_state expects a vulkan generator.");
    state_ = other->state_;
}

c10::intrusive_ptr<c10::GeneratorImpl> VulkanGeneratorImpl::graphsafe_get_state() const
{
    return c10::make_intrusive<VulkanGeneratorImpl>(device().index(), state_);
}

VulkanGeneratorImpl* VulkanGeneratorImpl::clone_impl() const
{
    auto state = c10::make_intrusive<VulkanGeneratorState>();
    state->seed = state_->seed;
    state->offset = state_->offset;
    return new VulkanGeneratorImpl(device().index(), std::move(state));
}

std::pair<uint64_t, uint64_t> VulkanGeneratorImpl::advance()
{
    std::pair<uint64_t, uint64_t> inputs{state_->seed, state_->offset};
    state_->offset += 1;
    return inputs;
}

c10::DeviceIndex VulkanGeneratorImpl::resolve_device(c10::DeviceIndex device_index)
{
    return device_index == -1 ? VulkanContext::CurrentDevice() : device_index;
}

void VulkanGeneratorImpl::seed_generator(at::Generator generator, uint64_t seed)
{
    std::lock_guard<std::mutex> lock(generator.mutex());
    generator.set_current_seed(seed);
}

std::vector<at::Generator> VulkanGeneratorImpl::create_default_generators(size_t count)
{
    std::lock_guard<std::mutex> lock(pending_seeds_mutex_);
    std::vector<at::Generator> generators;
    generators.reserve(count);

    for (size_t i = 0; i < count; i++)
    {
        c10::DeviceIndex index = static_cast<c10::DeviceIndex>(i);
        at::Generator generator = at::make_generator<VulkanGeneratorImpl>(index);

        if (pending_seeds_.count(index)) generator.set_current_seed(pending_seeds_[index]);
        else if (pending_seed_all_.has_value()) generator.set_current_seed(*pending_seed_all_);

        generators.push_back(std::move(generator));
    }

    pending_seeds_.clear();
    pending_seed_all_.reset();
    default_generators_created_ = true;
    return generators;
}

const at::Generator& VulkanGeneratorImpl::default_generator(c10::DeviceIndex device_index)
{
    VulkanContext& context = VulkanContext::Instance();
    c10::DeviceIndex index = resolve_device(device_index);
    TORCH_CHECK(index >= 0 && index < static_cast<c10::DeviceIndex>(context.getDeviceCount()),
        "torchvulkan [ERROR]: Invalid device index ", index, " for a generator."
    );
    return context.devices[index]->generator;
}

at::Generator VulkanGeneratorImpl::create(c10::DeviceIndex device_index)
{
    return at::make_generator<VulkanGeneratorImpl>(resolve_device(device_index));
}

void VulkanGeneratorImpl::manual_seed(c10::DeviceIndex device_index, uint64_t seed)
{
    {
        std::lock_guard<std::mutex> lock(pending_seeds_mutex_);
        if (!default_generators_created_) {
            pending_seeds_[resolve_device(device_index)] = seed;
            return;
        }
    }
    seed_generator(default_generator(device_index), seed);
}

void VulkanGeneratorImpl::manual_seed_all(uint64_t seed)
{
    {
        std::lock_guard<std::mutex> lock(pending_seeds_mutex_);
        if (!default_generators_created_) {
            pending_seeds_.clear();
            pending_seed_all_ = seed;
            return;
        }
    }
    for (DeviceContext* device : VulkanContext::Instance().devices) seed_generator(device->generator, seed);
}

std::pair<uint64_t, uint64_t> VulkanGeneratorImpl::reserve(const std::optional<at::Generator>& generator, c10::DeviceIndex device_index)
{
    auto* impl = at::get_generator_or_default<VulkanGeneratorImpl>(generator, default_generator(device_index));
    std::lock_guard<std::mutex> lock(impl->mutex_);
    return impl->advance();
}
