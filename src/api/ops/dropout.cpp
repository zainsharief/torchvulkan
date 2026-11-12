#include <torch/extension.h>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "api/ops/helpers.h"
#include "vulkan/generator.h"

namespace {

std::tuple<at::Tensor, at::Tensor> native_dropout_vulkan(const at::Tensor& input, double p, std::optional<bool> train)
{
    if (input.numel() == 0) {
        return std::make_tuple(input, at::empty_like(input, input.options()));
    }

    if (train.has_value() && !*train) {
        return std::make_tuple(input.clone(), at::ones_like(input, input.options().dtype(at::kBool)));
    }

    double keep_prob = 1.0 - p;
    double scale = keep_prob == 0 ? 0.0 : 1.0 / keep_prob;

    if (!at::isFloatingType(input.scalar_type()) || !is_dtype_supported(input, input.scalar_type()) || !is_dtype_supported(input, at::kBool) || !fits_32bit_indexing({input})) {
        at::Tensor mask = at::empty_like(input, input.options().dtype(at::kBool));
        mask.bernoulli_(keep_prob);
        return std::make_tuple(input.mul(mask).mul_(scale), mask);
    }

    DeviceContext* device = get_device_context(input);
    at::ScalarType real = random_real_dtype(input.scalar_type());

    at::Tensor src = input.contiguous();
    at::Tensor output = at::empty_like(src, at::MemoryFormat::Contiguous);
    at::Tensor mask = at::empty_like(src, src.options().dtype(at::kBool), at::MemoryFormat::Contiguous);
    uint64_t numel = src.numel();
    uint32_t vecSize = get_dtype_vec_size(src.scalar_type());
    uint32_t workgroupSizeX = get_dtype_workgroup_size(src.scalar_type(), vecSize, device);

    SpecializationBuilder spd{};
    spd.push(workgroupSizeX);
    SpecializationArgs specialization = spd.build();

    auto [seed, offset] = VulkanGeneratorImpl::reserve(std::nullopt, src.device().index());

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(src))
        .push(get_tensor_address(output))
        .push(get_tensor_address(mask))
        .push(numel)
        .push(seed)
        .push(offset)
        .push(random_real_bits(keep_prob, real))
        .push(random_real_bits(scale, real));
    PushConstants pushConstants = pcs.build();

    uint64_t threads = (numel + vecSize - 1) / vecSize;
    uint32_t groupX = static_cast<uint32_t>((threads + workgroupSizeX - 1) / workgroupSizeX);

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::DROPOUT, src.scalar_type()},
        device->default_subgroup_size,
        specialization,
        pushConstants,
        /* read = */ {src},
        /* write = */ {output, mask},
        groupX, 1, 1
    );

    return std::make_tuple(output, mask);
}

at::Tensor native_dropout_backward_vulkan(const at::Tensor& grad_output, const at::Tensor& mask, double scale)
{
    return grad_output * mask * scale;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("native_dropout", &native_dropout_vulkan);
    m.impl("native_dropout_backward", &native_dropout_backward_vulkan);
}
