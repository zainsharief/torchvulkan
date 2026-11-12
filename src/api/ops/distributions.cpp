#include <torch/extension.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "api/ops/helpers.h"
#include "vulkan/generator.h"

namespace {

enum class DistributionOp {
    GAMMA = 0,
    POISSON = 1,
    BINOMIAL = 2
};

at::Tensor dispatch_distribution_shader(
    DistributionOp op,
    const at::Tensor& input0,
    const at::Tensor& input1,
    const std::optional<at::Generator>& generator)
{
    at::ScalarType dtype = input0.scalar_type();

    if (!is_dtype_supported(input0, dtype)) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Vulkan device does not support ", dtype, ". Sampling in Float instead.");
        dtype = at::kFloat;
    }
    // GAMMA's smallest result, positive in both the requested dtype and the one sampled in
    double tiny = std::max(get_dtype_smallest_normal(input0.scalar_type()), get_dtype_smallest_normal(dtype));

    at::Tensor in0 = input0.to(dtype).contiguous();
    at::Tensor in1 = input1.defined() ? input1.to(dtype).expand(input0.sizes()).contiguous() : at::Tensor();
    at::Tensor out = at::empty(in0.sizes(), in0.options());
    uint64_t numel = out.numel();
    if (numel == 0) return out;

    DeviceContext* device = get_device_context(input0);
    uint32_t workgroupSizeX = get_dtype_workgroup_size(dtype, 1, device);

    SpecializationBuilder spd{};
    spd.push(static_cast<uint32_t>(op))
        .push(workgroupSizeX);
    SpecializationArgs specialization = spd.build();

    auto [seed, offset] = VulkanGeneratorImpl::reserve(generator, input0.device().index());

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(out))
        .push(get_tensor_address(in0))
        .push(in1.defined() ? get_tensor_address(in1) : uint64_t(0))
        .push(numel)
        .push(seed)
        .push(offset)
        .push(random_real_bits(tiny, random_real_dtype(dtype)));
    PushConstants pushConstants = pcs.build();

    uint32_t groupX = static_cast<uint32_t>((numel + workgroupSizeX - 1) / workgroupSizeX);

    std::vector<at::Tensor> reads{in0};
    if (in1.defined()) reads.push_back(in1);

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::DISTRIBUTION, dtype},
        device->default_subgroup_size,
        specialization,
        pushConstants,
        /* read = */ reads,
        /* write = */ {out},
        groupX, 1, 1
    );

    return out;
}

at::Tensor standard_gamma_vulkan(const at::Tensor& alpha, std::optional<at::Generator> generator)
{
    check_floating(alpha, "_standard_gamma");
    return dispatch_distribution_shader(DistributionOp::GAMMA, alpha, at::Tensor(), generator).to(alpha.scalar_type());
}

at::Tensor poisson_vulkan(const at::Tensor& rates, std::optional<at::Generator> generator)
{
    check_floating(rates, "poisson");
    TORCH_CHECK(rates.numel() == 0 || (at::amin(rates).item<double>() >= 0 && !std::isnan(rates.sum().item<double>())),
        "invalid Poisson rate, expected rate to be non-negative"
    );
    return dispatch_distribution_shader(DistributionOp::POISSON, rates, at::Tensor(), generator).to(rates.scalar_type());
}

at::Tensor binomial_vulkan(const at::Tensor& count, const at::Tensor& prob, std::optional<at::Generator> generator)
{
    TORCH_CHECK_VALUE(at::isFloatingType(count.scalar_type()), "binomial only supports floating-point dtypes for count, got: ", count.scalar_type());
    TORCH_CHECK_VALUE(at::isFloatingType(prob.scalar_type()), "binomial only supports floating-point dtypes for prob, got: ", prob.scalar_type());
    return dispatch_distribution_shader(DistributionOp::BINOMIAL, count, prob, generator).to(count.scalar_type());
}

at::Tensor sample_dirichlet_vulkan(const at::Tensor& alpha, std::optional<at::Generator> generator)
{
    check_floating(alpha, "_sample_dirichlet");
    at::ScalarType real = random_real_dtype(alpha.scalar_type());
    at::Tensor gamma = dispatch_distribution_shader(DistributionOp::GAMMA, alpha.to(real), at::Tensor(), generator);

    at::Tensor result = gamma / gamma.sum(-1, /*keepdim=*/true);
    // torch keeps each value in [smallest normal, the largest value below 1]
    double below_one = alpha.scalar_type() == at::kDouble ? std::nextafter(1.0, 0.0) : static_cast<double>(std::nextafter(1.0f, 0.0f));
    return result.clamp(get_dtype_smallest_normal(alpha.scalar_type()), below_one).to(alpha.scalar_type());
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("_standard_gamma", &standard_gamma_vulkan);
    m.impl("poisson", &poisson_vulkan);
    m.impl("binomial", &binomial_vulkan);
    m.impl("_sample_dirichlet", &sample_dirichlet_vulkan);
}
