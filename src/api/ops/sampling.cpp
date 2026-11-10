#include <torch/extension.h>
#include <cmath>
#include <limits>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "api/ops/helpers.h"
#include "vulkan/generator.h"

namespace {

void dispatch_multinomial_shader(
    const at::Tensor& self,
    int64_t n_sample,
    const std::optional<at::Generator>& generator,
    at::Tensor& result)
{
    at::ScalarType real = random_real_dtype(self.scalar_type());
    if (!is_dtype_supported(self, real)) real = at::kFloat;

    int64_t n_categories = self.size(-1);
    at::Tensor cdf = self.reshape({-1, n_categories}).to(real).cumsum(-1).contiguous();
    at::Tensor out = result.is_contiguous() ? result : at::empty(result.sizes(), result.options());
    uint64_t numel = out.numel();

    DeviceContext* device = get_device_context(self);
    uint32_t workgroupSizeX = get_dtype_workgroup_size(real, 1, device);

    SpecializationBuilder spd{};
    spd.push(workgroupSizeX);
    SpecializationArgs specialization = spd.build();

    auto [seed, offset] = VulkanGeneratorImpl::reserve(generator, self.device().index());

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(cdf))
        .push(get_tensor_address(out))
        .push(numel)
        .push(static_cast<uint64_t>(n_sample))
        .push(static_cast<uint64_t>(n_categories))
        .push(seed)
        .push(offset);
    PushConstants pushConstants = pcs.build();

    uint32_t groupX = static_cast<uint32_t>((numel + workgroupSizeX - 1) / workgroupSizeX);

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::MULTINOMIAL, real},
        device->default_subgroup_size,
        specialization,
        pushConstants,
        /* read = */ {cdf},
        /* write = */ {out},
        groupX, 1, 1
    );

    if (!out.is_same(result)) result.copy_(out);
}

at::Tensor& multinomial_out_vulkan(const at::Tensor& self, int64_t n_sample, bool with_replacement, std::optional<at::Generator> generator, at::Tensor& result)
{
    TORCH_CHECK(result.device() == self.device(), "multinomial arguments must have the same device");
    TORCH_CHECK(self.dim() > 0 && self.dim() <= 2, "prob_dist must be 1 or 2 dim");
    TORCH_CHECK(at::isFloatingType(self.scalar_type()), "multinomial only supports floating-point dtypes for input, got: ", self.scalar_type());
    TORCH_CHECK(result.scalar_type() == at::kLong, "multinomial expects Long tensor out, got: ", result.scalar_type());
    TORCH_CHECK(n_sample > 0, "cannot sample n_sample <= 0 samples");
    
    int64_t n_categories = self.size(-1);
    TORCH_CHECK(with_replacement || (n_sample <= n_categories), "cannot sample n_sample > prob_dist.size(-1) samples without replacement");
    int64_t FLOAT32_MAX_CONSECUTIVE_INT = int64_t(1) << std::numeric_limits<float>::digits;
    TORCH_CHECK(n_categories <= FLOAT32_MAX_CONSECUTIVE_INT, "number of categories cannot exceed 2^24");

    if (self.dim() == 1) {
        result.resize_({n_sample});
    } else {
        result.resize_({self.size(0), n_sample});
    }
    if (result.numel() == 0) return result;

    double total = self.sum().item<double>();
    TORCH_CHECK(at::amin(self).item<double>() >= 0 && at::amax(self).item<double>() < INFINITY && !std::isnan(total),
        "probability tensor contains either `inf`, `nan` or element < 0"
    );
    int64_t zero_rows = (self.sum(-1) == 0).sum().item<int64_t>();
    TORCH_CHECK(zero_rows == 0, "invalid multinomial distribution (sum of probabilities <= 0)");

    if (with_replacement && n_sample > 1) {
        dispatch_multinomial_shader(self, n_sample, generator, result);
        return result;
    }

    // argmax / top-k of p / q with q ~ Exp(1) samples without replacement
    at::Tensor q = self / at::empty_like(self).exponential_(1, generator);
    if (n_sample == 1) {
        at::argmax_out(result, q, /*dim=*/-1, /*keepdim=*/true);
    } else {
        at::Tensor values = at::empty(result.sizes(), self.options());
        at::topk_out(values, result, q, n_sample);
    }
    return result;
}

at::Tensor multinomial_vulkan(const at::Tensor& self, int64_t n_sample, bool with_replacement, std::optional<at::Generator> generator)
{
    at::Tensor result = at::empty({0}, self.options().dtype(at::kLong));
    multinomial_out_vulkan(self, n_sample, with_replacement, std::move(generator), result);
    return result;
}

void check_randperm_fits(int64_t n, const at::Tensor& result)
{
    at::ScalarType dtype = result.scalar_type();
    if (dtype == at::kLong || dtype == at::kUInt64 || dtype == at::kBool) return;

    int64_t limit = int64_t(1) << get_dtype_digits(dtype);
    TORCH_CHECK(n <= limit, "n cannot be greater than ", limit, " for ", dtype, ", got ", n);
}

at::Tensor& randperm_out_vulkan(int64_t n, std::optional<at::Generator> generator, at::Tensor& result)
{
    TORCH_CHECK(n >= 0, "n must be non-negative, got", n);
    TORCH_CHECK(!generator.has_value() || !generator->defined() || result.device().type() == generator->device().type(),
        "Expected a '", result.device(), "' generator device but found '", generator->device(), "'"
    );
    check_randperm_fits(n, result);

    result.resize_({n});
    if (n == 0) return result;

    at::ScalarType key_dtype = random_proxy_dtype(at::kLong);
    at::Tensor keys = at::empty({n}, result.options().dtype(key_dtype)).random_(generator);
    at::Tensor order = std::get<1>(keys.sort());
    result.copy_(order);
    return result;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("multinomial", &multinomial_vulkan);
    m.impl("multinomial.out", &multinomial_out_vulkan);
    m.impl("randperm.generator_out", &randperm_out_vulkan);
}
