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
        .push(get_output_address(out))
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
    TORCH_CHECK(result.device() == self.device(), "Expected all tensors to be on the same device, but found at least two devices, ",
        self.device(), " and ", result.device(), "!");
    TORCH_CHECK(self.dim() > 0 && self.dim() <= 2, "prob_dist must be 1 or 2 dim");
    TORCH_CHECK(at::isFloatingType(self.scalar_type()), "multinomial only supports floating-point dtypes for input, got: ", self.scalar_type());
    TORCH_CHECK(result.scalar_type() == at::kLong, "multinomial expects Long tensor out, got: ", result.scalar_type());
    TORCH_CHECK(n_sample > 0, "cannot sample n_sample <= 0 samples");
    
    int64_t n_categories = self.size(-1);
    TORCH_CHECK(with_replacement || (n_sample <= n_categories), "cannot sample n_sample > prob_dist.size(-1) samples without replacement");
    int64_t FLOAT32_MAX_CONSECUTIVE_INT = int64_t(1) << std::numeric_limits<float>::digits;
    TORCH_CHECK(n_categories <= FLOAT32_MAX_CONSECUTIVE_INT, "number of categories cannot exceed 2^24");

    if (self.dim() == 1) {
        at::native::resize_output(result, {n_sample});
    } else {
        at::native::resize_output(result, {self.size(0), n_sample});
    }
    if (result.numel() == 0) return result;

    // both checks in one read back: a draw is often too small for its syncs not to dominate
    at::Tensor invalid = at::logical_or(at::lt(self, 0), at::logical_not(at::isfinite(self))).any();
    at::Tensor zero_row = at::eq(self.sum(-1), 0).any();
    at::Tensor flags = at::stack({invalid, zero_row}).cpu();
    const bool* flag = flags.const_data_ptr<bool>();
    TORCH_CHECK(!flag[0], "probability tensor contains either `inf`, `nan` or element < 0");
    TORCH_CHECK(!flag[1], "invalid multinomial distribution (sum of probabilities <= 0)");

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

// torch's check_supported_max_int_with_precision: n - 1 must fit the dtype, and a float must hold every integer up to it
void check_randperm_fits(int64_t n, const at::Tensor& result)
{
    at::ScalarType dtype = result.scalar_type();
    if (dtype == at::kBool) return;
    check_scalar_convertible(n > 0 ? n - 1 : n, dtype);
    if (!at::isFloatingType(dtype)) return;
    int digits = get_dtype_digits(dtype);
    TORCH_CHECK(n <= (int64_t(1) << digits) + 1, "n cannot be greater than 2^", digits, "+1 for ", dtype);
}

at::Tensor& randperm_out_vulkan(int64_t n, std::optional<at::Generator> generator, at::Tensor& result)
{
    TORCH_CHECK(n >= 0, "n must be non-negative, got", n);
    // n = 0 draws nothing, so any generator goes
    if (n == 0) {
        at::native::resize_output(result, {0});
        return result;
    }
    TORCH_CHECK(!generator.has_value() || !generator->defined() || result.device().type() == generator->device().type(),
        "Expected a '", result.device().type(), "' device type for generator but found '", generator->device().type(), "'"
    );
    check_randperm_fits(n, result);

    at::native::resize_output(result, {n});
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
