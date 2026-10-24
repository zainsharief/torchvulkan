#include <torch/extension.h>
#include <ATen/WrapDimUtils.h>
#include <limits>
#include <tuple>
#include <vector>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "api/ops/helpers.h"
#include "api/fallback.h"

namespace {

enum class SortMode : uint32_t {
    LOAD = 0,
    STEP = 1
};

std::optional<std::tuple<at::Tensor, at::Tensor>> dispatch_sort_shader(const at::Tensor& lines, bool descending)
{
    DeviceContext* device = get_device_context(lines);
    int64_t L = lines.size(0);
    int64_t n = lines.size(1);
    int64_t padded = 1;
    while (padded < n) padded <<= 1;

    if (!is_dtype_supported(device, lines.scalar_type()) || L * padded > std::numeric_limits<int32_t>::max()) return std::nullopt;

    // two copies of the keys and positions: each step reads one and writes the other
    at::Tensor keys[2] = {at::empty({L, padded}, lines.options()), at::empty({L, padded}, lines.options())};
    at::Tensor positions[2] = {at::empty({L, padded}, lines.options().dtype(at::kLong)), at::empty({L, padded}, lines.options().dtype(at::kLong))};
    int current = 0;
    uint32_t workgroupSizeX = get_dtype_workgroup_size(lines.scalar_type(), 1, device);

    auto dispatch = [&](SortMode mode, uint64_t threads, uint32_t block, uint32_t distance) {
        int next = mode == SortMode::LOAD ? current : 1 - current;

        SpecializationBuilder spd{};
        spd.push(static_cast<uint32_t>(mode))
           .push(static_cast<uint32_t>(descending))
           .push(workgroupSizeX);

        PushConstantBuilder pcs{};
        pcs.push(get_tensor_address(keys[current]))
           .push(get_tensor_address(positions[current]))
           .push(get_tensor_address(keys[next]))
           .push(get_tensor_address(positions[next]))
           .push(get_tensor_address(lines))
           .push(threads)
           .push(static_cast<uint32_t>(n))
           .push(static_cast<uint32_t>(padded))
           .push(block)
           .push(distance);

        std::vector<at::Tensor> reads = {keys[current], positions[current]};
        if (mode == SortMode::LOAD) reads = {lines};

        device->shader_manager->dispatchShader(
            ShaderKey{torchvulkan::Kernel::SORT, lines.scalar_type()},
            spd.build(),
            pcs.build(),
            /* read = */ reads,
            /* write = */ {keys[next], positions[next]},
            get_group_count(threads, workgroupSizeX), 1, 1
        );
        current = next;
    };

    dispatch(SortMode::LOAD, static_cast<uint64_t>(L * padded), 0, 0);
    for (int64_t block = 2; block <= padded; block <<= 1) {
        for (int64_t distance = block >> 1; distance > 0; distance >>= 1) {
            dispatch(SortMode::STEP, static_cast<uint64_t>(L * padded / 2), static_cast<uint32_t>(block), static_cast<uint32_t>(distance));
        }
    }

    return std::make_tuple(keys[current].narrow(1, 0, n), positions[current].narrow(1, 0, n));
}

std::tuple<at::Tensor, at::Tensor> sort_along(const at::Tensor& self, int64_t dim, bool descending)
{
    if (self.dim() == 0) return std::make_tuple(self.clone(), at::zeros({}, self.options().dtype(at::kLong)));

    dim = c10::maybe_wrap_dim(dim, self.dim());
    int64_t n = self.size(dim);
    if (self.numel() == 0 || n == 1) {
        return std::make_tuple(self.clone(), at::zeros(self.sizes(), self.options().dtype(at::kLong)));
    }

    at::Tensor moved = self.movedim(dim, -1).contiguous();
    at::Tensor lines = moved.view({-1, n});

    auto sorted = dispatch_sort_shader(lines, descending);
    if (!sorted.has_value()) {
        TORCHVULKAN_CPU_FALLBACK("A ", self.scalar_type(), " sort of this size can't run on the Vulkan device. Falling back to CPU.");
        auto [values, indices] = at::sort(self.cpu(), /*stable=*/true, dim, descending);
        return std::make_tuple(values.to(self.device()), indices.to(self.device()));
    }

    auto [values, indices] = *sorted;
    return std::make_tuple(values.reshape(moved.sizes()).movedim(-1, dim), indices.reshape(moved.sizes()).movedim(-1, dim));
}

std::tuple<at::Tensor&, at::Tensor&> sort_stable_out_vulkan(
    const at::Tensor& self,
    std::optional<bool> /* stable */,
    int64_t dim,
    bool descending,
    at::Tensor& values,
    at::Tensor& indices)
{
    auto [sorted_values, sorted_indices] = sort_along(self, dim, descending);
    fill_out(values, sorted_values);
    fill_out(indices, sorted_indices);
    return std::forward_as_tuple(values, indices);
}

std::tuple<at::Tensor&, at::Tensor&> topk_out_vulkan(
    const at::Tensor& self,
    c10::SymInt k_,
    int64_t dim,
    bool largest,
    bool /* sorted */,
    at::Tensor& values,
    at::Tensor& indices)
{
    int64_t k = k_.guard_int(__FILE__, __LINE__);
    int64_t d = c10::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);
    int64_t size = self.dim() == 0 ? 1 : self.size(d);
    TORCH_CHECK(k >= 0 && k <= size, "selected index k out of range");

    auto [sorted_values, sorted_indices] = sort_along(self, d, /*descending=*/largest);
    if (self.dim() > 0) {
        sorted_values = sorted_values.narrow(d, 0, k);
        sorted_indices = sorted_indices.narrow(d, 0, k);
    }
    fill_out(values, sorted_values);
    fill_out(indices, sorted_indices);
    return std::forward_as_tuple(values, indices);
}

std::tuple<at::Tensor, at::Tensor> select_sorted(const at::Tensor& sorted, const at::Tensor& order, int64_t dim, const at::Tensor& position, bool keepdim)
{
    at::Tensor values = at::gather(sorted, dim, position);
    at::Tensor indices = at::gather(order, dim, position);
    if (!keepdim) {
        values = values.squeeze(dim);
        indices = indices.squeeze(dim);
    }
    return std::make_tuple(values, indices);
}

at::Tensor nan_count(const at::Tensor& self, int64_t dim)
{
    if (!at::isFloatingType(self.scalar_type())) return at::zeros({}, self.options().dtype(at::kLong));
    return at::sum(at::isnan(self), {dim}, /*keepdim=*/true, at::kLong);
}

// the stable sort orders ties by position and NaNs last, which is the element the CPU's selection returns
std::tuple<at::Tensor, at::Tensor> median_along(const at::Tensor& self, int64_t dim, bool keepdim, bool ignore_nan, const char* name)
{
    if (self.dim() == 0) return std::make_tuple(self.clone(), at::zeros({}, self.options().dtype(at::kLong)));
    int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    int64_t n = self.size(d);
    TORCH_CHECK_INDEX(n > 0, name, "(): Expected reduction dim ", d, " to have non-zero size.");

    auto [sorted, order] = sort_along(self, d, /*descending=*/false);
    at::Tensor nans = nan_count(self, d);
    std::vector<int64_t> kept_shape(self.sizes().begin(), self.sizes().end());
    kept_shape[d] = 1;

    at::Tensor position;
    if (ignore_nan) {
        at::Tensor valid = at::rsub(nans, n);
        position = at::clamp_min(at::floor_divide(at::sub(valid, 1), 2), 0);
    } else {
        position = at::where(at::gt(nans, 0), at::rsub(nans, n), at::full_like(nans, (n - 1) / 2));
    }
    return select_sorted(sorted, order, d, position.expand(kept_shape).contiguous(), keepdim);
}

std::tuple<at::Tensor&, at::Tensor&> median_dim_values_vulkan(const at::Tensor& self, int64_t dim, bool keepdim, at::Tensor& values, at::Tensor& indices)
{
    auto [v, i] = median_along(self, dim, keepdim, /*ignore_nan=*/false, "median");
    fill_out(values, v);
    fill_out(indices, i);
    return std::forward_as_tuple(values, indices);
}

std::tuple<at::Tensor&, at::Tensor&> nanmedian_dim_values_vulkan(const at::Tensor& self, int64_t dim, bool keepdim, at::Tensor& values, at::Tensor& indices)
{
    auto [v, i] = median_along(self, dim, keepdim, /*ignore_nan=*/true, "nanmedian");
    fill_out(values, v);
    fill_out(indices, i);
    return std::forward_as_tuple(values, indices);
}

at::Tensor median_all(const at::Tensor& self, bool ignore_nan, const char* name)
{
    if (self.numel() == 0) {
        TORCH_CHECK(at::isFloatingType(self.scalar_type()), name, "() is not implemented for empty integer tensors");
        return at::full({}, std::numeric_limits<double>::quiet_NaN(), self.options());
    }
    return std::get<0>(median_along(self.reshape({-1}), 0, /*keepdim=*/false, ignore_nan, name));
}

at::Tensor median_vulkan(const at::Tensor& self)
{
    return median_all(self, /*ignore_nan=*/false, "median");
}

at::Tensor nanmedian_vulkan(const at::Tensor& self)
{
    return median_all(self, /*ignore_nan=*/true, "nanmedian");
}

std::tuple<at::Tensor&, at::Tensor&> kthvalue_values_vulkan(const at::Tensor& self, int64_t k, int64_t dim, bool keepdim, at::Tensor& values, at::Tensor& indices)
{
    int64_t d = c10::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);
    int64_t n = self.dim() == 0 ? 1 : self.size(d);
    TORCH_CHECK(k >= 1 && k <= n, "kthvalue(): selected number k out of range for dimension ", d);
    if (self.dim() == 0) {
        fill_out(values, self.clone());
        fill_out(indices, at::zeros({}, self.options().dtype(at::kLong)));
        return std::forward_as_tuple(values, indices);
    }

    auto [sorted, order] = sort_along(self, d, /*descending=*/false);
    std::vector<int64_t> kept_shape(self.sizes().begin(), self.sizes().end());
    kept_shape[d] = 1;
    at::Tensor position = at::full(kept_shape, k - 1, self.options().dtype(at::kLong));
    auto [v, i] = select_sorted(sorted, order, d, position, keepdim);
    fill_out(values, v);
    fill_out(indices, i);
    return std::forward_as_tuple(values, indices);
}

// the most frequent value, the smallest on a tie, at the index of its last occurrence
std::tuple<at::Tensor, at::Tensor> mode_vulkan(const at::Tensor& self, int64_t dim, bool keepdim)
{
    if (self.dim() == 0) return std::make_tuple(self.clone(), at::zeros({}, self.options().dtype(at::kLong)));
    int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    int64_t n = self.size(d);
    TORCH_CHECK_INDEX(n > 0, "mode(): Expected reduction dim ", d, " to have non-zero size.");

    auto [sorted, order] = sort_along(self, d, /*descending=*/false);
    at::Tensor s = sorted.movedim(d, -1).contiguous();
    at::Tensor o = order.movedim(d, -1).contiguous();

    at::Tensor first = at::ones(s.sizes().vec(), s.options().dtype(at::kBool)).narrow(-1, 0, 1);
    at::Tensor starts = n == 1 ? first : at::cat({first, at::ne(s.narrow(-1, 1, n - 1), s.narrow(-1, 0, n - 1))}, -1);
    at::Tensor last = first.clone();
    at::Tensor ends = n == 1 ? last : at::cat({starts.narrow(-1, 1, n - 1), last}, -1);

    std::vector<int64_t> position_shape(s.dim(), 1);
    position_shape.back() = n;
    at::Tensor positions = at::arange(n, s.options().dtype(at::kLong)).view(position_shape).expand(s.sizes());
    at::Tensor run_start = std::get<0>(at::cummax(at::where(starts, positions, at::zeros_like(positions)), -1));
    at::Tensor run_length = at::where(ends, at::add(at::sub(positions, run_start), 1), at::zeros_like(positions));

    at::Tensor best = at::argmax(run_length, -1, /*keepdim=*/true);
    at::Tensor values = at::gather(s, -1, best).squeeze(-1);
    at::Tensor indices = at::gather(o, -1, best).squeeze(-1);
    if (keepdim) {
        values = values.unsqueeze(d);
        indices = indices.unsqueeze(d);
    }
    return std::make_tuple(values, indices);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("sort.values_stable", &sort_stable_out_vulkan);
    m.impl("topk.values", &topk_out_vulkan);
    m.impl("median", &median_vulkan);
    m.impl("median.dim_values", &median_dim_values_vulkan);
    m.impl("nanmedian", &nanmedian_vulkan);
    m.impl("nanmedian.dim_values", &nanmedian_dim_values_vulkan);
    m.impl("kthvalue.values", &kthvalue_values_vulkan);
    m.impl("mode", &mode_vulkan);
}
