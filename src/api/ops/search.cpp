#include <torch/extension.h>
#include <ATen/WrapDimUtils.h>
#include <tuple>
#include <vector>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "api/ops/helpers.h"
#include "api/fallback.h"

namespace {

void dispatch_searchsorted_shader(
    const at::Tensor& sorted,
    const at::Tensor& sorter,
    const at::Tensor& values,
    const at::Tensor& out,
    bool right,
    bool shared,
    int64_t values_per_row)
{
    uint64_t count = values.numel();
    if (count == 0) return;

    DeviceContext* device = get_device_context(out);
    at::ScalarType dtype = sorted.scalar_type();
    uint32_t workgroupSizeX = get_dtype_workgroup_size(dtype, 1, device);

    SpecializationBuilder spd{};
    spd.push(static_cast<uint32_t>(right))
       .push(static_cast<uint32_t>(sorter.defined()))
       .push(static_cast<uint32_t>(out.scalar_type() == at::kInt))
       .push(workgroupSizeX);

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(sorted))
       .push(sorter.defined() ? get_tensor_address(sorter) : uint64_t(0))
       .push(get_tensor_address(values))
       .push(get_tensor_address(out))
       .push(count)
       .push(static_cast<uint32_t>(sorted.size(-1)))
       .push(static_cast<uint32_t>(std::max<int64_t>(values_per_row, 1)))
       .push(static_cast<uint32_t>(shared));

    std::vector<at::Tensor> reads = {sorted, values};
    if (sorter.defined()) reads.push_back(sorter);

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::SEARCHSORTED, dtype},
        spd.build(),
        pcs.build(),
        /* read = */ reads,
        /* write = */ {out},
        get_group_count(count, workgroupSizeX), 1, 1
    );
}

at::Tensor searchsorted_impl(
    const at::Tensor& sorted_sequence,
    const at::Tensor& self,
    bool out_int32,
    bool right,
    c10::optional<c10::string_view> side,
    const c10::optional<at::Tensor>& sorter)
{
    if (side.has_value()) {
        TORCH_CHECK(*side == "left" || *side == "right", "torch.searchsorted(): side can only be 'left' or 'right' but got ", *side);
        TORCH_CHECK(!right || *side == "right", "torch.searchsorted(): side and right can't be set to opposites, got side of left while right was True");
        right = *side == "right";
    }
    TORCH_CHECK(sorted_sequence.dim() >= 1, "torch.searchsorted(): boundaries tensor should have positive dimension, but got 0 dimension");
    bool shared = sorted_sequence.dim() == 1;
    if (!shared) {
        TORCH_CHECK(sorted_sequence.dim() == self.dim() && sorted_sequence.sizes().slice(0, self.dim() - 1).equals(self.sizes().slice(0, self.dim() - 1)),
            "torch.searchsorted(): boundaries tensor should be 1 dimension or the first N-1 dimensions of boundaries tensor and input value tensor must match, but we got boundaries tensor ",
            sorted_sequence.sizes(), " and input value tensor ", self.sizes()
        );
    }
    bool has_sorter = sorter.has_value() && sorter->defined();
    if (has_sorter) {
        TORCH_CHECK(sorter->scalar_type() == at::kLong, "torch.searchsorted(): sorter must be a tensor of long dtype but got dtype ", sorter->scalar_type());
        TORCH_CHECK(sorter->sizes().equals(sorted_sequence.sizes()), "torch.searchsorted(): boundary and sorter must have the same size, but got boundary tensor ",
            sorted_sequence.sizes(), " and got sorter tensor ", sorter->sizes()
        );
    }

    at::ScalarType dtype = at::result_type(sorted_sequence, self);
    at::Tensor sorted = sorted_sequence.to(dtype).contiguous();
    at::Tensor values = self.to(sorted.device(), dtype).contiguous();
    at::Tensor out = at::empty(self.sizes(), values.options().dtype(out_int32 ? at::kInt : at::kLong));
    if (out.numel() == 0) return out;
    if (sorted.size(-1) == 0) return out.zero_();

    if (!is_dtype_supported(sorted, dtype) || values.numel() >= std::numeric_limits<uint32_t>::max()) {
        TORCHVULKAN_CPU_FALLBACK("This searchsorted can't run on the Vulkan device. Falling back to CPU.");
        return at::searchsorted(sorted.cpu(), values.cpu(), out_int32, right, c10::nullopt, has_sorter ? c10::optional<at::Tensor>(sorter->cpu()) : c10::nullopt).to(self.device());
    }
    at::Tensor order = has_sorter ? sorter->contiguous() : at::Tensor();
    dispatch_searchsorted_shader(sorted, order, values, out, right, shared, self.dim() == 0 ? 1 : self.size(-1));
    return out;
}

at::Tensor searchsorted_tensor_vulkan(
    const at::Tensor& sorted_sequence,
    const at::Tensor& self,
    bool out_int32,
    bool right,
    c10::optional<c10::string_view> side,
    const c10::optional<at::Tensor>& sorter)
{
    return searchsorted_impl(sorted_sequence, self, out_int32, right, side, sorter);
}

at::Tensor& searchsorted_tensor_out_vulkan(
    const at::Tensor& sorted_sequence,
    const at::Tensor& self,
    bool out_int32,
    bool right,
    c10::optional<c10::string_view> side,
    const c10::optional<at::Tensor>& sorter,
    at::Tensor& out)
{
    return fill_out(out, searchsorted_impl(sorted_sequence, self, out_int32, right, side, sorter));
}

at::Tensor scalar_value(const at::Scalar& value, const at::Tensor& like)
{
    at::Tensor t = at::scalar_tensor(value, like.options().dtype(value.isFloatingPoint() ? at::kDouble : value.isBoolean() ? at::kBool : at::kLong));
    t.unsafeGetTensorImpl()->set_wrapped_number(true);
    return t;
}

at::Tensor searchsorted_scalar_vulkan(
    const at::Tensor& sorted_sequence,
    const at::Scalar& self,
    bool out_int32,
    bool right,
    c10::optional<c10::string_view> side,
    const c10::optional<at::Tensor>& sorter)
{
    TORCH_CHECK(sorted_sequence.dim() == 1, "torch.searchsorted(): input value can be a scalar only when boundaries tensor dimension is 1, but we got boundaries tensor dim(",
        sorted_sequence.dim(), ") and input value's dim(0) numel(1)"
    );
    return searchsorted_impl(sorted_sequence, scalar_value(self, sorted_sequence), out_int32, right, side, sorter);
}

at::Tensor& searchsorted_scalar_out_vulkan(
    const at::Tensor& sorted_sequence,
    const at::Scalar& self,
    bool out_int32,
    bool right,
    c10::optional<c10::string_view> side,
    const c10::optional<at::Tensor>& sorter,
    at::Tensor& out)
{
    return fill_out(out, searchsorted_scalar_vulkan(sorted_sequence, self, out_int32, right, side, sorter));
}

at::Tensor bucketize_tensor_vulkan(const at::Tensor& self, const at::Tensor& boundaries, bool out_int32, bool right)
{
    TORCH_CHECK(boundaries.dim() == 1, "torch.bucketize(): boundaries tensor must be 1 dimension, but got dim(", boundaries.dim(), ")");
    return searchsorted_impl(boundaries, self, out_int32, right, c10::nullopt, c10::nullopt);
}

at::Tensor& bucketize_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& boundaries, bool out_int32, bool right, at::Tensor& out)
{
    return fill_out(out, bucketize_tensor_vulkan(self, boundaries, out_int32, right));
}

at::Tensor bucketize_scalar_vulkan(const at::Scalar& self, const at::Tensor& boundaries, bool out_int32, bool right)
{
    return bucketize_tensor_vulkan(scalar_value(self, boundaries), boundaries, out_int32, right);
}

at::Tensor accumulate_bins(const at::Tensor& keys, const c10::optional<at::Tensor>& weight, int64_t bins, const at::TensorOptions& options)
{
    at::Tensor hist = at::zeros({bins + 1}, options);
    if (keys.numel() == 0) return hist.narrow(0, 0, bins);
    at::Tensor values = weight.has_value() && weight->defined() ? weight->reshape({-1}).to(options.dtype()) : at::ones({keys.numel()}, options);
    hist.index_add_(0, keys.reshape({-1}), values);
    return hist.narrow(0, 0, bins);
}

at::Tensor histc_vulkan(const at::Tensor& self, int64_t bins, const at::Scalar& min, const at::Scalar& max)
{
    TORCH_CHECK(bins > 0, "torch.histc(): bins must be > 0, but got ", bins, " for dimension 0");
    TORCH_CHECK(at::isFloatingType(self.scalar_type()), "torch.histogram: input tensor and bins tensors should have the same dtype, but got input ",
        self.scalar_type(), " and bins ", self.scalar_type()
    );
    double lo = min.toDouble();
    double hi = max.toDouble();
    if (lo == hi && self.numel() > 0) {
        auto [mn, mx] = at::aminmax(self);
        lo = mn.item<double>();
        hi = mx.item<double>();
    }
    if (lo == hi) {
        lo -= 1;
        hi += 1;
    }
    TORCH_CHECK(lo < hi, "torch.histc(): max must be larger than min");
    TORCH_CHECK(std::isfinite(lo) && std::isfinite(hi), "torch.histc(): range of [", lo, ", ", hi, "] is not finite");

    // every step rounded to the input dtype, as the CPU; x == hi goes in the last bin
    at::Tensor x = self.reshape({-1});
    at::Tensor span = at::sub(at::scalar_tensor(hi, at::TensorOptions().dtype(self.scalar_type())), at::scalar_tensor(lo, at::TensorOptions().dtype(self.scalar_type())));
    at::Tensor scaled = at::div(at::mul(at::sub(x, lo), static_cast<double>(bins)), span);
    at::Tensor position = at::clamp_max(at::trunc(scaled).to(at::kLong), bins - 1);
    at::Tensor inside = at::logical_and(at::ge(x, lo), at::le(x, hi));
    at::Tensor keys = at::where(inside, position, at::full_like(position, bins));
    return accumulate_bins(keys, c10::nullopt, bins, self.options());
}

at::Tensor& histc_out_vulkan(const at::Tensor& self, int64_t bins, const at::Scalar& min, const at::Scalar& max, at::Tensor& out)
{
    return fill_out(out, histc_vulkan(self, bins, min, max));
}

std::vector<std::pair<double, double>> outer_bin_edges(const at::Tensor& input, int64_t dims, c10::optional<at::ArrayRef<double>> range)
{
    std::vector<std::pair<double, double>> edges(dims, {0.0, 1.0});
    if (range.has_value()) {
        TORCH_CHECK(static_cast<int64_t>(range->size()) == 2 * dims, "torch.histogramdd: for a ", dims, "-dimensional histogram range should have ",
            2 * dims, " elements, but got ", range->size()
        );
        for (int64_t d = 0; d < dims; d++) edges[d] = {(*range)[2 * d], (*range)[2 * d + 1]};
    } else if (input.numel() > 0) {
        auto [mn, mx] = at::aminmax(input, 0);
        at::Tensor lo = mn.to(at::kDouble).cpu();
        at::Tensor hi = mx.to(at::kDouble).cpu();
        for (int64_t d = 0; d < dims; d++) edges[d] = {lo[d].item<double>(), hi[d].item<double>()};
    }
    for (int64_t d = 0; d < dims; d++) {
        auto& [lo, hi] = edges[d];
        TORCH_CHECK(std::isfinite(lo) && std::isfinite(hi), "torch.histogramdd: dimension ", d, "'s range [", lo, ", ", hi, "] is not finite");
        TORCH_CHECK(lo <= hi, "torch.histogramdd: min should not exceed max, but got min ", lo, " max ", hi, " for dimension ", d);
        if (lo == hi) {
            lo -= 0.5;
            hi += 0.5;
        }
    }
    return edges;
}

std::vector<at::Tensor> histogramdd_bin_edges_vulkan(
    const at::Tensor& self,
    at::IntArrayRef bin_ct,
    c10::optional<at::ArrayRef<double>> range,
    const c10::optional<at::Tensor>& /* weight */,
    bool /* density */)
{
    int64_t dims = self.size(-1);
    TORCH_CHECK(static_cast<int64_t>(bin_ct.size()) == dims, "torch.histogramdd: expected ", dims, " sequences of bin edges for a ", dims,
        "-dimensional histogram but got ", bin_ct.size()
    );
    at::Tensor input = self.reshape({-1, dims});
    auto outer = outer_bin_edges(input, dims, range);
    std::vector<at::Tensor> edges;
    for (int64_t d = 0; d < dims; d++) {
        TORCH_CHECK(bin_ct[d] > 0, "torch.histogramdd(): bins must be > 0, but got ", bin_ct[d], " for dimension ", d);
        edges.push_back(at::linspace(outer[d].first, outer[d].second, bin_ct[d] + 1, self.options()));
    }
    return edges;
}

// a value equal to the last edge goes in the last bin
at::Tensor histogramdd_from_edges(const at::Tensor& self, at::TensorList bins, const c10::optional<at::Tensor>& weight, bool density)
{
    int64_t dims = self.size(-1);
    TORCH_CHECK(static_cast<int64_t>(bins.size()) == dims, "torch.histogramdd: expected ", dims, " sequences of bin edges for a ", dims,
        "-dimensional histogram but got ", bins.size()
    );
    at::Tensor input = self.reshape({-1, dims});
    std::vector<int64_t> hist_shape;
    for (const at::Tensor& edges : bins) {
        TORCH_CHECK(edges.dim() == 1, "torch.histogramdd: bins tensor should have one dimension");
        TORCH_CHECK(edges.numel() > 0, "torch.histogramdd: bins tensor should have at least 1 element");
        hist_shape.push_back(edges.numel() - 1);
    }
    int64_t total = 1;
    for (int64_t s : hist_shape) total *= s;

    at::Tensor keys = at::zeros({input.size(0)}, input.options().dtype(at::kLong));
    at::Tensor inside = at::ones({input.size(0)}, input.options().dtype(at::kBool));
    int64_t stride = total;
    for (int64_t d = 0; d < dims; d++) {
        at::Tensor column = input.select(1, d).contiguous();
        const at::Tensor& edges = bins[d];
        int64_t nbins = hist_shape[d];
        stride = nbins == 0 ? 0 : stride / nbins;
        at::Tensor position = at::sub(at::searchsorted(edges, column, /*out_int32=*/false, /*right=*/true), 1);
        position = at::clamp_max(position, nbins - 1);
        inside = at::logical_and(inside, at::logical_and(at::ge(column, edges[0]), at::le(column, edges[-1])));
        keys = at::add(keys, at::mul(position, stride));
    }
    if (weight.has_value() && weight->defined()) {
        TORCH_CHECK(weight->numel() == input.size(0), "torch.histogramdd: weight should have the same shape as input excluding its innermost dimension");
    }
    keys = at::where(inside, keys, at::full_like(keys, total));
    at::Tensor hist = accumulate_bins(keys, weight, total, self.options()).view(hist_shape);

    if (density && total > 0) {
        at::Tensor normalized = at::div(hist, hist.sum());
        for (int64_t d = 0; d < dims; d++) {
            at::Tensor widths = at::sub(bins[d].narrow(0, 1, hist_shape[d]), bins[d].narrow(0, 0, hist_shape[d]));
            std::vector<int64_t> view(dims, 1);
            view[d] = hist_shape[d];
            normalized = at::div(normalized, widths.view(view));
        }
        hist = normalized;
    }
    return hist;
}

at::Tensor histogramdd_from_bin_tensors_vulkan(const at::Tensor& self, at::TensorList bins, const c10::optional<at::Tensor>& weight, bool density)
{
    return histogramdd_from_edges(self, bins, weight, density);
}

at::Tensor histogramdd_from_bin_cts_vulkan(
    const at::Tensor& self,
    at::IntArrayRef bins,
    c10::optional<at::ArrayRef<double>> range,
    const c10::optional<at::Tensor>& weight,
    bool density)
{
    std::vector<at::Tensor> edges = histogramdd_bin_edges_vulkan(self, bins, range, weight, density);
    return histogramdd_from_edges(self, edges, weight, density);
}

std::tuple<at::Tensor, at::Tensor> histogram_bins_tensor_vulkan(const at::Tensor& self, const at::Tensor& bins, const c10::optional<at::Tensor>& weight, bool density)
{
    TORCH_CHECK(self.scalar_type() == bins.scalar_type(), "torch.histogram: input tensor and bins tensors should have the same dtype, but got input ",
        self.scalar_type(), " and bins ", bins.scalar_type()
    );
    at::Tensor hist = histogramdd_from_edges(self.reshape({-1, 1}), {bins}, weight.has_value() ? c10::optional<at::Tensor>(weight->reshape({-1})) : c10::nullopt, density);
    return std::make_tuple(hist, bins.clone());
}

std::tuple<at::Tensor&, at::Tensor&> histogram_bins_tensor_out_vulkan(
    const at::Tensor& self,
    const at::Tensor& bins,
    const c10::optional<at::Tensor>& weight,
    bool density,
    at::Tensor& hist,
    at::Tensor& bin_edges)
{
    auto [h, e] = histogram_bins_tensor_vulkan(self, bins, weight, density);
    fill_out(hist, h);
    fill_out(bin_edges, e);
    return std::forward_as_tuple(hist, bin_edges);
}

std::tuple<at::Tensor, at::Tensor> histogram_bin_ct_vulkan(
    const at::Tensor& self,
    int64_t bins,
    c10::optional<at::ArrayRef<double>> range,
    const c10::optional<at::Tensor>& weight,
    bool density)
{
    at::Tensor input = self.reshape({-1, 1});
    at::Tensor edges = histogramdd_bin_edges_vulkan(input, {bins}, range, weight, density)[0];
    at::Tensor hist = histogramdd_from_edges(input, {edges}, weight.has_value() ? c10::optional<at::Tensor>(weight->reshape({-1})) : c10::nullopt, density);
    return std::make_tuple(hist, edges);
}

std::tuple<at::Tensor&, at::Tensor&> histogram_bin_ct_out_vulkan(
    const at::Tensor& self,
    int64_t bins,
    c10::optional<at::ArrayRef<double>> range,
    const c10::optional<at::Tensor>& weight,
    bool density,
    at::Tensor& hist,
    at::Tensor& bin_edges)
{
    auto [h, e] = histogram_bin_ct_vulkan(self, bins, range, weight, density);
    fill_out(hist, h);
    fill_out(bin_edges, e);
    return std::forward_as_tuple(hist, bin_edges);
}

at::Tensor bincount_vulkan(const at::Tensor& self, const c10::optional<at::Tensor>& weights, int64_t minlength)
{
    TORCH_CHECK(self.dim() == 1 && !at::isFloatingType(self.scalar_type()) && self.scalar_type() != at::kBool, "bincount only supports 1-d non-negative integral inputs.");
    TORCH_CHECK(minlength >= 0, "minlength should be >= 0");
    bool has_weights = weights.has_value() && weights->defined();
    if (has_weights) TORCH_CHECK(weights->dim() == 1 && weights->size(0) == self.size(0), "weights should be 1-d and have the same length as input");
    at::ScalarType dtype = !has_weights ? at::kLong : weights->scalar_type() == at::kFloat ? at::kFloat : at::kDouble;

    int64_t length = minlength;
    if (self.numel() > 0) {
        auto [mn, mx] = at::aminmax(self);
        TORCH_CHECK(mn.item<int64_t>() >= 0, "bincount only supports 1-d non-negative integral inputs.");
        length = std::max(length, mx.item<int64_t>() + 1);
    }
    return accumulate_bins(self.to(at::kLong), has_weights ? c10::optional<at::Tensor>(weights->to(dtype)) : c10::nullopt, length, self.options().dtype(dtype));
}

at::Tensor isin_impl(const at::Tensor& elements, const at::Tensor& test_elements, bool invert)
{
    at::ScalarType dtype = at::result_type(elements, test_elements);
    at::Tensor values = elements.to(dtype);
    at::Tensor test = test_elements.to(values.device(), dtype).reshape({-1});
    at::Tensor found;
    if (test.numel() == 0 || values.numel() == 0) {
        found = at::zeros(values.sizes(), values.options().dtype(at::kBool));
    } else {
        at::Tensor sorted = std::get<0>(at::sort(test, /*stable=*/false, 0, false));
        at::Tensor position = at::clamp_max(at::searchsorted(sorted, values.contiguous()), test.numel() - 1);
        found = at::eq(at::index_select(sorted, 0, position.reshape({-1})).view(values.sizes()), values);
    }
    return invert ? at::logical_not(found) : found;
}

at::Tensor& isin_tensor_tensor_out_vulkan(const at::Tensor& elements, const at::Tensor& test_elements, bool /* assume_unique */, bool invert, at::Tensor& out)
{
    return fill_out(out, isin_impl(elements, test_elements, invert));
}

at::Tensor& isin_tensor_scalar_out_vulkan(const at::Tensor& elements, const at::Scalar& test_element, bool /* assume_unique */, bool invert, at::Tensor& out)
{
    return fill_out(out, isin_impl(elements, scalar_value(test_element, elements), invert));
}

at::Tensor& isin_scalar_tensor_out_vulkan(const at::Scalar& element, const at::Tensor& test_elements, bool /* assume_unique */, bool invert, at::Tensor& out)
{
    return fill_out(out, isin_impl(scalar_value(element, test_elements).to(test_elements.device()), test_elements, invert));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_runs(const at::Tensor& rows)
{
    int64_t n = rows.size(0);
    at::TensorOptions long_options = rows.options().dtype(at::kLong);
    if (n == 0) return std::make_tuple(rows.clone(), at::empty({0}, long_options), at::empty({0}, long_options));

    at::Tensor starts = at::ones({1}, rows.options().dtype(at::kBool));
    if (n > 1) {
        at::Tensor differs = at::ne(rows.narrow(0, 1, n - 1), rows.narrow(0, 0, n - 1));
        if (differs.dim() > 1) differs = differs.reshape({n - 1, -1}).any(1);
        starts = at::cat({starts, differs}, 0);
    }
    at::Tensor run_id = at::sub(at::cumsum(starts, 0, at::kLong), 1);
    at::Tensor start_positions = at::masked_select(at::arange(n, long_options), starts);
    at::Tensor ends = at::cat({start_positions.narrow(0, 1, start_positions.numel() - 1), at::full({1}, n, long_options)}, 0);
    at::Tensor unique = at::index_select(rows, 0, start_positions);
    return std::make_tuple(unique, run_id, at::sub(ends, start_positions));
}

at::Tensor unpermute(const at::Tensor& values, const at::Tensor& order)
{
    return at::empty_like(values).index_copy_(0, order, values);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_sorted(const at::Tensor& self, bool return_inverse, bool return_counts)
{
    at::Tensor flat = self.reshape({-1});
    auto [sorted, order] = at::sort(flat, /*stable=*/true, 0, false);
    auto [unique, run_id, counts] = unique_runs(sorted);
    at::Tensor inverse = return_inverse ? unpermute(run_id, order).view(self.sizes()) : at::empty({0}, self.options().dtype(at::kLong));
    if (!return_counts) counts = at::empty({0}, self.options().dtype(at::kLong));
    return std::make_tuple(unique, inverse, counts);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> unique2_vulkan(const at::Tensor& self, bool /* sorted */, bool return_inverse, bool return_counts)
{
    return unique_sorted(self, return_inverse, return_counts);
}

std::tuple<at::Tensor, at::Tensor> unique_vulkan(const at::Tensor& self, bool /* sorted */, bool return_inverse)
{
    auto [unique, inverse, counts] = unique_sorted(self, return_inverse, false);
    return std::make_tuple(unique, inverse);
}

at::Tensor rows_along(const at::Tensor& self, int64_t dim)
{
    at::Tensor moved = self.movedim(dim, 0).contiguous();
    int64_t rest = 1;
    for (int64_t d = 1; d < moved.dim(); d++) rest *= moved.size(d);
    return moved.reshape({moved.size(0), rest});
}

at::Tensor rows_back(const at::Tensor& rows, const at::Tensor& self, int64_t dim)
{
    std::vector<int64_t> shape = self.movedim(dim, 0).sizes().vec();
    shape[0] = rows.size(0);
    return rows.reshape(shape).movedim(0, dim);
}

void check_unique_dim(const at::Tensor& self, int64_t dim)
{
    for (int64_t d = 0; d < self.dim(); d++) {
        TORCH_CHECK(d == dim || self.size(d) > 0, "There are 0 sized dimensions, and they aren't selected, so unique cannot be applied");
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_consecutive_vulkan(const at::Tensor& self, bool return_inverse, bool return_counts, c10::optional<int64_t> dim)
{
    if (dim.has_value() && self.dim() > 0) {
        int64_t d = c10::maybe_wrap_dim(*dim, self.dim());
        check_unique_dim(self, d);
        auto [unique, run_id, counts] = unique_runs(rows_along(self, d));
        return std::make_tuple(rows_back(unique, self, d), return_inverse ? run_id : at::empty({0}, run_id.options()),
            return_counts ? counts : at::empty({0}, counts.options()));
    }
    auto [unique, run_id, counts] = unique_runs(self.reshape({-1}));
    return std::make_tuple(unique, return_inverse ? run_id.view(self.sizes()) : at::empty({0}, run_id.options()),
        return_counts ? counts : at::empty({0}, counts.options()));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_dim_vulkan(const at::Tensor& self, int64_t dim, bool /* sorted */, bool /* return_inverse */, bool /* return_counts */)
{
    int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    check_unique_dim(self, d);
    at::Tensor rows = rows_along(self, d);
    int64_t n = rows.size(0);
    if (n == 0) {
        at::Tensor empty_long = at::empty({0}, self.options().dtype(at::kLong));
        return std::make_tuple(self.clone(), empty_long, empty_long);
    }

    at::Tensor order = at::arange(n, rows.options().dtype(at::kLong));
    for (int64_t c = rows.size(1) - 1; c >= 0; c--) {
        at::Tensor column = at::index_select(rows.select(1, c), 0, order);
        at::Tensor step = std::get<1>(at::sort(column, /*stable=*/true, 0, false));
        order = at::index_select(order, 0, step);
    }
    auto [unique, run_id, counts] = unique_runs(at::index_select(rows, 0, order));
    return std::make_tuple(rows_back(unique, self, d), unpermute(run_id, order), counts);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_dim_consecutive_vulkan(const at::Tensor& self, int64_t dim, bool return_inverse, bool return_counts)
{
    return unique_consecutive_vulkan(self, return_inverse, return_counts, dim);
}

at::Tensor repeat_interleave_vulkan(const at::Tensor& repeats, c10::optional<c10::SymInt> output_size)
{
    TORCH_CHECK(repeats.dim() == 1, "repeat_interleave only accept 1D vector as repeat");
    TORCH_CHECK(repeats.scalar_type() == at::kLong || repeats.scalar_type() == at::kInt, "repeats has to be Long or Int tensor");
    if (repeats.numel() == 0) return at::empty({0}, repeats.options().dtype(repeats.scalar_type()));
    TORCH_CHECK(at::lt(repeats, 0).sum().item<int64_t>() == 0, "repeats can not be negative");
    at::Tensor running = at::cumsum(repeats, 0, at::kLong);
    int64_t total = output_size.has_value() ? output_size->guard_int(__FILE__, __LINE__) : running[-1].item<int64_t>();
    at::Tensor positions = at::arange(total, repeats.options().dtype(at::kLong));
    return at::searchsorted(running, positions, /*out_int32=*/false, /*right=*/true).to(repeats.scalar_type());
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("searchsorted.Tensor", &searchsorted_tensor_vulkan);
    m.impl("searchsorted.Tensor_out", &searchsorted_tensor_out_vulkan);
    m.impl("searchsorted.Scalar", &searchsorted_scalar_vulkan);
    m.impl("searchsorted.Scalar_out", &searchsorted_scalar_out_vulkan);
    m.impl("bucketize.Tensor", &bucketize_tensor_vulkan);
    m.impl("bucketize.Tensor_out", &bucketize_tensor_out_vulkan);
    m.impl("bucketize.Scalar", &bucketize_scalar_vulkan);
    m.impl("histc", &histc_vulkan);
    m.impl("histc.out", &histc_out_vulkan);
    m.impl("histogram.bins_tensor", &histogram_bins_tensor_vulkan);
    m.impl("histogram.bins_tensor_out", &histogram_bins_tensor_out_vulkan);
    m.impl("histogram.bin_ct", &histogram_bin_ct_vulkan);
    m.impl("histogram.bin_ct_out", &histogram_bin_ct_out_vulkan);
    m.impl("_histogramdd_bin_edges", &histogramdd_bin_edges_vulkan);
    m.impl("_histogramdd_from_bin_cts", &histogramdd_from_bin_cts_vulkan);
    m.impl("_histogramdd_from_bin_tensors", &histogramdd_from_bin_tensors_vulkan);
    m.impl("bincount", &bincount_vulkan);
    m.impl("isin.Tensor_Tensor_out", &isin_tensor_tensor_out_vulkan);
    m.impl("isin.Tensor_Scalar_out", &isin_tensor_scalar_out_vulkan);
    m.impl("isin.Scalar_Tensor_out", &isin_scalar_tensor_out_vulkan);
    m.impl("_unique", &unique_vulkan);
    m.impl("_unique2", &unique2_vulkan);
    m.impl("unique_consecutive", &unique_consecutive_vulkan);
    m.impl("unique_dim", &unique_dim_vulkan);
    m.impl("unique_dim_consecutive", &unique_dim_consecutive_vulkan);
    m.impl("repeat_interleave.Tensor", &repeat_interleave_vulkan);
}
