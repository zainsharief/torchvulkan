#include <torch/extension.h>
#include <ATen/ExpandUtils.h>
#include <ATen/WrapDimUtils.h>
#include <string>
#include <tuple>
#include <vector>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "api/ops/helpers.h"
#include "api/fallback.h"

namespace {

enum class ScatterMode {
    DIRECT = 0,
    SUM = 1,
    PROD = 2,
    MEAN = 3,
    AMAX = 4,
    AMIN = 5,
    LAST = 6
};

ScatterMode scatter_mode_from_name(c10::string_view reduce)
{
    if (reduce == "sum" || reduce == "add") return ScatterMode::SUM;
    if (reduce == "prod" || reduce == "multiply") return ScatterMode::PROD;
    if (reduce == "mean") return ScatterMode::MEAN;
    if (reduce == "amax") return ScatterMode::AMAX;
    if (reduce == "amin") return ScatterMode::AMIN;
    TORCH_CHECK(false, "reduce argument must be either sum, prod, mean, amax or amin, got ", reduce);
}

void dispatch_scatter_shader(
    const at::Tensor& out,
    const at::Tensor& self,
    const at::Tensor& src,
    const at::Tensor& keys,
    const at::Tensor& order,
    ScatterMode mode,
    bool include_self,
    bool round_bf16 = false)
{
    // an empty output has no buffer to address
    uint64_t count = keys.numel();
    if (count == 0 || out.numel() == 0) return;

    DeviceContext* device = get_device_context(out);
    at::ScalarType dtype = out.scalar_type();
    uint32_t workgroupSizeX = get_dtype_workgroup_size(dtype, 1, device);

    SpecializationBuilder spd{};
    spd.push(static_cast<uint32_t>(mode))
       .push(static_cast<uint32_t>(include_self))
       .push(static_cast<uint32_t>(at::isIntegralType(dtype, /*includeBool=*/true)))
       .push(static_cast<uint32_t>(round_bf16))
       .push(workgroupSizeX);

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(out))
       .push(get_tensor_address(self))
       .push(get_tensor_address(src))
       .push(get_tensor_address(keys))
       .push(order.defined() ? get_tensor_address(order) : uint64_t(0))
       .push(count);

    std::vector<at::Tensor> reads = {self, src, keys};
    if (order.defined()) reads.push_back(order);

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::SCATTER, dtype},
        spd.build(),
        pcs.build(),
        /* read = */ reads,
        /* write = */ {out},
        get_group_count(count, workgroupSizeX), 1, 1
    );
}

bool can_scatter(const at::Tensor& self, int64_t count)
{
    return count < std::numeric_limits<int32_t>::max();
}

bool is_copy(ScatterMode mode)
{
    return mode == ScatterMode::DIRECT || mode == ScatterMode::LAST;
}

// bfloat16 reductions run in float32: MoltenVK can't build its constants
at::ScalarType scatter_kernel_dtype(const at::Tensor& self, ScatterMode mode)
{
    at::ScalarType dtype = self.scalar_type();
    if (is_copy(mode)) {
        switch (c10::elementSize(dtype)) {
            case 1: return at::kByte;
            case 2: return at::kShort;
            case 4: return at::kInt;
            default: return at::kLong;
        }
    }
    if (dtype == at::kBFloat16 || !is_dtype_supported(self, dtype)) return at::kFloat;
    return dtype;
}

at::Tensor as_kernel_dtype(const at::Tensor& t, at::ScalarType kernel, ScatterMode mode)
{
    if (t.scalar_type() == kernel) return t;
    return is_copy(mode) ? t.view(kernel) : t.to(kernel);
}

at::Tensor scatter_into(const at::Tensor& self, const at::Tensor& keys, const at::Tensor& src, ScatterMode mode, bool include_self)
{
    at::ScalarType dtype = self.scalar_type();
    at::ScalarType kernel = scatter_kernel_dtype(self, mode);
    at::Tensor base = as_kernel_dtype(self.contiguous(), kernel, mode);
    at::Tensor result = base.clone();
    at::Tensor flat_keys = keys.reshape({-1}).to(at::kLong).contiguous();
    at::Tensor values = as_kernel_dtype(src.to(dtype).reshape({-1}).contiguous(), kernel, mode);
    if (flat_keys.numel() > 0) {
        bool round_bf16 = dtype == at::kBFloat16 && kernel == at::kFloat;
        if (mode == ScatterMode::DIRECT) {
            dispatch_scatter_shader(result, base, values, flat_keys, at::Tensor(), mode, include_self);
        } else {
            auto [sorted, order] = at::sort(flat_keys, /*stable=*/true, /*dim=*/0, /*descending=*/false);
            dispatch_scatter_shader(result, base, values, sorted.contiguous(), order.contiguous(), mode, include_self, round_bf16);
        }
    }
    if (kernel == dtype) return result;
    return is_copy(mode) ? result.view(dtype) : result.to(dtype);
}

at::Tensor scatter_keys(const at::Tensor& index, int64_t dim, c10::IntArrayRef shape)
{
    auto strides = c10::contiguous_strides(shape);
    at::Tensor keys = at::mul(index.to(at::kLong), strides[dim]);
    for (int64_t d = 0; d < index.dim(); d++) {
        if (d == dim) continue;
        std::vector<int64_t> view_shape(index.dim(), 1);
        view_shape[d] = index.size(d);
        keys = at::add(keys, at::mul(at::arange(index.size(d), index.options().dtype(at::kLong)).view(view_shape), strides[d]));
    }
    return keys;
}

void check_in_range(const at::Tensor& index, int64_t size, const char* message)
{
    if (index.numel() == 0) return;
    bool out_of_range = at::logical_or(at::lt(index, 0), at::ge(index, size)).any().item<bool>();
    TORCH_CHECK_INDEX(!out_of_range, message);
}

at::Tensor wrap_negative(const at::Tensor& index, int64_t size)
{
    return at::where(at::lt(index, 0), at::add(index, size), index);
}

at::Tensor scatter_along_dim(const at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Tensor& src, ScatterMode mode, bool include_self)
{
    TORCH_CHECK_INDEX(at::isIntegralType(index.scalar_type(), /*includeBool=*/false), "scatter(): Expected dtype int64 for index");
    int64_t ndim = std::max<int64_t>(self.dim(), 1);
    dim = c10::maybe_wrap_dim(dim, ndim);
    if (index.numel() == 0) return self.clone();
    TORCH_CHECK(std::max<int64_t>(index.dim(), 1) == ndim, "Index tensor must have the same number of dimensions as self tensor");

    at::Tensor s = self.dim() == 0 ? self.reshape({1}) : self;
    at::Tensor idx = index.dim() == 0 ? index.reshape({1}) : index;
    for (int64_t d = 0; d < ndim; d++) {
        if (d != dim) TORCH_CHECK(idx.size(d) <= s.size(d), "Expected index ", idx.sizes(), " to be no larger than self ", s.sizes(), " apart from dimension ", dim);
    }
    std::string message = "index is out of bounds for dimension " + std::to_string(dim) + " with size " + std::to_string(s.size(dim));
    check_in_range(idx, s.size(dim), message.c_str());

    at::Tensor values;
    if (src.dim() == 0) values = src.expand(idx.sizes());
    else {
        at::Tensor src_nd = src.dim() == 0 ? src.reshape({1}) : src;
        for (int64_t d = 0; d < ndim; d++) {
            TORCH_CHECK(idx.size(d) <= src_nd.size(d), "Expected index ", idx.sizes(), " to be no larger than src ", src_nd.sizes());
        }
        values = src_nd.as_strided(idx.sizes(), src_nd.strides(), src_nd.storage_offset());
    }

    if (!can_scatter(s, idx.numel())) {
        TORCHVULKAN_CPU_FALLBACK("This scatter can't run on the Vulkan device. Falling back to CPU.");
        at::Tensor cpu = s.cpu().clone();
        if (mode == ScatterMode::LAST) cpu.scatter_(dim, idx.cpu(), values.cpu());
        else cpu.scatter_reduce_(dim, idx.cpu(), values.cpu(), mode == ScatterMode::SUM ? "sum" : mode == ScatterMode::PROD ? "prod" : mode == ScatterMode::MEAN ? "mean" : mode == ScatterMode::AMAX ? "amax" : "amin", include_self);
        return cpu.to(self.device()).view(self.sizes());
    }

    at::Tensor result = scatter_into(s, scatter_keys(idx, dim, s.sizes()), values.contiguous(), mode, include_self);
    return result.view(self.sizes());
}

at::Tensor value_tensor(const at::Scalar& value, const at::Tensor& like)
{
    return at::scalar_tensor(value, like.options());
}

at::Tensor& scatter_src_out_vulkan(const at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Tensor& src, at::Tensor& out)
{
    return fill_out(out, scatter_along_dim(self, dim, index, src, ScatterMode::LAST, true));
}

at::Tensor& scatter_value_out_vulkan(const at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Scalar& value, at::Tensor& out)
{
    return fill_out(out, scatter_along_dim(self, dim, index, value_tensor(value, self), ScatterMode::LAST, true));
}

at::Tensor& scatter_reduce_out_vulkan(const at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Tensor& src, c10::string_view reduce, at::Tensor& out)
{
    return fill_out(out, scatter_along_dim(self, dim, index, src, scatter_mode_from_name(reduce), true));
}

at::Tensor& scatter_value_reduce_out_vulkan(const at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Scalar& value, c10::string_view reduce, at::Tensor& out)
{
    return fill_out(out, scatter_along_dim(self, dim, index, value_tensor(value, self), scatter_mode_from_name(reduce), true));
}

at::Tensor& scatter_add_out_vulkan(const at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Tensor& src, at::Tensor& out)
{
    return fill_out(out, scatter_along_dim(self, dim, index, src, ScatterMode::SUM, true));
}

at::Tensor& scatter_reduce_two_out_vulkan(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src,
    c10::string_view reduce,
    bool include_self,
    at::Tensor& out)
{
    return fill_out(out, scatter_along_dim(self, dim, index, src, scatter_mode_from_name(reduce), include_self));
}

at::Tensor index_along_dim(const at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Tensor& source, ScatterMode mode, bool include_self, bool wrap)
{
    TORCH_CHECK_INDEX(index.dim() <= 1, "index_add_(): Index is supposed to be a vector");
    TORCH_CHECK(index.scalar_type() == at::kLong || index.scalar_type() == at::kInt, "index_add_(): Expected dtype int32/int64 for index");
    int64_t ndim = std::max<int64_t>(self.dim(), 1);
    dim = c10::maybe_wrap_dim(dim, ndim);
    at::Tensor s = self.dim() == 0 ? self.reshape({1}) : self;
    at::Tensor idx = index.reshape({-1}).to(at::kLong);
    if (idx.numel() == 0) return self.clone();

    if (wrap) idx = wrap_negative(idx, s.size(dim));
    check_in_range(idx, s.size(dim), "index out of range in self");

    std::vector<int64_t> slice_shape(s.sizes().begin(), s.sizes().end());
    slice_shape[dim] = idx.numel();
    std::vector<int64_t> index_view(ndim, 1);
    index_view[dim] = idx.numel();
    at::Tensor expanded = idx.view(index_view).expand(slice_shape);

    at::Tensor values = source.dim() == 0 && source.numel() == 1 ? source.expand(slice_shape) : source.reshape(slice_shape);
    at::Tensor result = scatter_into(s, scatter_keys(expanded, dim, s.sizes()), values.contiguous(), mode, include_self);
    return result.view(self.sizes());
}

at::Tensor& index_add_out_vulkan(const at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Tensor& source, const at::Scalar& alpha, at::Tensor& out)
{
    at::Tensor scaled = alpha.toDouble() == 1.0 ? source : at::mul(source, alpha);
    return fill_out(out, index_along_dim(self, dim, index, scaled, ScatterMode::SUM, true, false));
}

at::Tensor& index_copy_out_vulkan(const at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Tensor& source, at::Tensor& out)
{
    return fill_out(out, index_along_dim(self, dim, index, source, ScatterMode::LAST, true, false));
}

at::Tensor& index_reduce_out_vulkan(const at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Tensor& source, c10::string_view reduce, bool include_self, at::Tensor& out)
{
    TORCH_CHECK(reduce == "prod" || reduce == "mean" || reduce == "amax" || reduce == "amin",
        "index_reduce(): Expected reduce to be one of prod, mean, amax or amin but got ", reduce, "."
    );
    return fill_out(out, index_along_dim(self, dim, index, source, scatter_mode_from_name(reduce), include_self, false));
}

at::Tensor& index_fill_scalar_vulkan_(at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Scalar& value)
{
    return self.copy_(index_along_dim(self, dim, index, value_tensor(value, self), ScatterMode::LAST, true, true));
}

at::Tensor& index_fill_tensor_vulkan_(at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Tensor& value)
{
    TORCH_CHECK(value.dim() == 0, "index_fill_ only supports a 0-dimensional value tensor, but got tensor with ", value.dim(), " dimension(s).");
    return index_fill_scalar_vulkan_(self, dim, index, value.item());
}

at::Tensor element_offsets(const at::Tensor& self)
{
    return at::arange(self.numel(), self.options().dtype(at::kLong)).view(self.sizes());
}

at::Tensor& index_put_impl_vulkan_(at::Tensor& self, const c10::List<std::optional<at::Tensor>>& indices, const at::Tensor& values, bool accumulate, bool /* unsafe */)
{
    for (size_t i = 0; i < indices.size(); i++) {
        const std::optional<at::Tensor>& index = indices.get(i);
        if (!index.has_value() || !index->defined() || index->scalar_type() == at::kBool || index->scalar_type() == at::kByte) continue;
        int64_t size = self.size(static_cast<int64_t>(i));
        bool out_of_range = index->numel() > 0 && at::logical_or(at::lt(*index, -size), at::ge(*index, size)).any().item<bool>();
        TORCH_CHECK_INDEX(!out_of_range, "index out of bounds for dimension ", i, " with size ", size);
    }

    at::Tensor keys = at::index(element_offsets(self), indices);
    if (keys.numel() == 0) return self;
    TORCH_CHECK(at::is_expandable_to(values.sizes(), keys.sizes()), "shape mismatch: value tensor of shape ", values.sizes(),
        " cannot be broadcast to indexing result of shape ", keys.sizes()
    );
    at::Tensor expanded = values.to(self.device(), self.scalar_type()).expand(keys.sizes());
    return self.copy_(scatter_into(self, keys, expanded.contiguous(), accumulate ? ScatterMode::SUM : ScatterMode::LAST, true).view(self.sizes()));
}

at::Tensor& put_vulkan_(at::Tensor& self, const at::Tensor& index, const at::Tensor& source, bool accumulate)
{
    TORCH_CHECK_INDEX(index.scalar_type() == at::kLong, "put_(): Expected a long tensor for index, but got ", index.scalar_type());
    TORCH_CHECK(index.numel() == source.numel(), "put_(): Expected source and index to have the same number of elements, but got source.numel() = ",
        source.numel(), ", index.numel() = ", index.numel()
    );
    if (index.numel() == 0) return self;
    int64_t n = self.numel();
    TORCH_CHECK_INDEX(n > 0, "put_(): Tried to put elements into an empty tensor");
    at::Tensor idx = wrap_negative(index.reshape({-1}), n);
    check_in_range(idx, n, "put_(): index out of range");
    return self.copy_(scatter_into(self, idx, source.reshape({-1}), accumulate ? ScatterMode::SUM : ScatterMode::LAST, true).view(self.sizes()));
}

at::Tensor take_vulkan(const at::Tensor& self, const at::Tensor& index)
{
    TORCH_CHECK_INDEX(index.scalar_type() == at::kLong, "take(): Expected a long tensor for index, but got ", index.scalar_type());
    int64_t n = self.numel();
    if (index.numel() == 0) return at::empty(index.sizes(), self.options());
    TORCH_CHECK_INDEX(n > 0, "take(): tried to take from an empty tensor");
    at::Tensor idx = wrap_negative(index, n);
    check_in_range(idx, n, "take(): index out of range");
    return at::index_select(self.reshape({-1}), 0, idx.reshape({-1})).view(index.sizes());
}

at::Tensor& take_out_vulkan(const at::Tensor& self, const at::Tensor& index, at::Tensor& out)
{
    return fill_out(out, take_vulkan(self, index));
}

std::tuple<at::Tensor, int64_t> compaction_keys(const at::Tensor& mask)
{
    at::Tensor flat = mask.reshape({-1});
    at::Tensor running = at::sub(at::cumsum(flat, 0, at::kLong), 1);
    int64_t count = flat.numel() == 0 ? 0 : (running[-1].item<int64_t>() + 1);
    return std::make_tuple(at::where(flat, running, at::full_like(running, -1)), count);
}

at::Tensor masked_select_vulkan(const at::Tensor& self, const at::Tensor& mask)
{
    TORCH_CHECK(mask.scalar_type() == at::kBool || mask.scalar_type() == at::kByte, "masked_select: expected BoolTensor or ByteTensor for mask");
    std::vector<int64_t> shape = at::infer_size(self.sizes(), mask.sizes());
    at::Tensor values = self.expand(shape).contiguous();
    auto [keys, count] = compaction_keys(mask.to(self.device(), at::kBool).expand(shape));
    return scatter_into(at::empty({count}, self.options()), keys, values, ScatterMode::DIRECT, false);
}

at::Tensor& masked_select_out_vulkan(const at::Tensor& self, const at::Tensor& mask, at::Tensor& out)
{
    return fill_out(out, masked_select_vulkan(self, mask));
}

at::Tensor& masked_scatter_vulkan_(at::Tensor& self, const at::Tensor& mask, const at::Tensor& source)
{
    TORCH_CHECK(mask.scalar_type() == at::kBool, "masked_scatter_ only supports boolean masks, but got mask with dtype ", mask.scalar_type());
    TORCH_CHECK(self.scalar_type() == source.scalar_type(), "masked_scatter: expected self and source to have same dtypes but got ", self.scalar_type(), " and ", source.scalar_type());
    at::Tensor m = mask.to(self.device()).expand(self.sizes()).reshape({-1});
    if (m.numel() == 0) return self;
    at::Tensor position = at::sub(at::cumsum(m, 0, at::kLong), 1);
    int64_t needed = position[-1].item<int64_t>() + 1;
    TORCH_CHECK(needed <= source.numel(), "Number of elements of source < number of ones in mask");
    if (needed == 0) return self;
    at::Tensor picked = at::index_select(source.reshape({-1}), 0, at::clamp_min(position, 0));
    return self.copy_(at::where(m, picked, self.reshape({-1})).view(self.sizes()));
}

at::Tensor nonzero_vulkan(const at::Tensor& self)
{
    int64_t ndim = self.dim();
    auto [keys, count] = compaction_keys(at::ne(self, 0));
    at::Tensor positions = at::arange(self.numel(), self.options().dtype(at::kLong));
    at::Tensor linear = scatter_into(at::empty({count}, positions.options()), keys, positions, ScatterMode::DIRECT, false);
    if (count == 0) return at::empty({0, ndim}, linear.options());

    std::vector<at::Tensor> coords(ndim);
    at::Tensor rest = linear;
    for (int64_t d = ndim - 1; d >= 0; d--) {
        coords[d] = at::remainder(rest, self.size(d));
        rest = at::floor_divide(rest, self.size(d));
    }
    if (ndim == 0) return at::empty({count, 0}, self.options().dtype(at::kLong));
    return at::stack(coords, 1);
}

at::Tensor& nonzero_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return fill_out(out, nonzero_vulkan(self));
}

at::Tensor nonzero_static_vulkan(const at::Tensor& self, c10::SymInt size_, int64_t fill_value)
{
    int64_t size = size_.guard_int(__FILE__, __LINE__);
    TORCH_CHECK(size >= 0, "nonzero_static: 'size' must be an non-negative integer");
    at::Tensor found = nonzero_vulkan(self);
    int64_t count = found.size(0);
    if (count >= size) return found.narrow(0, 0, size).contiguous();
    at::Tensor padding = at::full({size - count, self.dim()}, fill_value, found.options());
    return at::cat({found, padding}, 0);
}

at::Tensor& nonzero_static_out_vulkan(const at::Tensor& self, c10::SymInt size, int64_t fill_value, at::Tensor& out)
{
    return fill_out(out, nonzero_static_vulkan(self, size, fill_value));
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("scatter.src_out", &scatter_src_out_vulkan);
    m.impl("scatter.value_out", &scatter_value_out_vulkan);
    m.impl("scatter.reduce_out", &scatter_reduce_out_vulkan);
    m.impl("scatter.value_reduce_out", &scatter_value_reduce_out_vulkan);
    m.impl("scatter_add.out", &scatter_add_out_vulkan);
    m.impl("scatter_reduce.two_out", &scatter_reduce_two_out_vulkan);
    m.impl("index_add.out", &index_add_out_vulkan);
    m.impl("index_copy.out", &index_copy_out_vulkan);
    m.impl("index_reduce.out", &index_reduce_out_vulkan);
    m.impl("index_fill_.int_Scalar", &index_fill_scalar_vulkan_);
    m.impl("index_fill_.int_Tensor", &index_fill_tensor_vulkan_);
    m.impl("_index_put_impl_", &index_put_impl_vulkan_);
    m.impl("put_", &put_vulkan_);
    m.impl("take", &take_vulkan);
    m.impl("take.out", &take_out_vulkan);
    m.impl("masked_select", &masked_select_vulkan);
    m.impl("masked_select.out", &masked_select_out_vulkan);
    m.impl("masked_scatter_", &masked_scatter_vulkan_);
    m.impl("nonzero", &nonzero_vulkan);
    m.impl("nonzero.out", &nonzero_out_vulkan);
    m.impl("nonzero_static", &nonzero_static_vulkan);
    m.impl("nonzero_static.out", &nonzero_static_out_vulkan);
}
