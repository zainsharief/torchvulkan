#include <torch/extension.h>
#include <ATen/WrapDimUtils.h>
#include <c10/core/MemoryFormat.h>
#include <algorithm>
#include <limits>
#include <tuple>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "api/ops/helpers.h"
#include "api/ops/internal.h"
#include "api/fallback.h"

namespace {

enum class ReduceOp {
    SUM = 0,
    AMAX = 1,
    AMIN = 2,
    PROD = 3
};

at::Tensor dispatch_reduce_shader(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    ReduceOp operation,
    double empty_value,
    bool has_subgroup)
{
    int64_t ndim = self.dim();
    if (ndim == 0) return self.clone();

    dim = c10::maybe_wrap_dim(dim, ndim);
    TORCH_CHECK(ndim <= MAX_DIMS, "torchvulkan [ERROR]: Tensor rank (", ndim, ") exceeds the maximum supported (", MAX_DIMS, ").");

    std::vector<int64_t> out_sizes(self.sizes().begin(), self.sizes().end());
    int64_t reduce_size = out_sizes[dim];
    out_sizes[dim] = 1;

    at::Tensor out = at::empty(out_sizes, self.options());
    uint64_t numel_out = out.numel();
    if (numel_out == 0) return keepdim ? out : out.squeeze(dim);
    if (reduce_size == 0) {
        out.fill_(empty_value);
        return keepdim ? out : out.squeeze(dim);
    }

    DeviceContext* device = get_device_context(self);
    at::ScalarType stype = self.scalar_type();
    bool dtype_needs_extended = c10::elementSize(stype) != 4;
    bool dtype_supported_subgroup =
        at::isFloatingType(stype) && stype != at::kBFloat16 &&
        c10::elementSize(stype) <= 4 &&
        (!dtype_needs_extended || device->support_subgroup_extended_types);

    bool use_subgroup =
        has_subgroup &&
        device->support_subgroup_arithmetic &&
        device->subgroup_size > 0 &&
        dtype_supported_subgroup &&
        reduce_size >= static_cast<int64_t>(device->subgroup_size) &&
        numel_out <= static_cast<uint64_t>(device->properties.limits.maxComputeWorkGroupCount[0]);

    uint32_t workgroupSizeX = use_subgroup ? device->subgroup_size : get_dtype_workgroup_size(stype, 1, device);

    static constexpr uint64_t TARGET_ITEMS = 1u << 14;
    static constexpr uint64_t MIN_CHUNK = 1024;
    
    uint64_t split = 1;
    if (numel_out < TARGET_ITEMS && static_cast<uint64_t>(reduce_size) >= 2 * MIN_CHUNK) {
        split = std::min<uint64_t>((TARGET_ITEMS + numel_out - 1) / numel_out, static_cast<uint64_t>(reduce_size) / MIN_CHUNK);
    }
    uint32_t chunk = static_cast<uint32_t>((reduce_size + split - 1) / split);
    split = (reduce_size + chunk - 1) / chunk; // every chunk non-empty

    at::Tensor partial = split == 1 ? out : at::empty({static_cast<int64_t>(numel_out), static_cast<int64_t>(split)}, self.options());

    torchvulkan::Kernel kernel = use_subgroup ? torchvulkan::Kernel::REDUCE_SUBGROUP : torchvulkan::Kernel::REDUCE;
    int32_t ndim32 = static_cast<int32_t>(ndim);

    SpecializationBuilder spd{};
    spd.push(static_cast<uint32_t>(operation))
       .push(ndim32)
       .push(workgroupSizeX);

    MetadataBuilder metadataBuilder = get_shape_metadata(out_sizes, {self.strides()});
    Metadata metadata = metadataBuilder.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(self))
       .push(get_tensor_address(partial))
       .push_metadata(&metadata)
       .push(numel_out)
       .push(static_cast<uint32_t>(self.stride(dim)))
       .push(static_cast<uint32_t>(reduce_size))
       .push(static_cast<uint32_t>(split))
       .push(chunk);

    uint64_t work_items = numel_out * split;
    uint32_t groupX = use_subgroup ? static_cast<uint32_t>(work_items) : get_group_count(work_items, workgroupSizeX);

    device->shader_manager->dispatchShader(
        ShaderKey{kernel, stype},
        spd.build(),
        pcs.build(),
        /* read = */ {self},
        /* write = */ {partial},
        groupX, 1, 1
    );

    if (split > 1) {
        at::Tensor folded = dispatch_reduce_shader(partial, 1, false, operation, empty_value, has_subgroup);
        out = folded.view(out_sizes);
    }
    return keepdim ? out : out.squeeze(dim);
}

at::Tensor cpu_reduce_fallback(
    const at::Tensor& cpu_self,
    at::OptionalIntArrayRef dims,
    bool keepdim,
    ReduceOp operation)
{
    at::IntArrayRef dim_ref = dims.has_value() ? *dims : at::IntArrayRef{};
    switch (operation) {
        case ReduceOp::SUM:  return at::sum(cpu_self, dims, keepdim);
        case ReduceOp::AMAX: return at::amax(cpu_self, dim_ref, keepdim);
        case ReduceOp::AMIN: return at::amin(cpu_self, dim_ref, keepdim);
        case ReduceOp::PROD: {
            if (!dims.has_value() || dims->size() == 0) return at::prod(cpu_self);
            std::vector<int64_t> dl;
            for (int64_t d : *dims) dl.push_back(c10::maybe_wrap_dim(d, cpu_self.dim()));
            std::sort(dl.begin(), dl.end(), std::greater<int64_t>());
            at::Tensor r = cpu_self;
            for (int64_t d : dl) r = at::prod(r, d, keepdim);
            return r;
        }
    }
    return at::sum(cpu_self, dims, keepdim);
}

at::Tensor reduce_dims_vulkan(
    const at::Tensor& self,
    at::OptionalIntArrayRef dims,
    bool keepdim,
    ReduceOp operation,
    double empty_value,
    bool has_subgroup)
{
    if (!is_dtype_supported(self, self.scalar_type()) || self.dim() > MAX_DIMS || !fits_32bit_indexing({self})) {
        TORCHVULKAN_CPU_FALLBACK("A ", self.scalar_type(), " reduction of this shape can't run on the Vulkan device. Falling back to CPU.");
        return cpu_reduce_fallback(self.cpu(), dims, keepdim, operation).to(self.device());
    }

    std::vector<int64_t> dim_list;
    if (dims.has_value() && dims->size() > 0) {
        for (int64_t d : *dims) dim_list.push_back(c10::maybe_wrap_dim(d, self.dim()));
    } else {
        for (int64_t d = 0; d < self.dim(); d++) dim_list.push_back(d);
    }

    if (dim_list.empty()) return self.clone();

    // reduce highest dims first so earlier indices stay valid when keepdim=false
    std::sort(dim_list.begin(), dim_list.end(), std::greater<int64_t>());

    at::Tensor result = self;
    for (int64_t d : dim_list) {
        result = dispatch_reduce_shader(result, d, keepdim, operation, empty_value, has_subgroup);
    }
    return result;
}

// the sum in its accumulation dtype (float32 for half and bfloat16), before it's cast to the output dtype
at::Tensor sum_accumulated(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    at::ScalarType out_dtype)
{
    at::Tensor acc = self.to(reduce_accumulate_dtype(out_dtype));
    return reduce_dims_vulkan(acc, dim, keepdim, ReduceOp::SUM, 0.0, true);
}

at::Tensor sum_dim_vulkan(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype)
{
    at::ScalarType out_dtype = reduce_compute_dtype(self, dtype);
    return sum_accumulated(self, dim, keepdim, out_dtype).to(out_dtype);
}

} // namespace

at::Tensor torchvulkan::sum_vulkan(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype)
{
    return sum_dim_vulkan(self, at::OptionalIntArrayRef(), false, dtype);
}

namespace {

at::Tensor amax_vulkan(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim)
{
    return reduce_dims_vulkan(self, dim, keepdim, ReduceOp::AMAX, -std::numeric_limits<double>::infinity(), true);
}

at::Tensor amin_vulkan(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim)
{
    return reduce_dims_vulkan(self, dim, keepdim, ReduceOp::AMIN, std::numeric_limits<double>::infinity(), true);
}

at::Tensor prod_dim_vulkan(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype)
{
    at::Tensor self_typed = self.to(reduce_compute_dtype(self, dtype));
    at::ScalarType out_dtype = self_typed.scalar_type();
    at::Tensor acc = self_typed.to(reduce_accumulate_dtype(out_dtype));

    return reduce_dims_vulkan(acc, at::IntArrayRef{dim}, keepdim, ReduceOp::PROD, 1.0, false).to(out_dtype);
}

at::Tensor prod_vulkan(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype)
{
    at::Tensor self_typed = self.to(reduce_compute_dtype(self, dtype));
    at::ScalarType out_dtype = self_typed.scalar_type();
    at::Tensor acc = self_typed.to(reduce_accumulate_dtype(out_dtype));

    return reduce_dims_vulkan(acc, at::OptionalIntArrayRef(), false, ReduceOp::PROD, 1.0, false).to(out_dtype);
}

at::Tensor mean_dim_vulkan(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype)
{
    at::ScalarType out_dtype = dtype.value_or(self.scalar_type());
    TORCH_CHECK(at::isFloatingType(out_dtype) || at::isComplexType(out_dtype),
        "mean(): could not infer output dtype. ", (dtype.has_value() ? "Optional" : "Input"),
        " dtype must be either a floating point or complex dtype. Got: ", out_dtype
    );
    if (self.dim() == 0) return self.to(out_dtype);

    int64_t count = 1;
    if (dim.has_value() && dim->size() > 0) {
        for (int64_t d : *dim) count *= self.size(c10::maybe_wrap_dim(d, self.dim()));
    } else {
        count = self.numel();
    }

    at::Tensor summed = sum_accumulated(self, dim, keepdim, out_dtype);
    return torchvulkan::divide_scalar_vulkan(summed, (double)count).to(out_dtype);
}

at::Tensor mean_vulkan(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype)
{
    return mean_dim_vulkan(self, at::OptionalIntArrayRef(), false, dtype);
}

enum class ArgOp 
{ 
    ARGMAX = 0, 
    ARGMIN = 1 
};

at::Tensor argreduce(
    const at::Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim,
    ArgOp op)
{
    at::Tensor x = self;
    bool flatten = !dim.has_value();
    int64_t d = 0;
    if (flatten) {
        x = x.reshape({-1});
        keepdim = false;
    } else {
        d = c10::maybe_wrap_dim(*dim, x.dim());
    }

    if (!is_dtype_supported(x, x.scalar_type()) || x.dim() == 0 || x.dim() > MAX_DIMS || x.numel() == 0 || !fits_32bit_indexing({x})) {
        TORCHVULKAN_CPU_FALLBACK("This ", op == ArgOp::ARGMAX ? "argmax" : "argmin", " on ", x.scalar_type(), " can't run on the Vulkan device. Falling back to CPU.");
        at::Tensor cpu = op == ArgOp::ARGMAX ? at::argmax(self.cpu(), dim, keepdim) : at::argmin(self.cpu(), dim, keepdim);
        return cpu.to(self.device());
    }

    int64_t ndim = x.dim();
    std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
    int64_t reduce_size = out_sizes[d];
    out_sizes[d] = 1;
    int64_t num_lines = reduce_size > 0 ? x.numel() / reduce_size : 0;
    at::Tensor idx = at::empty(out_sizes, x.options().dtype(at::kLong));
    if (num_lines == 0) return keepdim ? idx : idx.squeeze(d);

    DeviceContext* device = get_device_context(x);
    uint32_t workgroupSizeX = get_dtype_workgroup_size(x.scalar_type(), 1, device);
    ShaderKey shader_key{torchvulkan::Kernel::ARG_REDUCE, x.scalar_type()};
    int32_t ndim32 = static_cast<int32_t>(ndim);

    SpecializationBuilder spd{};
    spd.push(static_cast<uint32_t>(op))
       .push(ndim32)
       .push(workgroupSizeX);

    MetadataBuilder metadataBuilder = get_shape_metadata(out_sizes, {x.strides()});
    Metadata metadata = metadataBuilder.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(x))
       .push(get_tensor_address(idx))
       .push_metadata(&metadata)
       .push(static_cast<uint64_t>(num_lines))
       .push(static_cast<uint32_t>(x.stride(d)))
       .push(static_cast<uint32_t>(reduce_size));

    device->shader_manager->dispatchShader(
        shader_key,
        spd.build(),
        pcs.build(),
        /* read = */ {x},
        /* write = */ {idx},
        get_group_count(static_cast<uint64_t>(num_lines), workgroupSizeX), 1, 1
    );

    return keepdim ? idx : idx.squeeze(d);
}

at::Tensor& argmax_out_vulkan(
    const at::Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim,
    at::Tensor& out)
{
    out.copy_(argreduce(self, dim, keepdim, ArgOp::ARGMAX));
    return out;
}

at::Tensor& argmin_out_vulkan(
    const at::Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim,
    at::Tensor& out)
{
    out.copy_(argreduce(self, dim, keepdim, ArgOp::ARGMIN));
    return out;
}

// NaN counts as nonzero
at::Tensor count_nonzero_along(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim)
{
    return sum_dim_vulkan(at::ne(self, 0), dim, keepdim, at::kLong);
}

at::Tensor& any_out_vulkan(const at::Tensor& self, int64_t dim, bool keepdim, at::Tensor& out)
{
    return fill_out(out, at::gt(count_nonzero_along(self, at::IntArrayRef{dim}, keepdim), 0));
}

at::Tensor& any_all_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return fill_out(out, at::gt(count_nonzero_along(self, at::OptionalIntArrayRef(), false), 0));
}

at::Tensor& all_out_vulkan(const at::Tensor& self, int64_t dim, bool keepdim, at::Tensor& out)
{
    return fill_out(out, at::eq(sum_dim_vulkan(at::eq(self, 0), at::IntArrayRef{dim}, keepdim, at::kLong), 0));
}

at::Tensor& all_all_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return fill_out(out, at::eq(sum_dim_vulkan(at::eq(self, 0), at::OptionalIntArrayRef(), false, at::kLong), 0));
}

at::Tensor count_nonzero_vulkan(const at::Tensor& self, at::IntArrayRef dim)
{
    return count_nonzero_along(self, dim, false);
}

at::Tensor nansum_vulkan(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype)
{
    at::Tensor x = at::isFloatingType(self.scalar_type()) ? at::where(at::isnan(self), at::zeros_like(self), self) : self;
    return sum_dim_vulkan(x, dim, keepdim, dtype);
}

at::Tensor& nansum_out_vulkan(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor& out)
{
    return fill_out(out, nansum_vulkan(self, dim, keepdim, dtype));
}

at::Tensor max_vulkan(const at::Tensor& self)
{
    TORCH_CHECK(self.numel() > 0, "max(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.");
    return amax_vulkan(self, {}, false);
}

at::Tensor min_vulkan(const at::Tensor& self)
{
    TORCH_CHECK(self.numel() > 0, "min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.");
    return amin_vulkan(self, {}, false);
}

at::Tensor& amax_out_vulkan(const at::Tensor& self, at::IntArrayRef dim, bool keepdim, at::Tensor& out)
{
    return fill_out(out, amax_vulkan(self, dim, keepdim));
}

at::Tensor& amin_out_vulkan(const at::Tensor& self, at::IntArrayRef dim, bool keepdim, at::Tensor& out)
{
    return fill_out(out, amin_vulkan(self, dim, keepdim));
}

at::Tensor& sum_out_vulkan(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor& out)
{
    return fill_out(out, sum_dim_vulkan(self, dim, keepdim, dtype));
}

at::Tensor& max_unary_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return fill_out(out, max_vulkan(self));
}

at::Tensor& min_unary_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return fill_out(out, min_vulkan(self));
}

std::tuple<at::Tensor&, at::Tensor&> aminmax_out_vulkan(const at::Tensor& self, c10::optional<int64_t> dim, bool keepdim, at::Tensor& min, at::Tensor& max)
{
    if (dim.has_value()) {
        TORCH_CHECK(self.dim() == 0 || self.size(c10::maybe_wrap_dim(*dim, self.dim())) > 0,
            "aminmax(): Expected reduction dim ", *dim, " to have non-zero size."
        );
        fill_out(min, amin_vulkan(self, {*dim}, keepdim));
        fill_out(max, amax_vulkan(self, {*dim}, keepdim));
    } else {
        TORCH_CHECK(self.numel() > 0, "aminmax(): cannot compute aminmax over an empty dimension as the operation has no identity.");
        at::Tensor lo = amin_vulkan(self, {}, false);
        at::Tensor hi = amax_vulkan(self, {}, false);
        if (keepdim) {
            std::vector<int64_t> ones(self.dim(), 1);
            lo = lo.reshape(ones);
            hi = hi.reshape(ones);
        }
        fill_out(min, lo);
        fill_out(max, hi);
    }
    return std::forward_as_tuple(min, max);
}

bool equal_vulkan(const at::Tensor& self, const at::Tensor& other)
{
    TORCH_CHECK(self.device() == other.device(), "Cannot compare two tensors on different devices. Got: ", self.device(), " and ", other.device());
    if (!self.sizes().equals(other.sizes())) return false;
    if (self.numel() == 0) return true;
    return sum_dim_vulkan(at::ne(self, other), at::OptionalIntArrayRef(), false, at::kLong).item<int64_t>() == 0;
}

// the float64 bit pattern of a float32 value, built with integer ops so no float64 arithmetic is needed
at::Tensor float32_to_float64_bits(const at::Tensor& x)
{
    at::Tensor bits = x.contiguous().view(at::kInt).to(at::kLong);
    at::Tensor sign = at::bitwise_and(bits, 0x80000000LL);
    at::Tensor exponent = at::bitwise_and(at::bitwise_right_shift(bits, 23), 0xff);
    at::Tensor mantissa = at::bitwise_and(bits, 0x7fffff);

    at::Tensor normal = at::bitwise_or(at::bitwise_left_shift(at::add(exponent, 1023 - 127), 52), at::bitwise_left_shift(mantissa, 29));
    at::Tensor special = at::bitwise_or(at::bitwise_left_shift(at::full_like(bits, 0x7ff), 52), at::bitwise_left_shift(mantissa, 29));

    // a subnormal's leading bit p comes from the exponent of float(mantissa), which is exact below 2^24
    at::Tensor lead = at::sub(at::bitwise_and(at::bitwise_right_shift(mantissa.to(at::kFloat).view(at::kInt).to(at::kLong), 23), 0xff), 127);
    at::Tensor normalized = at::bitwise_and(at::bitwise_left_shift(mantissa, at::rsub(lead, 23)), 0x7fffff);
    at::Tensor subnormal = at::bitwise_or(at::bitwise_left_shift(at::add(lead, 1023 - 149), 52), at::bitwise_left_shift(normalized, 29));

    at::Tensor magnitude = at::where(at::eq(exponent, 0xff), special, normal);
    magnitude = at::where(at::logical_and(at::eq(exponent, 0), at::ne(mantissa, 0)), subnormal, magnitude);
    magnitude = at::where(at::logical_and(at::eq(exponent, 0), at::eq(mantissa, 0)), at::zeros_like(magnitude), magnitude);
    return at::bitwise_or(magnitude, at::bitwise_left_shift(sign, 32));
}

// floats hash as their float64 pattern, integers sign-extended
at::Tensor& hash_tensor_out_vulkan(const at::Tensor& self, at::IntArrayRef dim, bool keepdim, int64_t mode, at::Tensor& out)
{
    TORCH_CHECK(mode == 0, "hash_tensor only supports mode=0 (xor_sum)");
    at::ScalarType dtype = self.scalar_type();
    at::Tensor words;
    if (dtype == at::kDouble) words = self.contiguous().view(at::kLong);
    else if (at::isFloatingType(dtype)) words = float32_to_float64_bits(self.to(at::kFloat));
    else words = self.to(at::kLong);

    std::vector<int64_t> reduce_dims;
    if (self.dim() == 0) {
        // a 0-dim tensor hashes to its own word whatever dim names
    } else if (dim.empty()) {
        for (int64_t d = 0; d < self.dim(); d++) reduce_dims.push_back(d);
    } else {
        for (int64_t d : dim) reduce_dims.push_back(c10::maybe_wrap_dim(d, self.dim()));
    }
    std::vector<int64_t> keep_dims;
    std::vector<int64_t> out_shape;
    for (int64_t d = 0; d < self.dim(); d++) {
        bool reduced = std::find(reduce_dims.begin(), reduce_dims.end(), d) != reduce_dims.end();
        if (!reduced) keep_dims.push_back(d);
        if (!reduced) out_shape.push_back(self.size(d));
        else if (keepdim) out_shape.push_back(1);
    }

    std::vector<int64_t> order = keep_dims;
    order.insert(order.end(), reduce_dims.begin(), reduce_dims.end());
    int64_t kept = 1;
    for (int64_t d : keep_dims) kept *= self.size(d);
    int64_t reduced = 1;
    for (int64_t d : reduce_dims) reduced *= self.size(d);
    at::Tensor lines = words.dim() == 0 ? words.reshape({1, 1}) : words.permute(order).reshape({kept, reduced});

    // an odd length gets a zero column, xor's identity
    while (lines.size(1) > 1) {
        int64_t n = lines.size(1);
        if (n % 2 == 1) lines = at::cat({lines, at::zeros({kept, 1}, lines.options())}, 1);
        int64_t half = lines.size(1) / 2;
        lines = at::bitwise_xor(lines.narrow(1, 0, half), lines.narrow(1, half, half));
    }
    at::Tensor hashed = lines.size(1) == 0 ? at::zeros({kept}, lines.options()) : lines.reshape({kept});
    return fill_out(out, hashed.view(at::kUInt64).reshape(out_shape));
}

at::Tensor& prod_dim_out_vulkan(const at::Tensor& self, int64_t dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor& out)
{
    return fill_out(out, prod_dim_vulkan(self, dim, keepdim, dtype));
}

at::Tensor& mean_out_vulkan(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor& out)
{
    return fill_out(out, mean_dim_vulkan(self, dim, keepdim, dtype));
}

std::tuple<at::Tensor, at::Tensor> aminmax_all_deprecated_vulkan(const at::Tensor& self)
{
    return at::aminmax(self);
}

std::tuple<at::Tensor, at::Tensor> aminmax_dim_deprecated_vulkan(const at::Tensor& self, int64_t dim, bool keepdim)
{
    return at::aminmax(self, dim, keepdim);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("sum.dim_IntList", &sum_dim_vulkan);
    m.impl("sum", &torchvulkan::sum_vulkan);
    m.impl("amax", &amax_vulkan);
    m.impl("amin", &amin_vulkan);
    m.impl("prod", &prod_vulkan);
    m.impl("prod.dim_int", &prod_dim_vulkan);
    m.impl("mean.dim", &mean_dim_vulkan);
    m.impl("mean", &mean_vulkan);
    m.impl("argmax.out", &argmax_out_vulkan);
    m.impl("argmin.out", &argmin_out_vulkan);
    m.impl("any.out", &any_out_vulkan);
    m.impl("any.all_out", &any_all_out_vulkan);
    m.impl("all.out", &all_out_vulkan);
    m.impl("all.all_out", &all_all_out_vulkan);
    m.impl("count_nonzero.dim_IntList", &count_nonzero_vulkan);
    m.impl("nansum", &nansum_vulkan);
    m.impl("nansum.out", &nansum_out_vulkan);
    m.impl("max", &max_vulkan);
    m.impl("min", &min_vulkan);
    m.impl("max.unary_out", &max_unary_out_vulkan);
    m.impl("min.unary_out", &min_unary_out_vulkan);
    m.impl("aminmax.out", &aminmax_out_vulkan);
    m.impl("prod.int_out", &prod_dim_out_vulkan);
    m.impl("mean.out", &mean_out_vulkan);
    m.impl("_aminmax", &aminmax_all_deprecated_vulkan);
    m.impl("_aminmax.dim", &aminmax_dim_deprecated_vulkan);
    m.impl("equal", &equal_vulkan);
    m.impl("hash_tensor.out", &hash_tensor_out_vulkan);
    m.impl("amax.out", &amax_out_vulkan);
    m.impl("amin.out", &amin_out_vulkan);
    m.impl("sum.IntList_out", &sum_out_vulkan);
}
