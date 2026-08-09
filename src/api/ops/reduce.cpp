#include <torch/extension.h>
#include <ATen/WrapDimUtils.h>
#include <limits>
#include "api/ops/reduce.h"
#include "api/ops/binary.h"

at::Tensor torchvulkan::dispatch_reduce_shader(
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
    int64_t reduce_ndim = std::max<int64_t>(ndim, 1);

    TORCH_CHECK(reduce_ndim <= MAX_DIMS, "torchvulkan [ERROR]: Coalesced dimensions (", reduce_ndim, ") exceed maximum supported (", MAX_DIMS, ").");

    std::vector<int64_t> self_sizes(self.sizes().begin(), self.sizes().end());
    std::vector<int64_t> self_strides(self.strides().begin(), self.strides().end());

    std::vector<int64_t> out_sizes = self_sizes;
    int64_t reduce_size = out_sizes[dim];
    out_sizes[dim] = 1;

    at::Tensor out = at::empty(out_sizes, self.options());
    uint64_t numel_out = out.numel();
    if (numel_out == 0) return keepdim ? out : out.squeeze(dim);
    if (reduce_size == 0) {
        out.fill_(empty_value);
        return keepdim ? out : out.squeeze(dim);
    }

    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    uint32_t op = static_cast<uint32_t>(operation);
    int32_t ndim32 = static_cast<int32_t>(reduce_ndim);

    at::ScalarType stype = self.scalar_type();
    bool dtype_needs_extended = c10::elementSize(stype) != 4;
    bool dtype_supported_subgroup =
        at::isFloatingType(stype) && c10::elementSize(stype) <= 4 &&
        (!dtype_needs_extended || device->support_subgroup_extended_types);

    bool use_subgroup =
        has_subgroup &&
        device->support_subgroup_arithmetic &&
        device->subgroup_size > 0 &&
        dtype_supported_subgroup &&
        reduce_size >= static_cast<int64_t>(device->subgroup_size) &&
        numel_out <= static_cast<uint64_t>(device->properties.limits.maxComputeWorkGroupCount[0]);

    uint32_t workgroupSizeX = use_subgroup
        ? device->subgroup_size
        : get_dtype_workgroup_size(self.scalar_type(), 1);
    torchvulkan::ShaderID shader_id = use_subgroup
        ? torchvulkan::get_shader_id_reduce_subgroup(self.scalar_type())
        : torchvulkan::get_shader_id_reduce(self.scalar_type());

    SpecializationBuilder spd{};
    spd.push(op)
       .push(ndim32)
       .push(workgroupSizeX);
    uint32_t key = (workgroupSizeX << 8) | (ndim32 << 4) | op;
    SpecializationArgs specialization = {spd.data(), spd.offsets(), spd.sizes(), spd.numConstants(), key};

    IntDivider sizes;
    uint32_t strides_in[MAX_DIMS] = {0};
    for (int64_t i = 0; i < reduce_ndim; i++) {
        int64_t actual_dim = reduce_ndim - 1 - i;
        sizes.set(i, static_cast<uint32_t>(out_sizes[actual_dim]));
        strides_in[i] = static_cast<uint32_t>(self_strides[actual_dim]);
    }

    PushConstantBuilder pcs{};
    pcs.push(sizes)
       .push_array(strides_in)
       .push(static_cast<uint32_t>(self_strides[dim]))
       .push(static_cast<uint32_t>(reduce_size))
       .push(numel_out);

    uint32_t groupX = use_subgroup
        ? static_cast<uint32_t>(numel_out)
        : static_cast<uint32_t>((numel_out + (workgroupSizeX - 1)) / workgroupSizeX);

    VulkanShader shader(shader_id, specialization, device);
    shader.dispatch(
        &pcs,
        pcs.size(),
        {self, out},
        groupX, 1, 1
    );

    return keepdim ? out : out.squeeze(dim);
}

namespace {

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

// integer reductions accumulate in int64 to avoid overflow
c10::ScalarType reduce_compute_dtype(const at::Tensor& self, c10::optional<at::ScalarType> dtype)
{
    if (dtype.has_value()) return *dtype;
    if (at::isIntegralType(self.scalar_type(), /*includeBool=*/true) && self.scalar_type() != at::kLong) {
        return at::kLong;
    }
    return self.scalar_type();
}

} // namespace

at::Tensor torchvulkan::reduce_dims_vulkan(
    const at::Tensor& self,
    at::OptionalIntArrayRef dims,
    bool keepdim,
    ReduceOp operation,
    double empty_value,
    bool has_subgroup)
{
    if (!is_dtype_supported(self.scalar_type())) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Vulkan device does not support ", self.scalar_type(), ". Falling back to CPU.");
        return cpu_reduce_fallback(self.cpu(), dims, keepdim, operation).to(self.device());
    }

    if (self.dim() > MAX_DIMS) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Tensor rank (", self.dim(), ") exceed maximum supported (", MAX_DIMS, "). Falling back to CPU.");
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

at::Tensor torchvulkan::sum_dim_vulkan(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype)
{
    at::Tensor self_typed = self.to(reduce_compute_dtype(self, dtype));
    return reduce_dims_vulkan(self_typed, dim, keepdim, ReduceOp::SUM, 0.0, true);
}

at::Tensor torchvulkan::sum_vulkan(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype)
{
    return sum_dim_vulkan(self, at::OptionalIntArrayRef(), false, dtype);
}

at::Tensor torchvulkan::amax_vulkan(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim)
{
    return reduce_dims_vulkan(self, dim, keepdim, ReduceOp::AMAX, -std::numeric_limits<double>::infinity(), true);
}

at::Tensor torchvulkan::amin_vulkan(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim)
{
    return reduce_dims_vulkan(self, dim, keepdim, ReduceOp::AMIN, std::numeric_limits<double>::infinity(), true);
}

at::Tensor torchvulkan::prod_dim_vulkan(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype)
{
    at::Tensor self_typed = self.to(reduce_compute_dtype(self, dtype));
    return reduce_dims_vulkan(self_typed, at::IntArrayRef{dim}, keepdim, ReduceOp::PROD, 1.0, false);
}

at::Tensor torchvulkan::prod_vulkan(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype)
{
    at::Tensor self_typed = self.to(reduce_compute_dtype(self, dtype));
    return reduce_dims_vulkan(self_typed, at::OptionalIntArrayRef(), false, ReduceOp::PROD, 1.0, false);
}

at::Tensor torchvulkan::mean_dim_vulkan(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype)
{
    if (self.dim() == 0) return dtype.has_value() ? self.to(*dtype) : self.clone();

    at::Tensor summed = sum_dim_vulkan(self, dim, keepdim, dtype);

    int64_t count = 1;
    if (dim.has_value() && dim->size() > 0) {
        for (int64_t d : *dim) count *= self.size(c10::maybe_wrap_dim(d, self.dim()));
    } else {
        count = self.numel();
    }

    return divide_scalar_vulkan(summed, (double)count);
}

at::Tensor torchvulkan::mean_vulkan(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype)
{
    return mean_dim_vulkan(self, at::OptionalIntArrayRef(), false, dtype);
}

at::Tensor torchvulkan::logsumexp_vulkan(
    const at::Tensor& self_in,
    at::IntArrayRef dim,
    bool keepdim)
{
    at::Tensor self = self_in.is_floating_point()
        ? self_in
        : self_in.to(c10::typeMetaToScalarType(at::get_default_dtype()));

    if (self.numel() == 0) return at::sum(at::exp(self), dim, keepdim).log_();

    at::Tensor m = at::amax(self, dim, /*keepdim=*/true);
    at::Tensor s = at::sum(at::exp(at::sub(self, m)), dim, /*keepdim=*/true);
    at::Tensor res = at::add(at::log(s), m);
    res = at::where(at::isinf(m), m, res);

    if (!keepdim) {
        std::vector<int64_t> dl(dim.begin(), dim.end());
        for (auto& d : dl) d = c10::maybe_wrap_dim(d, self.dim());
        std::sort(dl.begin(), dl.end(), std::greater<int64_t>());
        for (int64_t d : dl) res = res.squeeze(d);
    }
    return res;
}

namespace {

std::tuple<at::Tensor, at::Tensor> compute_var_mean(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    double correction,
    bool keepdim)
{
    at::Tensor mean_keep = at::mean(self, dim, /*keepdim=*/true);
    at::Tensor diff = at::sub(self, mean_keep);
    at::Tensor ssum = at::sum(at::mul(diff, diff), dim, keepdim);

    int64_t N = 1;
    if (dim.has_value() && dim->size() > 0) {
        for (int64_t d : *dim) N *= self.size(c10::maybe_wrap_dim(d, self.dim()));
    } else {
        N = self.numel();
    }

    double denom = static_cast<double>(N) - correction;
    at::Tensor var = denom > 0.0
        ? at::div(ssum, denom)
        : at::full_like(ssum, std::numeric_limits<double>::quiet_NaN());
    at::Tensor mean = keepdim ? mean_keep : at::mean(self, dim, /*keepdim=*/false);
    return std::make_tuple(var, mean);
}

double correction_or_default(const c10::optional<at::Scalar>& correction)
{
    return correction.has_value() ? correction->toDouble() : 1.0;
}

} // namespace

at::Tensor torchvulkan::var_correction_vulkan(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<at::Scalar>& correction,
    bool keepdim)
{
    return std::get<0>(compute_var_mean(self, dim, correction_or_default(correction), keepdim));
}

at::Tensor torchvulkan::std_correction_vulkan(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<at::Scalar>& correction,
    bool keepdim)
{
    return at::sqrt(var_correction_vulkan(self, dim, correction, keepdim));
}

std::tuple<at::Tensor, at::Tensor> torchvulkan::var_mean_correction_vulkan(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<at::Scalar>& correction,
    bool keepdim)
{
    return compute_var_mean(self, dim, correction_or_default(correction), keepdim);
}

std::tuple<at::Tensor, at::Tensor> torchvulkan::std_mean_correction_vulkan(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<at::Scalar>& correction,
    bool keepdim)
{
    at::Tensor var, mean;
    std::tie(var, mean) = compute_var_mean(self, dim, correction_or_default(correction), keepdim);
    return std::make_tuple(at::sqrt(var), mean);
}

namespace {

enum class ScanOp { SUM = 0, PROD = 1 };

at::Tensor cumscan(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype,
    ScanOp op)
{
    c10::ScalarType compute = reduce_compute_dtype(self, dtype);
    at::Tensor x = self.to(compute);
    if (x.dim() == 0 || x.numel() == 0) return x.clone();

    int64_t d = c10::maybe_wrap_dim(dim, x.dim());
    if (!is_dtype_supported(compute) || x.dim() > MAX_DIMS) {
        at::Tensor cpu = op == ScanOp::SUM ? at::cumsum(x.cpu(), dim) : at::cumprod(x.cpu(), dim);
        return cpu.to(self.device());
    }

    at::Tensor xt = x.transpose(d, -1).contiguous();
    int64_t scan_size = xt.size(-1);
    int64_t num_lines = scan_size > 0 ? xt.numel() / scan_size : 0;
    at::Tensor out_t = at::empty_like(xt);
    if (num_lines == 0) return out_t.transpose(d, -1).contiguous();

    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    uint32_t workgroupSizeX = get_dtype_workgroup_size(compute, 1);
    torchvulkan::ShaderID shader_id = torchvulkan::get_shader_id_scan(compute);
    uint32_t opv = static_cast<uint32_t>(op);

    SpecializationBuilder spd{};
    spd.push(opv)
       .push(workgroupSizeX);
    uint32_t key = (workgroupSizeX << 4) | opv;
    SpecializationArgs specialization = {spd.data(), spd.offsets(), spd.sizes(), spd.numConstants(), key};

    PushConstantBuilder pcs{};
    pcs.push(static_cast<uint64_t>(num_lines))
       .push(static_cast<uint32_t>(scan_size));

    uint32_t groupX = (static_cast<uint32_t>(num_lines) + (workgroupSizeX - 1)) / workgroupSizeX;

    VulkanShader shader(shader_id, specialization, device);
    shader.dispatch(
        &pcs,
        pcs.size(),
        {xt, out_t},
        groupX, 1, 1
    );

    return out_t.transpose(d, -1).contiguous();
}

} // namespace

at::Tensor& torchvulkan::cumsum_out_vulkan(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype,
    at::Tensor& out)
{
    out.copy_(cumscan(self, dim, dtype, ScanOp::SUM));
    return out;
}

at::Tensor& torchvulkan::cumprod_out_vulkan(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype,
    at::Tensor& out)
{
    out.copy_(cumscan(self, dim, dtype, ScanOp::PROD));
    return out;
}

namespace {

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

    if (!is_dtype_supported(x.scalar_type()) || x.dim() == 0 || x.dim() > MAX_DIMS || x.numel() == 0) {
        at::Tensor cpu = op == ArgOp::ARGMAX ? at::argmax(self.cpu(), dim, keepdim) : at::argmin(self.cpu(), dim, keepdim);
        return cpu.to(self.device());
    }

    std::vector<int64_t> perm;
    for (int64_t i = 0; i < x.dim(); i++) if (i != d) perm.push_back(i);
    perm.push_back(d);
    at::Tensor xt = x.permute(perm).contiguous();

    int64_t reduce_size = xt.size(-1);
    int64_t num_lines = reduce_size > 0 ? xt.numel() / reduce_size : 0;
    at::Tensor idx = at::empty({num_lines}, self.options().dtype(at::kLong));

    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    uint32_t workgroupSizeX = get_dtype_workgroup_size(x.scalar_type(), 1);
    torchvulkan::ShaderID shader_id = torchvulkan::get_shader_id_arg_reduce(x.scalar_type());
    uint32_t opv = static_cast<uint32_t>(op);

    SpecializationBuilder spd{};
    spd.push(opv)
       .push(workgroupSizeX);
    uint32_t key = (workgroupSizeX << 4) | opv;
    SpecializationArgs specialization = {spd.data(), spd.offsets(), spd.sizes(), spd.numConstants(), key};

    PushConstantBuilder pcs{};
    pcs.push(static_cast<uint64_t>(num_lines))
       .push(static_cast<uint32_t>(reduce_size));

    uint32_t groupX = (static_cast<uint32_t>(num_lines) + (workgroupSizeX - 1)) / workgroupSizeX;

    VulkanShader shader(shader_id, specialization, device);
    shader.dispatch(
        &pcs,
        pcs.size(),
        {xt, idx},
        groupX, 1, 1
    );

    std::vector<int64_t> oshape(xt.sizes().begin(), xt.sizes().end() - 1);
    at::Tensor result = idx.reshape(oshape);
    if (keepdim) result = result.unsqueeze(d);
    return result;
}

} // namespace

at::Tensor& torchvulkan::argmax_out_vulkan(
    const at::Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim,
    at::Tensor& out)
{
    out.copy_(argreduce(self, dim, keepdim, ArgOp::ARGMAX));
    return out;
}

at::Tensor& torchvulkan::argmin_out_vulkan(
    const at::Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim,
    at::Tensor& out)
{
    out.copy_(argreduce(self, dim, keepdim, ArgOp::ARGMIN));
    return out;
}

namespace {

enum class CumArgOp { MAX = 0, MIN = 1 };

void cumscanarg(
    const at::Tensor& self,
    at::Tensor& values,
    at::Tensor& indices,
    int64_t dim,
    CumArgOp op)
{
    if (self.dim() == 0 || self.numel() == 0) {
        values.copy_(self);
        if (indices.numel() > 0) indices.zero_();
        return;
    }
    int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    if (!is_dtype_supported(self.scalar_type()) || self.dim() > MAX_DIMS) {
        auto r = op == CumArgOp::MAX ? at::cummax(self.cpu(), dim) : at::cummin(self.cpu(), dim);
        values.copy_(std::get<0>(r));
        indices.copy_(std::get<1>(r));
        return;
    }

    at::Tensor xt = self.transpose(d, -1).contiguous();
    int64_t scan_size = xt.size(-1);
    int64_t num_lines = scan_size > 0 ? xt.numel() / scan_size : 0;
    at::Tensor val_t = at::empty_like(xt);
    at::Tensor idx_t = at::empty(xt.sizes(), self.options().dtype(at::kLong));
    if (num_lines == 0) return;

    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    uint32_t workgroupSizeX = get_dtype_workgroup_size(self.scalar_type(), 1);
    torchvulkan::ShaderID shader_id = torchvulkan::get_shader_id_scan_arg(self.scalar_type());

    uint32_t opv = static_cast<uint32_t>(op);
    SpecializationBuilder spd{};
    spd.push(opv)
       .push(workgroupSizeX);
    uint32_t key = (workgroupSizeX << 4) | opv;
    SpecializationArgs specialization = {spd.data(), spd.offsets(), spd.sizes(), spd.numConstants(), key};

    PushConstantBuilder pcs{};
    pcs.push(static_cast<uint64_t>(num_lines))
       .push(static_cast<uint32_t>(scan_size));

    uint32_t groupX = (static_cast<uint32_t>(num_lines) + (workgroupSizeX - 1)) / workgroupSizeX;

    VulkanShader shader(shader_id, specialization, device);
    shader.dispatch(
        &pcs,
        pcs.size(),
        {xt, val_t, idx_t},
        groupX, 1, 1
    );

    values.copy_(val_t.transpose(d, -1));
    indices.copy_(idx_t.transpose(d, -1));
}

} // namespace

void torchvulkan::cummax_helper_vulkan(
    const at::Tensor& self,
    at::Tensor& values,
    at::Tensor& indices,
    int64_t dim)
{
    cumscanarg(self, values, indices, dim, CumArgOp::MAX);
}

void torchvulkan::cummin_helper_vulkan(
    const at::Tensor& self,
    at::Tensor& values,
    at::Tensor& indices,
    int64_t dim)
{
    cumscanarg(self, values, indices, dim, CumArgOp::MIN);
}
