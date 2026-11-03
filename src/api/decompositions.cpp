#include <torch/extension.h>
#include <ATen/SDPBackend.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/ops/_native_multi_head_attention_native.h>
#include <ATen/ops/_transformer_encoder_layer_fwd_native.h>
#include <array>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>
#include "api/ops/helpers.h"

namespace {

std::vector<int64_t> all_dims(const at::Tensor& t) 
{
    std::vector<int64_t> d;
    for (int64_t i = 0; i < t.dim(); i++) d.push_back(i);
    return d;
}

at::Tensor p_norm(const at::Tensor& self, double p, at::OptionalIntArrayRef dim, bool keepdim)
{
    at::ScalarType odt = self.scalar_type();
    bool low = is_low_prec(odt);
    at::Tensor a = at::abs(low ? self.to(at::kFloat) : self);
    std::vector<int64_t> dv = (dim.has_value() && dim->size() > 0)
        ? std::vector<int64_t>(dim->begin(), dim->end())
        : all_dims(self);

    at::Tensor r;
    if (p == 2.0) r = at::sqrt(at::sum(at::mul(a, a), dv, keepdim));
    else if (p == 1.0) r = at::sum(a, dv, keepdim);
    else if (std::isinf(p)) r = p > 0 ? at::amax(a, dv, keepdim) : at::amin(a, dv, keepdim);
    else if (p == 0.0) r = at::sum(at::ne(a, 0).to(a.scalar_type()), dv, keepdim);
    else r = at::pow(at::sum(at::pow(a, p), dv, keepdim), 1.0 / p);
    return low ? r.to(odt) : r;
}

at::Tensor dot_vulkan(const at::Tensor& self, const at::Tensor& other)
{
    at::ScalarType odt = self.scalar_type();
    bool low = is_low_prec(odt);
    at::Tensor a = low ? self.to(at::kFloat) : self;
    at::Tensor b = low ? other.to(at::kFloat) : other;
    return at::sum(at::mul(a, b)).to(odt);
}

at::Tensor& addmv_out_vulkan(const at::Tensor& self, const at::Tensor& mat, const at::Tensor& vec, const at::Scalar& beta, const at::Scalar& alpha, at::Tensor& out)
{
    at::ScalarType odt = self.scalar_type();
    bool low = is_low_prec(odt);
    at::Scalar a = alpha, b = beta;
    if (at::isIntegralType(odt, /*includeBool=*/true)) { a = alpha.toLong(); b = beta.toLong(); }

    at::Tensor m = low ? mat.to(at::kFloat) : mat;
    at::Tensor v = low ? vec.to(at::kFloat) : vec;
    at::Tensor res = at::mul(at::mm(m, v.unsqueeze(1)).squeeze(1), a);
    if (b.toDouble() != 0.0) res = at::add(res, at::mul(low ? self.to(at::kFloat) : self, b));
    return fill_out(out, res.to(odt));
}

at::Tensor& linalg_vector_norm_out_vulkan(const at::Tensor& self, const at::Scalar& ord, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor& out)
{
    at::Tensor x = dtype.has_value() ? self.to(*dtype) : self;
    return fill_out(out, p_norm(x, ord.toDouble(), dim, keepdim));
}

at::Tensor& norm_scalaropt_dim_out_vulkan(const at::Tensor& self, const c10::optional<at::Scalar>& p, at::IntArrayRef dim, bool keepdim, at::Tensor& out)
{
    double pv = p.has_value() ? p->toDouble() : 2.0;
    return fill_out(out, p_norm(self, pv, dim, keepdim));
}

at::Tensor& renorm_out_vulkan(const at::Tensor& self, const at::Scalar& p, int64_t dim, const at::Scalar& maxnorm, at::Tensor& out)
{
    int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    std::vector<int64_t> rdims;
    for (int64_t i = 0; i < self.dim(); i++) if (i != d) rdims.push_back(i);

    at::Tensor norms = p_norm(self, p.toDouble(), rdims, /*keepdim=*/true);
    at::Tensor scaled = at::mul(at::reciprocal(at::add(norms, 1e-7)), maxnorm.toDouble());
    at::Tensor factor = at::where(at::gt(norms, maxnorm), scaled, at::ones_like(scaled));
    return fill_out(out, at::mul(self, factor));
}

at::Tensor& softshrink_out_vulkan(const at::Tensor& self, const at::Scalar& lambd, at::Tensor& out)
{
    at::Tensor res = at::mul(at::sign(self), at::clamp_min(at::sub(at::abs(self), lambd), 0));
    return fill_out(out, res);
}

at::Tensor& hardshrink_out_vulkan(const at::Tensor& self, const at::Scalar& lambd, at::Tensor& out)
{
    return fill_out(out, at::where(at::le(at::abs(self), lambd), at::zeros_like(self), self));
}

at::Tensor& threshold_out_vulkan(const at::Tensor& self, const at::Scalar& threshold, const at::Scalar& value, at::Tensor& out)
{
    return fill_out(out, at::where(at::le(self, threshold), at::scalar_tensor(value, self.options()), self));
}

at::Tensor& glu_out_vulkan(const at::Tensor& self, int64_t dim, at::Tensor& out)
{
    int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    int64_t half = self.size(d) / 2;
    at::Tensor a = self.narrow(d, 0, half);
    at::Tensor b = self.narrow(d, half, half);
    return fill_out(out, at::mul(a, at::sigmoid(b)));
}

at::Tensor prelu_vulkan(const at::Tensor& self, const at::Tensor& weight)
{
    at::Tensor w = weight;
    if (weight.numel() != 1 && self.dim() > 1) {
        std::vector<int64_t> vshape(self.dim(), 1);
        vshape[1] = weight.numel();
        w = weight.view(vshape);
    }
    return at::where(at::gt(self, 0), self, at::mul(w, self));
}

std::tuple<at::Tensor, at::Tensor> log_sigmoid_forward_vulkan(const at::Tensor& self)
{
    at::Tensor output = at::neg(at::softplus(at::neg(self), 1, 20));
    at::Tensor buffer = at::empty({0}, self.options());
    return std::make_tuple(output, buffer);
}

at::Tensor huber_loss_vulkan(const at::Tensor& self, const at::Tensor& target, int64_t reduction, double delta)
{
    at::Tensor d = at::sub(self, target);
    at::Tensor absd = at::abs(d);
    at::Tensor quad = at::mul(at::mul(d, d), 0.5);
    at::Tensor lin = at::mul(at::sub(absd, 0.5 * delta), delta);
    return reduce_loss(at::where(at::lt(absd, delta), quad, lin), reduction);
}

at::Tensor& smooth_l1_loss_out_vulkan(const at::Tensor& self, const at::Tensor& target, int64_t reduction, double beta, at::Tensor& out)
{
    at::Tensor d = at::sub(self, target);
    at::Tensor absd = at::abs(d);
    at::Tensor loss;
    if (beta == 0.0) {
        loss = absd;
    } else {
        at::Tensor quad = at::div(at::mul(at::mul(d, d), 0.5), beta);
        at::Tensor lin = at::sub(absd, 0.5 * beta);
        loss = at::where(at::lt(absd, beta), quad, lin);
    }
    return fill_out(out, reduce_loss(loss, reduction));
}

at::Tensor& mse_loss_out_vulkan(const at::Tensor& self, const at::Tensor& target, int64_t reduction, at::Tensor& out)
{
    at::Tensor d = at::sub(self, target);
    return fill_out(out, reduce_loss(at::mul(d, d), reduction));
}

at::Tensor binary_cross_entropy_vulkan(const at::Tensor& self, const at::Tensor& target, const c10::optional<at::Tensor>& weight, int64_t reduction)
{
    at::ScalarType odt = self.scalar_type();
    bool low = is_low_prec(odt);
    at::Tensor p = low ? self.to(at::kFloat) : self;
    at::Tensor t = low ? target.to(at::kFloat) : target;
    at::Tensor logp = at::clamp_min(at::log(p), -100);
    at::Tensor log1mp = at::clamp_min(at::log(at::rsub(p, 1)), -100);
    at::Tensor loss = at::neg(at::add(at::mul(t, logp), at::mul(at::rsub(t, 1), log1mp)));
    if (weight.has_value()) loss = at::mul(loss, low ? weight->to(at::kFloat) : *weight);
    at::Tensor r = reduce_loss(loss, reduction);
    return low ? r.to(odt) : r;
}

at::Tensor soft_margin_loss_vulkan(const at::Tensor& self, const at::Tensor& target, int64_t reduction)
{
    at::ScalarType odt = self.scalar_type();
    bool low = is_low_prec(odt);
    at::Tensor s = low ? self.to(at::kFloat) : self;
    at::Tensor t = low ? target.to(at::kFloat) : target;
    at::Tensor loss = at::softplus(at::neg(at::mul(t, s)), 1, 20);
    at::Tensor r = reduce_loss(loss, reduction);
    return low ? r.to(odt) : r;
}

at::Tensor& special_xlog1py_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out)
{
    auto up = [](const at::Tensor& t) {
        return (t.is_floating_point() && !is_low_prec(t.scalar_type())) ? t : t.to(at::kFloat);
    };
    return fill_out(out, at::xlogy(up(self), at::add(up(other), 1)));
}

at::Tensor trace_vulkan(const at::Tensor& self)
{
    return at::sum(at::diagonal(self));
}

at::Tensor& gelu_out_vulkan(const at::Tensor& self, c10::string_view approximate, at::Tensor& out)
{
    at::Tensor r;
    if (approximate == "tanh") {
        at::Tensor x3 = at::mul(at::mul(self, self), self);
        at::Tensor inner = at::mul(at::add(self, at::mul(x3, 0.044715)), 0.7978845608028654);
        r = at::mul(at::mul(self, 0.5), at::add(at::tanh(inner), 1));
    } else {
        r = at::mul(at::mul(self, 0.5), at::add(at::erf(at::mul(self, 0.7071067811865476)), 1));
    }
    return fill_out(out, r);
}

at::Tensor& masked_fill_scalar_vulkan_(at::Tensor& self, const at::Tensor& mask, const at::Scalar& value)
{
    TORCH_CHECK(mask.scalar_type() == at::kBool, "masked_fill_ only supports boolean masks, but got mask with dtype ", mask.scalar_type());
    return copy_inplace_result(self, at::where(mask, at::scalar_tensor(value, self.options()), self));
}

at::Tensor& masked_fill_tensor_vulkan_(at::Tensor& self, const at::Tensor& mask, const at::Tensor& value)
{
    TORCH_CHECK(mask.scalar_type() == at::kBool, "masked_fill_ only supports boolean masks, but got mask with dtype ", mask.scalar_type());
    TORCH_CHECK(value.dim() == 0, "masked_fill_ only supports a 0-dimensional value tensor, but got tensor with ", value.dim(), " dimension(s).");
    return copy_inplace_result(self, at::where(mask, value.to(self.options()), self));
}

at::Tensor& nan_to_num_out_vulkan(const at::Tensor& self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf, at::Tensor& out)
{
    if (!self.is_floating_point()) { out.copy_(self); return out; }
    double type_max, type_min;
    switch (self.scalar_type()) {
        case at::kHalf: type_max = static_cast<double>(std::numeric_limits<c10::Half>::max()); break;
        case at::kBFloat16: type_max = static_cast<double>(std::numeric_limits<c10::BFloat16>::max()); break;
        case at::kFloat: type_max = std::numeric_limits<float>::max(); break;
        default: type_max = std::numeric_limits<double>::max(); break;
    }
    type_min = -type_max;
    double nan_v = nan.value_or(0.0);
    double posinf_v = posinf.value_or(type_max);
    double neginf_v = neginf.value_or(type_min);

    at::Tensor r = self;
    r = at::where(at::isnan(r), at::scalar_tensor(nan_v, self.options()), r);
    r = at::where(at::isposinf(r), at::scalar_tensor(posinf_v, self.options()), r);
    r = at::where(at::isneginf(r), at::scalar_tensor(neginf_v, self.options()), r);
    out.copy_(r);
    return out;
}

at::Tensor& heaviside_out_vulkan(const at::Tensor& self, const at::Tensor& values, at::Tensor& out)
{
    at::Tensor one = at::scalar_tensor(1, self.options());
    at::Tensor zero = at::scalar_tensor(0, self.options());
    at::Tensor r = at::where(at::gt(self, 0), one, values);
    r = at::where(at::lt(self, 0), zero, r);
    if (self.is_floating_point()) r = at::where(at::isnan(self), self, r);
    out.copy_(r);
    return out;
}

at::Tensor& cat_out_vulkan(const at::ITensorListRef& tensors, int64_t dim, at::Tensor& out)
{
    int64_t d = c10::maybe_wrap_dim(dim, out.dim());
    int64_t offset = 0;
    for (const at::Tensor& t : tensors) {
        if (t.numel() == 0) continue;
        int64_t n = t.size(d);
        out.narrow(d, offset, n).copy_(t);
        offset += n;
    }
    return out;
}

at::Tensor roll_vulkan(const at::Tensor& self, at::IntArrayRef shifts, at::IntArrayRef dims)
{
    if (dims.empty()) {
        at::Tensor flat = self.contiguous().view({-1});
        return roll_vulkan(flat, shifts, {0}).view(self.sizes());
    }
    at::Tensor result = self;
    for (size_t i = 0; i < dims.size(); i++) {
        int64_t d = c10::maybe_wrap_dim(dims[i], result.dim());
        int64_t size = result.size(d);
        if (size == 0) continue;
        int64_t shift = ((shifts[i] % size) + size) % size;
        if (shift == 0) continue;
        at::Tensor front = result.narrow(d, size - shift, shift);
        at::Tensor back = result.narrow(d, 0, size - shift);
        result = at::cat({front, back}, d);
    }
    return result;
}

at::Tensor silu_vulkan(const at::Tensor& self)
{
    return at::mul(self, at::sigmoid(self));
}

at::Tensor hardtanh_vulkan(const at::Tensor& self, const at::Scalar& min_val, const at::Scalar& max_val)
{
    return at::clamp(self, min_val, max_val);
}

at::Tensor hardsigmoid_vulkan(const at::Tensor& self)
{
    return at::div(at::clamp(at::add(self, 3), 0, 6), 6);
}

at::Tensor hardswish_vulkan(const at::Tensor& self)
{
    return at::mul(self, at::div(at::clamp(at::add(self, 3), 0, 6), 6));
}

at::Tensor leaky_relu_vulkan(const at::Tensor& self, const at::Scalar& negative_slope)
{
    at::Tensor pos = at::clamp_min(self, 0);
    at::Tensor neg = at::clamp_max(self, 0);
    return at::add(pos, at::mul(neg, negative_slope));
}

at::Tensor elu_vulkan(const at::Tensor& self, const at::Scalar& alpha, const at::Scalar& scale, const at::Scalar& input_scale)
{
    at::Tensor pos = at::clamp_min(self, 0);
    at::Tensor inner = at::mul(at::expm1(at::mul(self, input_scale)), alpha);
    at::Tensor neg = at::clamp_max(inner, 0);
    return at::mul(at::add(pos, neg), scale);
}

at::Tensor softplus_vulkan(const at::Tensor& self, const at::Scalar& beta, const at::Scalar& threshold)
{
    at::Tensor bx = at::mul(self, beta);
    at::Tensor sp = at::div(at::add(at::clamp_min(bx, 0), at::log1p(at::exp(at::neg(at::abs(bx))))), beta);
    return at::where(at::gt(bx, threshold), self, sp);
}

at::Tensor mish_vulkan(const at::Tensor& self)
{
    return at::mul(self, at::tanh(softplus_vulkan(self, 1, 20)));
}

at::Tensor logit_vulkan(const at::Tensor& self, c10::optional<double> eps)
{
    at::Tensor x = promote_to_float(self);
    // lo is checked first, matching the CPU when eps > 0.5 makes lo > hi
    if (eps.has_value()) {
        at::Tensor lo = at::full({}, *eps, x.options());
        at::Tensor hi = at::full({}, 1.0 - *eps, x.options());
        x = at::where(at::lt(x, lo), lo, at::where(at::gt(x, hi), hi, x));
    }
    return at::log(at::div(x, at::rsub(x, 1)));
}

at::Tensor isnan_vulkan(const at::Tensor& self)
{
    return at::ne(self, self);
}

at::Tensor log_softmax_vulkan(const at::Tensor& self, int64_t dim, bool half_to_float)
{
    at::Tensor input = half_to_float ? self.to(at::kFloat) : self;
    if (input.numel() == 0) return input.clone();

    at::Tensor self_max = at::amax(input, {dim}, /*keepdim=*/true);
    at::Tensor shifted = at::sub(input, self_max);
    at::Tensor sum_exp = at::sum(at::exp(shifted), {dim}, /*keepdim=*/true);
    return at::sub(shifted, at::log(sum_exp));
}

at::Tensor log_softmax_backward_vulkan(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    at::ScalarType /* input_dtype */)
{
    if (grad_output.numel() == 0) return grad_output.clone();

    at::Tensor sum_grad = at::sum(grad_output, {dim}, /*keepdim=*/true);
    at::Tensor softmax = at::exp(output);
    return at::sub(grad_output, at::mul(softmax, sum_grad));
}

at::Tensor softmax_vulkan(const at::Tensor& self, int64_t dim, bool half_to_float)
{
    return at::exp(log_softmax_vulkan(self, dim, half_to_float));
}

at::Tensor softmax_backward_vulkan(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    at::ScalarType /* input_dtype */)
{
    if (grad_output.numel() == 0) return grad_output.clone();

    at::ScalarType out_type = grad_output.scalar_type();
    bool upcast = out_type == at::kHalf || out_type == at::kBFloat16;
    at::Tensor go = upcast ? grad_output.to(at::kFloat) : grad_output;
    at::Tensor out = upcast ? output.to(at::kFloat) : output;

    at::Tensor inner = at::sum(at::mul(go, out), {dim}, /*keepdim=*/true);
    at::Tensor res = at::mul(out, at::sub(go, inner));
    return upcast ? res.to(out_type) : res;
}

at::Tensor logsumexp_vulkan(
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

std::tuple<at::Tensor, at::Tensor> compute_var_mean(
    const at::Tensor& self_in,
    at::OptionalIntArrayRef dim,
    double correction,
    bool keepdim)
{
    TORCH_CHECK(at::isFloatingType(self_in.scalar_type()) || at::isComplexType(self_in.scalar_type()),
        "std and var only support floating point and complex dtypes"
    );

    at::ScalarType odt = self_in.scalar_type();
    bool low = is_low_prec(odt);
    at::Tensor self = low ? self_in.to(at::kFloat) : self_in;

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
    if (low) return std::make_tuple(var.to(odt), mean.to(odt));
    return std::make_tuple(var, mean);
}

double correction_or_default(const c10::optional<at::Scalar>& correction)
{
    return correction.has_value() ? correction->toDouble() : 1.0;
}

at::Tensor var_correction_vulkan(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<at::Scalar>& correction,
    bool keepdim)
{
    return std::get<0>(compute_var_mean(self, dim, correction_or_default(correction), keepdim));
}

at::Tensor std_correction_vulkan(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<at::Scalar>& correction,
    bool keepdim)
{
    return at::sqrt(var_correction_vulkan(self, dim, correction, keepdim));
}

std::tuple<at::Tensor, at::Tensor> var_mean_correction_vulkan(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<at::Scalar>& correction,
    bool keepdim)
{
    return compute_var_mean(self, dim, correction_or_default(correction), keepdim);
}

std::tuple<at::Tensor, at::Tensor> std_mean_correction_vulkan(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<at::Scalar>& correction,
    bool keepdim)
{
    at::Tensor var, mean;
    std::tie(var, mean) = compute_var_mean(self, dim, correction_or_default(correction), keepdim);
    return std::make_tuple(at::sqrt(var), mean);
}

at::Tensor& arange_start_out_vulkan(
    const at::Scalar& start, 
    const at::Scalar& end, 
    const at::Scalar& step, 
    at::Tensor& out
) {
    at::Tensor cpu = at::arange(start, end, step, out.options().device(at::kCPU));
    out.resize_(cpu.sizes());
    out.copy_(cpu);
    return out;
}

at::Tensor& linspace_out_vulkan(
    const at::Scalar& start,
    const at::Scalar& end, 
    int64_t steps, 
    at::Tensor& out
) {
    at::Tensor cpu = at::linspace(start, end, steps, out.options().device(at::kCPU));
    out.resize_(cpu.sizes());
    out.copy_(cpu);
    return out;
}

at::Tensor& logspace_out_vulkan(
    const at::Scalar& start, 
    const at::Scalar& end, 
    int64_t steps, 
    double base, 
    at::Tensor& out
) {
    at::Tensor cpu = at::logspace(start, end, steps, base, out.options().device(at::kCPU));
    out.resize_(cpu.sizes());
    out.copy_(cpu);
    return out;
}

at::Tensor& eye_m_out_vulkan(
    c10::SymInt n, 
    c10::SymInt m, 
    at::Tensor& out
) {
    at::Tensor cpu = at::eye(n.expect_int(), m.expect_int(), out.options().device(at::kCPU));
    out.resize_(cpu.sizes());
    out.copy_(cpu);
    return out;
}

at::Tensor tril_indices_vulkan(
    int64_t row, 
    int64_t col, 
    int64_t offset, 
    c10::optional<at::ScalarType> dtype, 
    c10::optional<at::Layout> layout, 
    c10::optional<at::Device> device, 
    c10::optional<bool> /* pin_memory */
) {
    at::TensorOptions cpu_opts = at::TensorOptions().dtype(dtype.value_or(at::kLong)).layout(layout.value_or(at::kStrided)).device(at::kCPU);
    at::Tensor cpu = at::tril_indices(row, col, offset, cpu_opts);
    return cpu.to(device.value_or(at::Device(at::DeviceType::PrivateUse1, VulkanContext::CurrentDevice())));
}

at::Tensor triu_indices_vulkan(
    int64_t row, 
    int64_t col, 
    int64_t offset, 
    c10::optional<at::ScalarType> dtype, 
    c10::optional<at::Layout> layout, 
    c10::optional<at::Device> device, 
    c10::optional<bool> /* pin_memory */
) {
    at::TensorOptions cpu_opts = at::TensorOptions().dtype(dtype.value_or(at::kLong)).layout(layout.value_or(at::kStrided)).device(at::kCPU);
    at::Tensor cpu = at::triu_indices(row, col, offset, cpu_opts);
    return cpu.to(device.value_or(at::Device(at::DeviceType::PrivateUse1, VulkanContext::CurrentDevice())));
}

at::Tensor gelu_backward_vulkan(const at::Tensor& grad_output, const at::Tensor& self, c10::string_view approximate)
{
    at::Tensor dy;
    if (approximate == "tanh") {
        // y = 0.5x(1 + tanh(inner)), inner = c(x + a*x^3), c = sqrt(2/pi), a = 0.044715
        // dy/dx = 0.5(1 + tanh(inner)) + 0.5x(1 - tanh(inner)^2) * c(1 + 3a*x^2)
        at::Tensor x2 = at::mul(self, self);
        at::Tensor inner = at::mul(at::add(self, at::mul(at::mul(x2, self), 0.044715)), 0.7978845608028654);
        at::Tensor t = at::tanh(inner);
        at::Tensor dinner = at::mul(at::add(at::mul(x2, 3 * 0.044715), 1), 0.7978845608028654);
        at::Tensor term1 = at::mul(at::add(t, 1), 0.5);
        at::Tensor term2 = at::mul(at::mul(at::mul(self, 0.5), at::rsub(at::mul(t, t), 1)), dinner);
        dy = at::add(term1, term2);
    } else {
        // y = x * Phi(x), Phi = 0.5(1 + erf(x/sqrt2))
        // dy/dx = Phi(x) + x * phi(x), phi = (1/sqrt(2*pi)) * exp(-x^2/2)
        at::Tensor cdf = at::mul(at::add(at::erf(at::mul(self, 0.7071067811865476)), 1), 0.5);
        at::Tensor pdf = at::mul(at::exp(at::mul(at::mul(self, self), -0.5)), 0.3989422804014327);
        dy = at::add(cdf, at::mul(self, pdf));
    }
    return at::mul(grad_output, dy);
}

int64_t fused_sdp_choice_vulkan(
    const at::Tensor& /* query */,
    const at::Tensor& /* key */,
    const at::Tensor& /* value */,
    const c10::optional<at::Tensor>& /* attn_mask */,
    double /* dropout_p */,
    bool /* is_causal */,
    c10::optional<double> /* scale */,
    bool /* enable_gqa */)
{
    return static_cast<int64_t>(at::SDPBackend::math);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_layer_norm_vulkan(
    const at::Tensor& input,
    c10::SymIntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    double eps)
{
    int64_t ndim = input.dim();
    int64_t nnorm = static_cast<int64_t>(normalized_shape.size());
    std::vector<int64_t> reduce_dims;
    for (int64_t i = ndim - nnorm; i < ndim; i++) reduce_dims.push_back(i);

    at::ScalarType odt = input.scalar_type();
    bool low = is_low_prec(odt);
    at::Tensor x = low ? input.to(at::kFloat) : input;

    auto [var, mean] = compute_var_mean(x, reduce_dims, /*correction=*/0.0, /*keepdim=*/true);
    at::Tensor rstd = at::rsqrt(at::add(var, eps));
    at::Tensor xhat = at::mul(at::sub(x, mean), rstd);

    at::Tensor y = xhat;
    if (weight.has_value()) y = at::mul(y, low ? weight->to(at::kFloat) : *weight);
    if (bias.has_value()) y = at::add(y, low ? bias->to(at::kFloat) : *bias);

    if (low) {
        y = y.to(odt);
        mean = mean.to(odt);
        rstd = rstd.to(odt);
    }
    return std::make_tuple(y, mean, rstd);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_group_norm_vulkan(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    c10::SymInt N,
    c10::SymInt C,
    c10::SymInt HxW,
    int64_t group,
    double eps)
{
    int64_t n = N.guard_int(__FILE__, __LINE__);
    int64_t c = C.guard_int(__FILE__, __LINE__);
    int64_t hw = HxW.guard_int(__FILE__, __LINE__);
    TORCH_CHECK(group > 0 && c % group == 0, "Expected number of channels in input to be divisible by num_groups, but got input of shape ",
        input.sizes(), " and num_groups=", group);

    at::ScalarType odt = input.scalar_type();
    bool low = is_low_prec(odt);
    at::Tensor x = (low ? input.to(at::kFloat) : input).reshape({n, group, (c / group) * hw});

    auto [var, mean] = compute_var_mean(x, {2}, /*correction=*/0.0, /*keepdim=*/true);
    at::Tensor rstd = at::rsqrt(at::add(var, eps));
    at::Tensor y = at::mul(at::sub(x, mean), rstd).reshape(input.sizes());

    // weight and bias are per channel
    std::vector<int64_t> channel_shape(input.dim(), 1);
    channel_shape[1] = c;
    if (weight.has_value() && weight->defined()) y = at::mul(y, (low ? weight->to(at::kFloat) : *weight).reshape(channel_shape));
    if (bias.has_value() && bias->defined()) y = at::add(y, (low ? bias->to(at::kFloat) : *bias).reshape(channel_shape));

    mean = mean.reshape({n, group});
    rstd = rstd.reshape({n, group});
    if (low) {
        y = y.to(odt);
        mean = mean.to(odt);
        rstd = rstd.to(odt);
    }
    return std::make_tuple(y, mean, rstd);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_layer_norm_backward_vulkan(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    c10::SymIntArrayRef normalized_shape,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    std::array<bool, 3> output_mask)
{
    int64_t ndim = input.dim();
    int64_t nnorm = static_cast<int64_t>(normalized_shape.size());
    std::vector<int64_t> reduce_dims, batch_dims;
    for (int64_t i = 0; i < ndim; i++) {
        (i < ndim - nnorm ? batch_dims : reduce_dims).push_back(i);
    }

    auto sum_over_batch = [&](const at::Tensor& t) -> at::Tensor {
        return batch_dims.empty() ? t : at::sum(t, batch_dims, /*keepdim=*/false);
    };

    at::ScalarType odt = input.scalar_type();
    bool low = is_low_prec(odt);
    at::Tensor x = low ? input.to(at::kFloat) : input;
    at::Tensor dy = low ? grad_out.to(at::kFloat) : grad_out;
    at::Tensor m = low ? mean.to(at::kFloat) : mean;
    at::Tensor r = low ? rstd.to(at::kFloat) : rstd;
    at::Tensor w = weight.has_value() ? (low ? weight->to(at::kFloat) : *weight) : at::Tensor();

    at::Tensor xhat = at::mul(at::sub(x, m), r);
    at::Tensor dxhat = w.defined() ? at::mul(dy, w) : dy;

    at::Tensor grad_input, grad_weight, grad_bias;
    if (output_mask[0]) {
        at::Tensor mean_dxhat = at::mean(dxhat, reduce_dims, /*keepdim=*/true);
        at::Tensor mean_dxhat_xhat = at::mean(at::mul(dxhat, xhat), reduce_dims, /*keepdim=*/true);
        at::Tensor gi = at::mul(r, at::sub(at::sub(dxhat, mean_dxhat), at::mul(xhat, mean_dxhat_xhat)));
        grad_input = low ? gi.to(odt) : gi;
    }
    if (output_mask[1] && weight.has_value()) {
        at::Tensor gw = sum_over_batch(at::mul(dy, xhat));
        grad_weight = low ? gw.to(weight->scalar_type()) : gw;
    }
    if (output_mask[2] && bias.has_value()) {
        at::Tensor gb = sum_over_batch(dy);
        grad_bias = low ? gb.to(bias->scalar_type()) : gb;
    }
    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

double loss_grad_scale(const at::Tensor& self, int64_t reduction)
{
    return (reduction == 1 && self.numel() > 0) ? 1.0 / static_cast<double>(self.numel()) : 1.0;
}

at::Tensor& sigmoid_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& output, at::Tensor& grad_input)
{
    return fill_out(grad_input, at::mul(grad_output, at::mul(at::rsub(output, 1), output)));
}

at::Tensor& tanh_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& output, at::Tensor& grad_input)
{
    return fill_out(grad_input, at::mul(grad_output, at::rsub(at::mul(output, output), 1)));
}

at::Tensor& leaky_relu_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& self, const at::Scalar& negative_slope, bool /* self_is_result */, at::Tensor& grad_input)
{
    return fill_out(grad_input, at::where(at::gt(self, 0), grad_output, at::mul(grad_output, negative_slope)));
}

at::Tensor& elu_backward_out_vulkan(
    const at::Tensor& grad_output,
    const at::Scalar& alpha,
    const at::Scalar& scale,
    const at::Scalar& input_scale,
    bool is_result,
    const at::Tensor& self_or_result,
    at::Tensor& grad_input)
{
    double negcoef = alpha.toDouble() * scale.toDouble();
    double negiptcoef = input_scale.toDouble();
    at::Tensor negative = is_result
        ? at::mul(grad_output, at::mul(at::add(self_or_result, negcoef), negiptcoef))
        : at::mul(grad_output, at::mul(at::exp(at::mul(self_or_result, negiptcoef)), negiptcoef * negcoef));
    at::Tensor positive = at::mul(grad_output, scale);
    return fill_out(grad_input, at::where(at::le(self_or_result, 0), negative, positive));
}

at::Tensor hardtanh_backward_vulkan(const at::Tensor& grad_output, const at::Tensor& self, const at::Scalar& min_val, const at::Scalar& max_val)
{
    at::Tensor outside = at::logical_or(at::le(self, min_val), at::ge(self, max_val));
    return at::where(outside, at::zeros_like(grad_output), grad_output);
}

at::Tensor& hardtanh_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& self, const at::Scalar& min_val, const at::Scalar& max_val, at::Tensor& grad_input)
{
    return fill_out(grad_input, hardtanh_backward_vulkan(grad_output, self, min_val, max_val));
}

at::Tensor& softplus_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& self, const at::Scalar& beta, const at::Scalar& threshold, at::Tensor& grad_input)
{
    at::Tensor bx = at::mul(self, beta);
    return fill_out(grad_input, at::where(at::gt(bx, threshold), grad_output, at::mul(grad_output, at::sigmoid(bx))));
}

at::Tensor& hardsigmoid_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& self, at::Tensor& grad_input)
{
    at::Tensor inside = at::logical_and(at::gt(self, -3), at::lt(self, 3));
    return fill_out(grad_input, at::where(inside, at::div(grad_output, 6), at::zeros_like(grad_output)));
}

at::Tensor hardswish_backward_vulkan(const at::Tensor& grad_output, const at::Tensor& self)
{
    at::Tensor middle = at::mul(grad_output, at::add(at::div(self, 3), 0.5));
    at::Tensor upper = at::where(at::le(self, 3), middle, grad_output);
    return at::where(at::lt(self, -3), at::zeros_like(grad_output), upper);
}

at::Tensor log_sigmoid_backward_vulkan(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& /* buffer */)
{
    return at::mul(grad_output, at::sigmoid(at::neg(self)));
}

at::Tensor& log_sigmoid_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& buffer, at::Tensor& grad_input)
{
    return fill_out(grad_input, log_sigmoid_backward_vulkan(grad_output, self, buffer));
}

at::Tensor mse_loss_backward_vulkan(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& target, int64_t reduction)
{
    double norm = 2.0 * loss_grad_scale(self, reduction);
    return at::mul(at::mul(at::sub(self, target), grad_output), norm);
}

at::Tensor& mse_loss_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& target, int64_t reduction, at::Tensor& grad_input)
{
    return fill_out(grad_input, mse_loss_backward_vulkan(grad_output, self, target, reduction));
}

at::Tensor& smooth_l1_loss_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& target, int64_t reduction, double beta, at::Tensor& grad_input)
{
    double norm = loss_grad_scale(self, reduction);
    at::Tensor x = at::sub(self, target);
    at::Tensor g = at::mul(grad_output, norm);
    if (beta == 0.0) return fill_out(grad_input, at::mul(at::sign(x), g));
    at::Tensor linear = at::mul(at::sign(x), g);
    at::Tensor quadratic = at::mul(at::div(x, beta), g);
    return fill_out(grad_input, at::where(at::lt(at::abs(x), beta), quadratic, linear));
}

at::Tensor& huber_loss_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& target, int64_t reduction, double delta, at::Tensor& grad_input)
{
    double norm = loss_grad_scale(self, reduction);
    at::Tensor x = at::sub(self, target);
    at::Tensor g = at::mul(grad_output, norm);
    at::Tensor linear = at::mul(at::sign(x), at::mul(g, delta));
    at::Tensor quadratic = at::mul(x, g);
    return fill_out(grad_input, at::where(at::le(at::abs(x), delta), quadratic, linear));
}

at::Tensor binary_cross_entropy_backward_vulkan(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction)
{
    static constexpr double EPSILON = 1e-12;
    at::Tensor denominator = at::clamp_min(at::mul(at::rsub(self, 1), self), EPSILON);
    at::Tensor grad = at::div(at::mul(at::sub(self, target), grad_output), denominator);
    if (weight.has_value() && weight->defined()) grad = at::mul(grad, *weight);
    return at::mul(grad, loss_grad_scale(self, reduction));
}

at::Tensor& binary_cross_entropy_backward_out_vulkan(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    at::Tensor& grad_input)
{
    return fill_out(grad_input, binary_cross_entropy_backward_vulkan(grad_output, self, target, weight, reduction));
}

at::Tensor& soft_margin_loss_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& target, int64_t reduction, at::Tensor& grad_input)
{
    // d/dx log(1 + exp(-y x)) = -y sigmoid(-y x)
    at::Tensor grad = at::mul(at::neg(target), at::sigmoid(at::neg(at::mul(target, self))));
    return fill_out(grad_input, at::mul(at::mul(grad, grad_output), loss_grad_scale(self, reduction)));
}

std::tuple<at::Tensor, at::Tensor> prelu_backward_vulkan(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& weight)
{
    at::Tensor w = weight;
    if (weight.numel() != 1 && self.dim() > 1) {
        std::vector<int64_t> vshape(self.dim(), 1);
        vshape[1] = weight.numel();
        w = weight.reshape(vshape);
    }
    at::Tensor positive = at::gt(self, 0);
    at::Tensor input_grad = at::where(positive, grad_output, at::mul(w, grad_output));
    at::Tensor weight_grad = at::where(positive, at::zeros_like(grad_output), at::mul(self, grad_output));
    return std::make_tuple(input_grad, weight_grad);
}

at::Tensor& logit_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& self, c10::optional<double> eps, at::Tensor& grad_input)
{
    at::Tensor grad = at::div(grad_output, at::mul(self, at::rsub(self, 1)));
    double lo = eps.has_value() && *eps >= 0 ? *eps : 0.0;
    double hi = 1.0 - lo;
    at::Tensor outside = at::logical_or(at::lt(self, lo), at::gt(self, hi));
    double fill = eps.has_value() && *eps >= 0 ? 0.0 : std::numeric_limits<double>::quiet_NaN();
    return fill_out(grad_input, at::where(outside, at::full_like(grad, fill), grad));
}

at::Tensor shrink_backward(const at::Tensor& grad_output, const at::Tensor& self, const at::Scalar& lambd)
{
    double l = lambd.toDouble();
    at::Tensor inside = at::logical_and(at::ge(self, -l), at::le(self, l));
    return at::where(inside, at::zeros_like(grad_output), grad_output);
}

at::Tensor& hardshrink_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& self, const at::Scalar& lambd, at::Tensor& grad_input)
{
    return fill_out(grad_input, shrink_backward(grad_output, self, lambd));
}

at::Tensor& softshrink_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& self, const at::Scalar& lambd, at::Tensor& grad_input)
{
    return fill_out(grad_input, shrink_backward(grad_output, self, lambd));
}

at::Tensor glu_backward_vulkan(const at::Tensor& grad_output, const at::Tensor& self, int64_t dim)
{
    int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    int64_t half = self.size(d) / 2;
    at::Tensor a = self.narrow(d, 0, half);
    at::Tensor sig = at::sigmoid(self.narrow(d, half, half));
    at::Tensor grad_a = at::mul(grad_output, sig);
    at::Tensor grad_b = at::mul(at::mul(grad_a, a), at::rsub(sig, 1));
    return at::cat({grad_a, grad_b}, d);
}

at::Tensor& glu_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& self, int64_t dim, at::Tensor& grad_input)
{
    return fill_out(grad_input, glu_backward_vulkan(grad_output, self, dim));
}

at::Tensor addbmm_vulkan(const at::Tensor& self, const at::Tensor& batch1, const at::Tensor& batch2, const at::Scalar& beta, const at::Scalar& alpha)
{
    TORCH_CHECK(batch1.dim() == 3 && batch2.dim() == 3, "batch1 and batch2 must be 3D tensors");
    TORCH_CHECK(batch1.size(0) == batch2.size(0), "batch1 and batch2 must have same number of batches, got ", batch1.size(0), " and ", batch2.size(0));
    TORCH_CHECK(batch1.size(2) == batch2.size(1), "Incompatible matrix sizes for bmm (", batch1.size(1), "x", batch1.size(2), " and ", batch2.size(1), "x", batch2.size(2), ")");
    int64_t b = batch1.size(0), n = batch1.size(1), k = batch1.size(2), p = batch2.size(2);

    at::Tensor product = (b == 0 || k == 0)
        ? at::zeros({n, p}, batch1.options())
        : at::mm(batch1.permute({1, 0, 2}).reshape({n, b * k}), batch2.reshape({b * k, p}));
    at::Scalar a = alpha, c = beta;
    if (at::isIntegralType(self.scalar_type(), /*includeBool=*/true)) { a = alpha.toLong(); c = beta.toLong(); }
    at::Tensor result = at::mul(product, a);
    if (c.toDouble() != 0.0) result = at::add(result, at::mul(self.expand({n, p}), c));
    return result.to(self.scalar_type());
}

at::Tensor& addbmm_out_vulkan(const at::Tensor& self, const at::Tensor& batch1, const at::Tensor& batch2, const at::Scalar& beta, const at::Scalar& alpha, at::Tensor& out)
{
    return fill_out(out, addbmm_vulkan(self, batch1, batch2, beta, alpha));
}

at::Tensor& addbmm_vulkan_(at::Tensor& self, const at::Tensor& batch1, const at::Tensor& batch2, const at::Scalar& beta, const at::Scalar& alpha)
{
    return copy_inplace_result(self, addbmm_vulkan(self, batch1, batch2, beta, alpha));
}

at::Tensor& mm_out_vulkan(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out)
{
    return fill_out(out, at::mm(self, mat2));
}

at::Tensor mm_dtype_vulkan(const at::Tensor& self, const at::Tensor& mat2, at::ScalarType out_dtype)
{
    TORCH_CHECK(out_dtype == self.scalar_type() || (out_dtype == at::kFloat && is_low_prec(self.scalar_type())),
        "out_dtype must be the same as input dtype or fp32 for fp16/bf16 inputs"
    );
    if (out_dtype == self.scalar_type()) return at::mm(self, mat2);
    return at::mm(self.to(at::kFloat), mat2.to(at::kFloat));
}

at::Tensor& mm_dtype_out_vulkan(const at::Tensor& self, const at::Tensor& mat2, at::ScalarType out_dtype, at::Tensor& out)
{
    return fill_out(out, mm_dtype_vulkan(self, mat2, out_dtype));
}

at::Tensor& log_softmax_out_vulkan(const at::Tensor& self, int64_t dim, bool half_to_float, at::Tensor& out)
{
    return fill_out(out, log_softmax_vulkan(self, dim, half_to_float));
}

at::Tensor& softmax_out_vulkan(const at::Tensor& self, int64_t dim, bool half_to_float, at::Tensor& out)
{
    return fill_out(out, softmax_vulkan(self, dim, half_to_float));
}

at::Tensor& log_softmax_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& output, int64_t dim, at::ScalarType input_dtype, at::Tensor& out)
{
    return fill_out(out, log_softmax_backward_vulkan(grad_output, output, dim, input_dtype));
}

at::Tensor& softmax_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& output, int64_t dim, at::ScalarType input_dtype, at::Tensor& grad_input)
{
    return fill_out(grad_input, softmax_backward_vulkan(grad_output, output, dim, input_dtype));
}

at::Tensor& binary_cross_entropy_out_vulkan(const at::Tensor& self, const at::Tensor& target, const c10::optional<at::Tensor>& weight, int64_t reduction, at::Tensor& out)
{
    return fill_out(out, binary_cross_entropy_vulkan(self, target, weight, reduction));
}

at::Tensor& huber_loss_out_vulkan(const at::Tensor& self, const at::Tensor& target, int64_t reduction, double delta, at::Tensor& out)
{
    return fill_out(out, huber_loss_vulkan(self, target, reduction, delta));
}

at::Tensor& elu_out_vulkan(const at::Tensor& self, const at::Scalar& alpha, const at::Scalar& scale, const at::Scalar& input_scale, at::Tensor& out)
{
    return fill_out(out, elu_vulkan(self, alpha, scale, input_scale));
}

at::Tensor& eye_out_vulkan(c10::SymInt n, at::Tensor& out)
{
    return eye_m_out_vulkan(n, n, out);
}

at::Tensor& gelu_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& self, c10::string_view approximate, at::Tensor& grad_input)
{
    return fill_out(grad_input, gelu_backward_vulkan(grad_output, self, approximate));
}

at::Tensor& hardsigmoid_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return fill_out(out, hardsigmoid_vulkan(self));
}

at::Tensor& hardswish_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return fill_out(out, hardswish_vulkan(self));
}

at::Tensor& hardswish_vulkan_(at::Tensor& self)
{
    return copy_inplace_result(self, hardswish_vulkan(self));
}

at::Tensor& hardtanh_out_vulkan(const at::Tensor& self, const at::Scalar& min_val, const at::Scalar& max_val, at::Tensor& out)
{
    return fill_out(out, hardtanh_vulkan(self, min_val, max_val));
}

at::Tensor& hardtanh_vulkan_(at::Tensor& self, const at::Scalar& min_val, const at::Scalar& max_val)
{
    return copy_inplace_result(self, hardtanh_vulkan(self, min_val, max_val));
}

at::Tensor& leaky_relu_out_vulkan(const at::Tensor& self, const at::Scalar& negative_slope, at::Tensor& out)
{
    return fill_out(out, leaky_relu_vulkan(self, negative_slope));
}

std::tuple<at::Tensor&, at::Tensor&> log_sigmoid_forward_out_vulkan(const at::Tensor& self, at::Tensor& output, at::Tensor& buffer)
{
    auto [out, buf] = log_sigmoid_forward_vulkan(self);
    fill_out(output, out);
    buffer.resize_(buf.sizes());
    return {output, buffer};
}

at::Tensor& logit_out_vulkan(const at::Tensor& self, c10::optional<double> eps, at::Tensor& out)
{
    return fill_out(out, logit_vulkan(self, eps));
}

at::Tensor& logit_vulkan_(at::Tensor& self, c10::optional<double> eps)
{
    return copy_inplace_result(self, logit_vulkan(self, eps));
}

at::Tensor& mish_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return fill_out(out, mish_vulkan(self));
}

at::Tensor& silu_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return fill_out(out, silu_vulkan(self));
}

at::Tensor& softplus_out_vulkan(const at::Tensor& self, const at::Scalar& beta, const at::Scalar& threshold, at::Tensor& out)
{
    return fill_out(out, softplus_vulkan(self, beta, threshold));
}

at::Tensor& norm_dtype_out_vulkan(const at::Tensor& self, const c10::optional<at::Scalar>& p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype, at::Tensor& out)
{
    return norm_scalaropt_dim_out_vulkan(self.to(dtype), p, dim, keepdim, out);
}

at::Tensor& var_correction_out_vulkan(const at::Tensor& self, at::OptionalIntArrayRef dim, const c10::optional<at::Scalar>& correction, bool keepdim, at::Tensor& out)
{
    return fill_out(out, var_correction_vulkan(self, dim, correction, keepdim));
}

at::Tensor& std_correction_out_vulkan(const at::Tensor& self, at::OptionalIntArrayRef dim, const c10::optional<at::Scalar>& correction, bool keepdim, at::Tensor& out)
{
    return fill_out(out, std_correction_vulkan(self, dim, correction, keepdim));
}

at::Tensor& mvlgamma_out_vulkan(const at::Tensor& self, int64_t p, at::Tensor& out)
{
    return fill_out(out, at::mvlgamma(self, p));
}

at::Tensor& narrow_copy_out_vulkan(const at::Tensor& self, int64_t dim, int64_t start, int64_t length, at::Tensor& out)
{
    return fill_out(out, self.narrow(dim, start, length).clone(at::MemoryFormat::Contiguous));
}

at::Tensor& range_out_vulkan(const at::Scalar& start, const at::Scalar& end, const at::Scalar& step, at::Tensor& out)
{
    at::Tensor cpu = at::range(start, end, step, out.options().device(at::kCPU));
    out.resize_(cpu.sizes());
    out.copy_(cpu);
    return out;
}

bool is_set_to_vulkan(const at::Tensor& self, const at::Tensor& src)
{
    return self.storage().unsafeGetStorageImpl() == src.storage().unsafeGetStorageImpl()
        && self.storage_offset() == src.storage_offset()
        && self.sizes() == src.sizes()
        && self.strides() == src.strides();
}

void assert_async_vulkan(const at::Tensor& self)
{
    TORCH_CHECK(at::is_nonzero(self), "Expected Tensor with single nonzero value, but got zero");
}

void assert_async_msg_vulkan(const at::Tensor& self, c10::string_view assert_msg)
{
    TORCH_CHECK(at::is_nonzero(self), assert_msg != "" ? assert_msg : "Assertion is failed");
}

at::Tensor functional_assert_async_msg_vulkan(const at::Tensor& self, c10::string_view assert_msg, const at::Tensor& dep_token)
{
    assert_async_msg_vulkan(self, assert_msg);
    return dep_token.clone();
}

at::Tensor make_dep_token_vulkan(
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<at::MemoryFormat> memory_format)
{
    return at::empty({}, at::TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory), memory_format);
}

void amp_foreach_non_finite_check_and_unscale_vulkan_(at::TensorList self, at::Tensor& found_inf, const at::Tensor& inv_scale)
{
    TORCH_CHECK(inv_scale.numel() == 1, "inv_scale must be a 1-element tensor.");
    TORCH_CHECK(found_inf.numel() == 1, "found_inf must be a 1-element tensor.");
    TORCH_CHECK(inv_scale.scalar_type() == at::kFloat, "inv_scale must be a float tensor.");
    TORCH_CHECK(found_inf.scalar_type() == at::kFloat, "found_inf must be a float tensor.");
    for (const at::Tensor& t : self) {
        if (t.numel() == 0) continue;
        at::Tensor nonfinite = at::logical_not(at::all(at::isfinite(t))).to(at::kFloat);
        found_inf.copy_(at::maximum(found_inf, nonfinite.reshape(found_inf.sizes())));
        t.mul_(inv_scale);
    }
}

at::Tensor& amp_update_scale_vulkan_(
    at::Tensor& self,
    at::Tensor& growth_tracker,
    const at::Tensor& found_inf,
    double scale_growth_factor,
    double scale_backoff_factor,
    int64_t growth_interval)
{
    TORCH_CHECK(growth_tracker.scalar_type() == at::kInt, "growth_tracker must be an int tensor.");
    TORCH_CHECK(self.scalar_type() == at::kFloat, "current_scale must be a float tensor.");
    TORCH_CHECK(found_inf.scalar_type() == at::kFloat, "found_inf must be a float tensor.");

    at::Tensor found = at::gt(found_inf, 0);
    at::Tensor successful = at::add(growth_tracker, 1);
    at::Tensor grow = at::logical_and(at::logical_not(found), at::eq(successful, growth_interval));
    at::Tensor grown = at::mul(self, scale_growth_factor);
    at::Tensor scale = at::where(found, at::mul(self, scale_backoff_factor), at::where(at::logical_and(grow, at::isfinite(grown)), grown, self));
    at::Tensor tracker = at::where(at::logical_or(found, grow), at::zeros_like(successful), successful);
    self.copy_(scale);
    growth_tracker.copy_(tracker);
    return self;
}

at::Tensor glu_jvp_vulkan(const at::Tensor& glu, const at::Tensor& x, const at::Tensor& dx, int64_t dim)
{
    dim = at::maybe_wrap_dim(dim, x.dim());
    const int64_t n = glu.size(dim);
    at::Tensor sig_b = at::sigmoid(x.narrow(dim, n, n));
    at::Tensor da = dx.narrow(dim, 0, n);
    at::Tensor db = dx.narrow(dim, n, n);
    return at::add(at::mul(sig_b, da), at::mul(at::mul(glu, at::rsub(sig_b, 1)), db));
}

at::Tensor glu_backward_jvp_vulkan(
    const at::Tensor& grad_x,
    const at::Tensor& grad_glu,
    const at::Tensor& x,
    const at::Tensor& dgrad_glu,
    const at::Tensor& dx,
    int64_t dim)
{
    dim = at::maybe_wrap_dim(dim, x.dim());
    const int64_t n = grad_glu.size(dim);
    at::Tensor a = x.narrow(dim, 0, n);
    at::Tensor sig_b = at::sigmoid(x.narrow(dim, n, n));
    at::Tensor da = dx.narrow(dim, 0, n);
    at::Tensor db = dx.narrow(dim, n, n);
    at::Tensor grad_x_a = grad_x.narrow(dim, 0, n);
    at::Tensor glu = at::mul(a, sig_b);
    at::Tensor db_neg_sig_b = at::sub(db, at::mul(db, sig_b));
    at::Tensor dgrad_x_a = at::add(at::mul(dgrad_glu, sig_b), at::mul(grad_x_a, db_neg_sig_b));
    at::Tensor dgrad_x_b = at::add(
        at::mul(dgrad_x_a, at::sub(a, glu)),
        at::mul(grad_x_a, at::sub(at::sub(da, at::mul(da, sig_b)), at::mul(glu, db_neg_sig_b))));
    return at::cat({dgrad_x_a, dgrad_x_b}, dim);
}

at::Tensor& silu_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& self, at::Tensor& grad_input)
{
    at::Tensor s = at::sigmoid(self);
    return fill_out(grad_input, at::mul(grad_output, at::mul(s, at::add(at::mul(self, at::rsub(s, 1)), 1))));
}

at::Tensor compute_linear_combination_vulkan(const at::Tensor& input, const at::Tensor& coefficients)
{
    return at::tensordot(coefficients.to(input.device(), input.scalar_type()), input, {1}, {0});
}

at::Tensor& compute_linear_combination_out_vulkan(const at::Tensor& input, const at::Tensor& coefficients, at::Tensor& out)
{
    return out.add_(compute_linear_combination_vulkan(input, coefficients));
}

std::tuple<at::Tensor, at::Tensor> batch_norm_update_stats_vulkan(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var,
    double momentum)
{
    TORCH_CHECK(input.numel() != 0, "input tensor must have at least one element, but got input_sizes = ", input.sizes());
    const bool has_mean = running_mean.has_value() && running_mean->defined();
    const bool has_var = running_var.has_value() && running_var->defined();
    at::ScalarType out_dtype = has_mean && running_mean->scalar_type() != input.scalar_type() ? running_mean->scalar_type() : input.scalar_type();

    std::vector<int64_t> dims;
    for (int64_t d = 0; d < input.dim(); ++d) if (d != 1) dims.push_back(d);
    at::Tensor x = is_low_prec(input.scalar_type()) ? input.to(at::kFloat) : input;
    auto [var, mean] = at::var_mean(x, dims, /*correction=*/0, /*keepdim=*/false);
    const int64_t n = input.numel() / input.size(1);

    if (has_mean) running_mean->copy_(at::add(at::mul(mean, momentum), at::mul(*running_mean, 1.0 - momentum)));
    if (has_var) running_var->copy_(at::add(at::mul(at::mul(var, double(n) / double(n - 1)), momentum), at::mul(*running_var, 1.0 - momentum)));
    return {mean.to(out_dtype), var.to(out_dtype)};
}

std::tuple<at::Tensor, at::Tensor> weight_norm_interface_vulkan(const at::Tensor& v, const at::Tensor& g, int64_t dim)
{
    at::ScalarType norm_dtype = is_low_prec(g.scalar_type()) ? at::kFloat : g.scalar_type();
    at::Tensor norm = at::norm_except_dim(v.to(norm_dtype), 2, dim).reshape(g.sizes());
    at::Tensor w = at::mul(v, at::div(g, norm).to(v.scalar_type())).contiguous();
    return {w, norm};
}

std::tuple<at::Tensor, at::Tensor> weight_norm_interface_backward_vulkan(
    const at::Tensor& grad_w,
    const at::Tensor& saved_v,
    const at::Tensor& saved_g,
    const at::Tensor& saved_norms,
    int64_t dim)
{
    return at::_weight_norm_differentiable_backward(grad_w.contiguous(), saved_v.contiguous(), saved_g.contiguous(), saved_norms.contiguous(), dim);
}

at::Tensor expand_softmax_mask(const at::Tensor& input, const at::Tensor& mask, c10::optional<int64_t> mask_type)
{
    TORCH_CHECK(mask.scalar_type() == at::kBool, "Mask should be a boolean tensor");
    int64_t type = (mask.dim() != 2 || input.dim() != 4) ? 2 : mask_type.value_or(-1);
    TORCH_CHECK(type == 0 || type == 1 || type == 2, "Mask Type should be 0 (src_mask) or 1 (src_key_padding_mask), or 2 (default_mask)");
    if (type == 2) {
        TORCH_CHECK(input.sizes() == mask.sizes(), "For mask_type == 2 mask shape should match input shape");
        return mask;
    }
    if (type == 1) {
        TORCH_CHECK(input.size(0) == mask.size(0) && input.size(2) == mask.size(1), "For mask_type == 1 mask shape should be (B, L)");
        return mask.view({input.size(0), 1, 1, input.size(2)}).expand(input.sizes());
    }
    TORCH_CHECK(input.size(2) == mask.size(0) && input.size(2) == mask.size(1), "For mask_type == 0 mask shape should be (L, L)");
    return mask.view({1, 1, input.size(2), input.size(2)}).expand(input.sizes());
}

at::Tensor masked_softmax_vulkan(const at::Tensor& self, const at::Tensor& mask, c10::optional<int64_t> dim, c10::optional<int64_t> mask_type)
{
    at::Tensor m = expand_softmax_mask(self, mask, mask_type);
    int64_t d = at::maybe_wrap_dim(dim.value_or(self.dim() - 1), self.dim());
    at::Tensor filled = at::masked_fill(self, m, -std::numeric_limits<double>::infinity());
    return at::_softmax(filled, d, false);
}

at::Tensor masked_softmax_backward_vulkan(const at::Tensor& grad_output, const at::Tensor& output, const at::Tensor& mask, c10::optional<int64_t> dim)
{
    TORCH_CHECK(grad_output.sizes() == mask.sizes(), "Mask shape should match grad shape");
    TORCH_CHECK(mask.scalar_type() == at::kBool, "Mask should be a boolean tensor");
    int64_t d = at::maybe_wrap_dim(dim.value_or(output.dim() - 1), output.dim());
    at::Tensor zero = at::zeros({}, grad_output.options());
    at::Tensor go = at::where(mask, zero, at::mul(grad_output, output));
    at::Tensor sum = at::sum(go, {d}, /*keepdim=*/true);
    return at::where(mask, zero, at::mul(output, at::sub(grad_output, sum)));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> transform_bias_rescale_qkv_vulkan(const at::Tensor& qkv, const at::Tensor& qkv_bias, int64_t num_head)
{
    TORCH_CHECK(!qkv.is_nested(), "torchvulkan [ERROR]: nested tensors are not supported.");
    const int64_t B = qkv.size(0), T = qkv.size(1), D = qkv.size(2) / 3;
    TORCH_CHECK(D % num_head == 0);
    TORCH_CHECK(qkv.size(2) % 3 == 0);
    const int64_t dim_per_head = D / num_head;
    at::Tensor q_k_v = at::add(qkv, qkv_bias).view({B, T, 3, num_head, dim_per_head}).permute({2, 0, 3, 1, 4}).contiguous();
    at::Tensor q = at::div(q_k_v[0], std::sqrt(double(dim_per_head)));
    return {q, q_k_v[1], q_k_v[2]};
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("arange.start_out", &arange_start_out_vulkan);
    m.impl("linspace.out", &linspace_out_vulkan);
    m.impl("logspace.out", &logspace_out_vulkan);
    m.impl("eye.m_out", &eye_m_out_vulkan);
    m.impl("tril_indices", &tril_indices_vulkan);
    m.impl("triu_indices", &triu_indices_vulkan);
    m.impl("logit", &logit_vulkan);
    m.impl("gelu.out", &gelu_out_vulkan);
    m.impl("gelu_backward", &gelu_backward_vulkan);
    m.impl("silu", &silu_vulkan);
    m.impl("hardtanh", &hardtanh_vulkan);
    m.impl("hardsigmoid", &hardsigmoid_vulkan);
    m.impl("hardswish", &hardswish_vulkan);
    m.impl("leaky_relu", &leaky_relu_vulkan);
    m.impl("elu", &elu_vulkan);
    m.impl("softplus", &softplus_vulkan);
    m.impl("mish", &mish_vulkan);
    m.impl("logsumexp", &logsumexp_vulkan);
    m.impl("var.correction", &var_correction_vulkan);
    m.impl("std.correction", &std_correction_vulkan);
    m.impl("var_mean.correction", &var_mean_correction_vulkan);
    m.impl("std_mean.correction", &std_mean_correction_vulkan);
    m.impl("_log_softmax", &log_softmax_vulkan);
    m.impl("_log_softmax_backward_data", &log_softmax_backward_vulkan);
    m.impl("_softmax", &softmax_vulkan);
    m.impl("_softmax_backward_data", &softmax_backward_vulkan);
    m.impl("dot", &dot_vulkan);
    m.impl("vdot", &dot_vulkan);
    m.impl("addmv.out", &addmv_out_vulkan);
    m.impl("linalg_vector_norm.out", &linalg_vector_norm_out_vulkan);
    m.impl("norm.out", &norm_scalaropt_dim_out_vulkan);
    m.impl("renorm.out", &renorm_out_vulkan);
    m.impl("softshrink.out", &softshrink_out_vulkan);
    m.impl("hardshrink.out", &hardshrink_out_vulkan);
    m.impl("threshold.out", &threshold_out_vulkan);
    m.impl("glu.out", &glu_out_vulkan);
    m.impl("_prelu_kernel", &prelu_vulkan);
    m.impl("log_sigmoid_forward", &log_sigmoid_forward_vulkan);
    m.impl("huber_loss", &huber_loss_vulkan);
    m.impl("smooth_l1_loss.out", &smooth_l1_loss_out_vulkan);
    m.impl("mse_loss.out", &mse_loss_out_vulkan);
    m.impl("binary_cross_entropy", &binary_cross_entropy_vulkan);
    m.impl("soft_margin_loss", &soft_margin_loss_vulkan);
    m.impl("special_xlog1py.out", &special_xlog1py_out_vulkan);
    m.impl("trace", &trace_vulkan);
    m.impl("cat.out", &cat_out_vulkan);
    m.impl("roll", &roll_vulkan);
    m.impl("masked_fill_.Scalar", &masked_fill_scalar_vulkan_);
    m.impl("masked_fill_.Tensor", &masked_fill_tensor_vulkan_);
    m.impl("nan_to_num.out", &nan_to_num_out_vulkan);
    m.impl("heaviside.out", &heaviside_out_vulkan);
    m.impl("isnan", &isnan_vulkan);
    m.impl("_fused_sdp_choice", &fused_sdp_choice_vulkan);
    m.impl("native_layer_norm", &native_layer_norm_vulkan);
    m.impl("native_layer_norm_backward", &native_layer_norm_backward_vulkan);
    m.impl("native_group_norm", &native_group_norm_vulkan);
    m.impl("sigmoid_backward.grad_input", &sigmoid_backward_out_vulkan);
    m.impl("tanh_backward.grad_input", &tanh_backward_out_vulkan);
    m.impl("leaky_relu_backward.grad_input", &leaky_relu_backward_out_vulkan);
    m.impl("elu_backward.grad_input", &elu_backward_out_vulkan);
    m.impl("hardtanh_backward", &hardtanh_backward_vulkan);
    m.impl("hardtanh_backward.grad_input", &hardtanh_backward_out_vulkan);
    m.impl("softplus_backward.grad_input", &softplus_backward_out_vulkan);
    m.impl("hardsigmoid_backward.grad_input", &hardsigmoid_backward_out_vulkan);
    m.impl("hardswish_backward", &hardswish_backward_vulkan);
    m.impl("log_sigmoid_backward", &log_sigmoid_backward_vulkan);
    m.impl("log_sigmoid_backward.grad_input", &log_sigmoid_backward_out_vulkan);
    m.impl("mse_loss_backward", &mse_loss_backward_vulkan);
    m.impl("mse_loss_backward.grad_input", &mse_loss_backward_out_vulkan);
    m.impl("smooth_l1_loss_backward.grad_input", &smooth_l1_loss_backward_out_vulkan);
    m.impl("huber_loss_backward.out", &huber_loss_backward_out_vulkan);
    m.impl("binary_cross_entropy_backward", &binary_cross_entropy_backward_vulkan);
    m.impl("binary_cross_entropy_backward.grad_input", &binary_cross_entropy_backward_out_vulkan);
    m.impl("soft_margin_loss_backward.grad_input", &soft_margin_loss_backward_out_vulkan);
    m.impl("_prelu_kernel_backward", &prelu_backward_vulkan);
    m.impl("logit_backward.grad_input", &logit_backward_out_vulkan);
    m.impl("hardshrink_backward.grad_input", &hardshrink_backward_out_vulkan);
    m.impl("softshrink_backward.grad_input", &softshrink_backward_out_vulkan);
    m.impl("glu_backward", &glu_backward_vulkan);
    m.impl("glu_backward.grad_input", &glu_backward_out_vulkan);
    m.impl("addbmm", &addbmm_vulkan);
    m.impl("addbmm.out", &addbmm_out_vulkan);
    m.impl("addbmm_", &addbmm_vulkan_);
    m.impl("mm.out", &mm_out_vulkan);
    m.impl("mm.dtype", &mm_dtype_vulkan);
    m.impl("mm.dtype_out", &mm_dtype_out_vulkan);
    m.impl("_log_softmax.out", &log_softmax_out_vulkan);
    m.impl("_softmax.out", &softmax_out_vulkan);
    m.impl("_log_softmax_backward_data.out", &log_softmax_backward_out_vulkan);
    m.impl("_softmax_backward_data.out", &softmax_backward_out_vulkan);
    m.impl("binary_cross_entropy.out", &binary_cross_entropy_out_vulkan);
    m.impl("huber_loss.out", &huber_loss_out_vulkan);
    m.impl("elu.out", &elu_out_vulkan);
    m.impl("eye.out", &eye_out_vulkan);
    m.impl("gelu_backward.grad_input", &gelu_backward_out_vulkan);
    m.impl("hardsigmoid.out", &hardsigmoid_out_vulkan);
    m.impl("hardswish.out", &hardswish_out_vulkan);
    m.impl("hardswish_", &hardswish_vulkan_);
    m.impl("hardtanh.out", &hardtanh_out_vulkan);
    m.impl("hardtanh_", &hardtanh_vulkan_);
    m.impl("leaky_relu.out", &leaky_relu_out_vulkan);
    m.impl("log_sigmoid_forward.output", &log_sigmoid_forward_out_vulkan);
    m.impl("logit.out", &logit_out_vulkan);
    m.impl("logit_", &logit_vulkan_);
    m.impl("mish.out", &mish_out_vulkan);
    m.impl("silu.out", &silu_out_vulkan);
    m.impl("softplus.out", &softplus_out_vulkan);
    m.impl("norm.dtype_out", &norm_dtype_out_vulkan);
    m.impl("var.correction_out", &var_correction_out_vulkan);
    m.impl("std.correction_out", &std_correction_out_vulkan);
    m.impl("mvlgamma.out", &mvlgamma_out_vulkan);
    m.impl("narrow_copy.out", &narrow_copy_out_vulkan);
    m.impl("range.out", &range_out_vulkan);
    m.impl("is_set_to", &is_set_to_vulkan);
    m.impl("_assert_async", &assert_async_vulkan);
    m.impl("_assert_async.msg", &assert_async_msg_vulkan);
    m.impl("_functional_assert_async.msg", &functional_assert_async_msg_vulkan);
    m.impl("_make_dep_token", &make_dep_token_vulkan);
    m.impl("_amp_foreach_non_finite_check_and_unscale_", &amp_foreach_non_finite_check_and_unscale_vulkan_);
    m.impl("_amp_update_scale_", &amp_update_scale_vulkan_);
    m.impl("glu_jvp", &glu_jvp_vulkan);
    m.impl("glu_backward_jvp", &glu_backward_jvp_vulkan);
    m.impl("silu_backward.grad_input", &silu_backward_out_vulkan);
    m.impl("_compute_linear_combination", &compute_linear_combination_vulkan);
    m.impl("_compute_linear_combination.out", &compute_linear_combination_out_vulkan);
    m.impl("batch_norm_update_stats", &batch_norm_update_stats_vulkan);
    m.impl("_weight_norm_interface", &weight_norm_interface_vulkan);
    m.impl("_weight_norm_interface_backward", &weight_norm_interface_backward_vulkan);
    m.impl("_masked_softmax", &masked_softmax_vulkan);
    m.impl("_masked_softmax_backward", &masked_softmax_backward_vulkan);
    m.impl("_transform_bias_rescale_qkv", &transform_bias_rescale_qkv_vulkan);
    m.impl("_native_multi_head_attention", &at::native::native_multi_head_attention_cpu);
    m.impl("_transformer_encoder_layer_fwd", &at::native::transformer_encoder_layer_forward);
}
