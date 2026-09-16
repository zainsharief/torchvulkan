#include <torch/extension.h>
#include <ATen/WrapDimUtils.h>
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

bool is_low_prec(at::ScalarType t)
{
    return t == at::kHalf || t == at::kBFloat16;
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

at::Tensor reduce_loss(const at::Tensor& t, int64_t reduction)
{
    if (reduction == 1) return at::mean(t);
    if (reduction == 2) return at::sum(t);
    return t;
}

at::Tensor gt_mask(const at::Tensor& v)
{
    return at::clamp(at::sign(v), 0, 1);
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
    at::Tensor res = at::mul(at::matmul(m, v), a);
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
    at::Tensor factor = at::clamp_max(at::mul(at::reciprocal(at::add(norms, 1e-7)), maxnorm.toDouble()), 1.0);
    return fill_out(out, at::mul(self, factor));
}

at::Tensor& softshrink_out_vulkan(const at::Tensor& self, const at::Scalar& lambd, at::Tensor& out)
{
    at::Tensor res = at::mul(at::sign(self), at::clamp_min(at::sub(at::abs(self), lambd), 0));
    return fill_out(out, res);
}

at::Tensor& hardshrink_out_vulkan(const at::Tensor& self, const at::Scalar& lambd, at::Tensor& out)
{
    at::Tensor mask = gt_mask(at::sub(at::abs(self), lambd));
    return fill_out(out, at::mul(self, mask));
}

at::Tensor& threshold_out_vulkan(const at::Tensor& self, const at::Scalar& threshold, const at::Scalar& value, at::Tensor& out)
{
    at::Tensor s = at::isIntegralType(self.scalar_type(), /*includeBool=*/true) ? self.to(at::kFloat) : self;
    at::Tensor mask = gt_mask(at::sub(s, threshold));
    at::Tensor res = at::add(at::mul(mask, s), at::mul(at::rsub(mask, 1), value));
    return fill_out(out, res);
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
    return at::add(at::clamp_min(self, 0), at::mul(w, at::clamp_max(self, 0)));
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
    at::Tensor mask = gt_mask(at::sub(absd, delta));
    at::Tensor loss = at::add(at::mul(at::rsub(mask, 1), quad), at::mul(mask, lin));
    return reduce_loss(loss, reduction);
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
        at::Tensor mask = gt_mask(at::sub(absd, beta));
        loss = at::add(at::mul(at::rsub(mask, 1), quad), at::mul(mask, lin));
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

at::Tensor& erfc_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return fill_out(out, at::rsub(at::erf(self), 1));
}

at::Tensor& log_ndtr_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return fill_out(out, at::log(at::special_ndtr(self)));
}

at::Tensor& i0e_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    at::Tensor x = self.is_floating_point() ? self : self.to(at::kFloat);
    return fill_out(out, at::mul(at::exp(at::neg(at::abs(x))), at::i0(x)));
}

at::Tensor& i1e_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    at::Tensor x = self.is_floating_point() ? self : self.to(at::kFloat);
    return fill_out(out, at::mul(at::exp(at::neg(at::abs(x))), at::special_i1(x)));
}

at::Tensor& erfcx_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return fill_out(out, at::mul(at::exp(at::mul(self, self)), at::erfc(self)));
}

at::Tensor& lgamma_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return fill_out(out, at::lgamma(self));
}

at::Tensor& masked_fill_scalar_vulkan_(at::Tensor& self, const at::Tensor& mask, const at::Scalar& value)
{
    self.copy_(at::where(mask, at::scalar_tensor(value, self.options()), self));
    return self;
}

at::Tensor& masked_fill_tensor_vulkan_(at::Tensor& self, const at::Tensor& mask, const at::Tensor& value)
{
    self.copy_(at::where(mask, value, self));
    return self;
}

at::Tensor& nan_to_num_out_vulkan(const at::Tensor& self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf, at::Tensor& out)
{
    if (!self.is_floating_point()) { out.copy_(self); return out; }
    double type_max, type_min;
    switch (self.scalar_type()) {
        case at::kHalf: type_max = 65504.0; break;
        case at::kBFloat16: type_max = 3.38953139e38; break;
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
    at::Tensor mask = at::clamp(at::sign(at::sub(bx, threshold)), 0, 1);
    return at::add(at::mul(mask, self), at::mul(at::rsub(mask, 1), sp));
}

at::Tensor mish_vulkan(const at::Tensor& self)
{
    return at::mul(self, at::tanh(softplus_vulkan(self, 1, 20)));
}

at::Tensor tanhshrink_vulkan(const at::Tensor& self)
{
    return at::sub(self, at::tanh(self));
}

at::Tensor square_vulkan(const at::Tensor& self)
{
    if (self.scalar_type() == at::kBool) {
        at::Tensor t = self.to(at::kLong);
        return at::mul(t, t);
    }
    return at::mul(self, self);
}

at::Tensor logit_vulkan(const at::Tensor& self, c10::optional<double> eps)
{
    at::Tensor x = promote_to_float(self);
    if (eps.has_value()) x = at::clamp(x, *eps, 1.0 - *eps);
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
    return cpu.to(device.value_or(at::Device(at::DeviceType::PrivateUse1, 0)));
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
    return cpu.to(device.value_or(at::Device(at::DeviceType::PrivateUse1, 0)));
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
    m.impl("lgamma.out", &lgamma_out_vulkan);
    m.impl("gelu.out", &gelu_out_vulkan);
    m.impl("erfc.out", &erfc_out_vulkan);
    m.impl("special_i0e.out", &i0e_out_vulkan);
    m.impl("special_i1e.out", &i1e_out_vulkan);
    m.impl("silu", &silu_vulkan);
    m.impl("hardtanh", &hardtanh_vulkan);
    m.impl("hardsigmoid", &hardsigmoid_vulkan);
    m.impl("hardswish", &hardswish_vulkan);
    m.impl("leaky_relu", &leaky_relu_vulkan);
    m.impl("elu", &elu_vulkan);
    m.impl("softplus", &softplus_vulkan);
    m.impl("mish", &mish_vulkan);
    m.impl("tanhshrink", &tanhshrink_vulkan);
    m.impl("square", &square_vulkan);
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
}
