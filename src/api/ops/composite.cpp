#include <torch/extension.h>
#include <ATen/SDPBackend.h>
#include <ATen/WrapDimUtils.h>
#include <cmath>
#include <limits>
#include <vector>
#include "api/ops/composite.h"

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

at::Tensor& fill_out(at::Tensor& out, const at::Tensor& res)
{
    if (out.sizes() != res.sizes()) out.resize_(res.sizes());
    out.copy_(res);
    return out;
}

at::Tensor gt_mask(const at::Tensor& v)
{
    return at::clamp(at::sign(v), 0, 1);
}


at::Tensor torchvulkan::dot_vulkan(const at::Tensor& self, const at::Tensor& other)
{
    at::ScalarType odt = self.scalar_type();
    bool low = is_low_prec(odt);
    at::Tensor a = low ? self.to(at::kFloat) : self;
    at::Tensor b = low ? other.to(at::kFloat) : other;
    return at::sum(at::mul(a, b)).to(odt);
}

at::Tensor& torchvulkan::addmv_out_vulkan(const at::Tensor& self, const at::Tensor& mat, const at::Tensor& vec, const at::Scalar& beta, const at::Scalar& alpha, at::Tensor& out)
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

at::Tensor& torchvulkan::linalg_vector_norm_out_vulkan(const at::Tensor& self, const at::Scalar& ord, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor& out)
{
    at::Tensor x = dtype.has_value() ? self.to(*dtype) : self;
    return fill_out(out, p_norm(x, ord.toDouble(), dim, keepdim));
}

at::Tensor& torchvulkan::norm_scalaropt_dim_out_vulkan(const at::Tensor& self, const c10::optional<at::Scalar>& p, at::IntArrayRef dim, bool keepdim, at::Tensor& out)
{
    double pv = p.has_value() ? p->toDouble() : 2.0;
    return fill_out(out, p_norm(self, pv, dim, keepdim));
}

at::Tensor& torchvulkan::renorm_out_vulkan(const at::Tensor& self, const at::Scalar& p, int64_t dim, const at::Scalar& maxnorm, at::Tensor& out)
{
    int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    std::vector<int64_t> rdims;
    for (int64_t i = 0; i < self.dim(); i++) if (i != d) rdims.push_back(i);

    at::Tensor norms = p_norm(self, p.toDouble(), rdims, /*keepdim=*/true);
    at::Tensor factor = at::clamp_max(at::mul(at::reciprocal(at::add(norms, 1e-7)), maxnorm.toDouble()), 1.0);
    return fill_out(out, at::mul(self, factor));
}

at::Tensor& torchvulkan::softshrink_out_vulkan(const at::Tensor& self, const at::Scalar& lambd, at::Tensor& out)
{
    at::Tensor res = at::mul(at::sign(self), at::clamp_min(at::sub(at::abs(self), lambd), 0));
    return fill_out(out, res);
}

at::Tensor& torchvulkan::hardshrink_out_vulkan(const at::Tensor& self, const at::Scalar& lambd, at::Tensor& out)
{
    at::Tensor mask = gt_mask(at::sub(at::abs(self), lambd));
    return fill_out(out, at::mul(self, mask));
}

at::Tensor& torchvulkan::threshold_out_vulkan(const at::Tensor& self, const at::Scalar& threshold, const at::Scalar& value, at::Tensor& out)
{
    at::Tensor s = at::isIntegralType(self.scalar_type(), /*includeBool=*/true) ? self.to(at::kFloat) : self;
    at::Tensor mask = gt_mask(at::sub(s, threshold));
    at::Tensor res = at::add(at::mul(mask, s), at::mul(at::rsub(mask, 1), value));
    return fill_out(out, res);
}

at::Tensor& torchvulkan::glu_out_vulkan(const at::Tensor& self, int64_t dim, at::Tensor& out)
{
    int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    int64_t half = self.size(d) / 2;
    at::Tensor a = self.narrow(d, 0, half);
    at::Tensor b = self.narrow(d, half, half);
    return fill_out(out, at::mul(a, at::sigmoid(b)));
}

at::Tensor torchvulkan::prelu_vulkan(const at::Tensor& self, const at::Tensor& weight)
{
    at::Tensor w = weight;
    if (weight.numel() != 1 && self.dim() > 1) {
        std::vector<int64_t> vshape(self.dim(), 1);
        vshape[1] = weight.numel();
        w = weight.view(vshape);
    }
    return at::add(at::clamp_min(self, 0), at::mul(w, at::clamp_max(self, 0)));
}

std::tuple<at::Tensor, at::Tensor> torchvulkan::log_sigmoid_forward_vulkan(const at::Tensor& self)
{
    at::Tensor output = at::neg(at::softplus(at::neg(self), 1, 20));
    at::Tensor buffer = at::empty({0}, self.options());
    return std::make_tuple(output, buffer);
}

at::Tensor torchvulkan::huber_loss_vulkan(const at::Tensor& self, const at::Tensor& target, int64_t reduction, double delta)
{
    at::Tensor d = at::sub(self, target);
    at::Tensor absd = at::abs(d);
    at::Tensor quad = at::mul(at::mul(d, d), 0.5);
    at::Tensor lin = at::mul(at::sub(absd, 0.5 * delta), delta);
    at::Tensor mask = gt_mask(at::sub(absd, delta));
    at::Tensor loss = at::add(at::mul(at::rsub(mask, 1), quad), at::mul(mask, lin));
    return reduce_loss(loss, reduction);
}

at::Tensor& torchvulkan::smooth_l1_loss_out_vulkan(const at::Tensor& self, const at::Tensor& target, int64_t reduction, double beta, at::Tensor& out)
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

at::Tensor& torchvulkan::mse_loss_out_vulkan(const at::Tensor& self, const at::Tensor& target, int64_t reduction, at::Tensor& out)
{
    at::Tensor d = at::sub(self, target);
    return fill_out(out, reduce_loss(at::mul(d, d), reduction));
}

at::Tensor torchvulkan::binary_cross_entropy_vulkan(const at::Tensor& self, const at::Tensor& target, const c10::optional<at::Tensor>& weight, int64_t reduction)
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

at::Tensor torchvulkan::soft_margin_loss_vulkan(const at::Tensor& self, const at::Tensor& target, int64_t reduction)
{
    at::ScalarType odt = self.scalar_type();
    bool low = is_low_prec(odt);
    at::Tensor s = low ? self.to(at::kFloat) : self;
    at::Tensor t = low ? target.to(at::kFloat) : target;
    at::Tensor loss = at::softplus(at::neg(at::mul(t, s)), 1, 20);
    at::Tensor r = reduce_loss(loss, reduction);
    return low ? r.to(odt) : r;
}

at::Tensor& torchvulkan::special_xlog1py_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out)
{
    auto up = [](const at::Tensor& t) {
        return (t.is_floating_point() && !is_low_prec(t.scalar_type())) ? t : t.to(at::kFloat);
    };
    return fill_out(out, at::xlogy(up(self), at::add(up(other), 1)));
}

at::Tensor torchvulkan::trace_vulkan(const at::Tensor& self)
{
    return at::sum(at::diagonal(self));
}

at::Tensor& torchvulkan::gelu_out_vulkan(const at::Tensor& self, c10::string_view approximate, at::Tensor& out)
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

at::Tensor torchvulkan::gelu_backward_vulkan(const at::Tensor& grad_output, const at::Tensor& self, c10::string_view approximate)
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

at::Tensor& torchvulkan::erfc_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return fill_out(out, at::rsub(at::erf(self), 1));
}

at::Tensor& torchvulkan::log_ndtr_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return fill_out(out, at::log(at::special_ndtr(self)));
}

at::Tensor& torchvulkan::i0e_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    at::Tensor x = self.is_floating_point() ? self : self.to(at::kFloat);
    return fill_out(out, at::mul(at::exp(at::neg(at::abs(x))), at::i0(x)));
}

at::Tensor& torchvulkan::i1e_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    at::Tensor x = self.is_floating_point() ? self : self.to(at::kFloat);
    return fill_out(out, at::mul(at::exp(at::neg(at::abs(x))), at::special_i1(x)));
}

at::Tensor& torchvulkan::erfcx_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return fill_out(out, at::mul(at::exp(at::mul(self, self)), at::erfc(self)));
}

at::Tensor& torchvulkan::lgamma_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return fill_out(out, at::lgamma(self));
}

at::Tensor& torchvulkan::masked_fill_scalar_vulkan_(at::Tensor& self, const at::Tensor& mask, const at::Scalar& value)
{
    self.copy_(at::where(mask, at::scalar_tensor(value, self.options()), self));
    return self;
}

at::Tensor& torchvulkan::masked_fill_tensor_vulkan_(at::Tensor& self, const at::Tensor& mask, const at::Tensor& value)
{
    self.copy_(at::where(mask, value, self));
    return self;
}

at::Tensor& torchvulkan::nan_to_num_out_vulkan(const at::Tensor& self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf, at::Tensor& out)
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

at::Tensor& torchvulkan::heaviside_out_vulkan(const at::Tensor& self, const at::Tensor& values, at::Tensor& out)
{
    at::Tensor one = at::scalar_tensor(1, self.options());
    at::Tensor zero = at::scalar_tensor(0, self.options());
    at::Tensor r = at::where(at::gt(self, 0), one, values);
    r = at::where(at::lt(self, 0), zero, r);
    if (self.is_floating_point()) r = at::where(at::isnan(self), self, r);
    out.copy_(r);
    return out;
}

at::Tensor& torchvulkan::cat_out_vulkan(const at::ITensorListRef& tensors, int64_t dim, at::Tensor& out)
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

at::Tensor torchvulkan::roll_vulkan(const at::Tensor& self, at::IntArrayRef shifts, at::IntArrayRef dims)
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

at::Tensor& torchvulkan::clamp_min_out_vulkan(const at::Tensor& self, const at::Scalar& min, at::Tensor& out)
{
    return fill_out(out, at::clamp_min(self, min));
}

at::Tensor& torchvulkan::clamp_max_out_vulkan(const at::Tensor& self, const at::Scalar& max, at::Tensor& out)
{
    return fill_out(out, at::clamp_max(self, max));
}

at::Tensor& torchvulkan::clamp_min_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& min, at::Tensor& out)
{
    return fill_out(out, at::maximum(self, min));
}

at::Tensor& torchvulkan::clamp_max_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& max, at::Tensor& out)
{
    return fill_out(out, at::minimum(self, max));
}

at::Tensor& torchvulkan::clamp_out_vulkan(const at::Tensor& self, const c10::optional<at::Scalar>& min, const c10::optional<at::Scalar>& max, at::Tensor& out)
{
    at::Tensor res = self;
    if (min.has_value()) res = at::clamp_min(res, *min);
    if (max.has_value()) res = at::clamp_max(res, *max);
    return fill_out(out, res);
}

at::Tensor& torchvulkan::clamp_tensor_out_vulkan(const at::Tensor& self, const c10::optional<at::Tensor>& min, const c10::optional<at::Tensor>& max, at::Tensor& out)
{
    at::Tensor res = self;
    if (min.has_value()) res = at::maximum(res, *min);
    if (max.has_value()) res = at::minimum(res, *max);
    return fill_out(out, res);
}

int64_t torchvulkan::fused_sdp_choice_vulkan(
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

at::Tensor torchvulkan::embedding_vulkan(
    const at::Tensor& weight,
    const at::Tensor& indices,
    c10::SymInt /* padding_idx */,
    bool scale_grad_by_freq,
    bool sparse)
{
    TORCH_CHECK(!sparse, "torchvulkan [NOT IMPLEMENTED]: sparse embedding gradients are not supported.");
    TORCH_CHECK(!scale_grad_by_freq, "torchvulkan [NOT IMPLEMENTED]: scale_grad_by_freq is not supported.");

    int64_t num_embeddings = weight.size(0);
    at::Tensor flat_idx = indices.reshape({-1});
    at::Tensor vocab = at::arange(num_embeddings, flat_idx.options());
    at::Tensor one_hot = at::eq(flat_idx.unsqueeze(1), vocab).to(weight.scalar_type());
    at::Tensor out = at::matmul(one_hot, weight);

    std::vector<int64_t> out_shape(indices.sizes().begin(), indices.sizes().end());
    out_shape.push_back(weight.size(1));
    return out.reshape(out_shape);
}

at::Tensor torchvulkan::embedding_dense_backward_vulkan(
    const at::Tensor& grad_output,
    const at::Tensor& indices,
    c10::SymInt num_weights,
    c10::SymInt padding_idx,
    bool scale_grad_by_freq)
{
    TORCH_CHECK(!scale_grad_by_freq, "torchvulkan [NOT IMPLEMENTED]: scale_grad_by_freq is not supported.");

    int64_t num_weights_ = num_weights.guard_int(__FILE__, __LINE__);
    int64_t padding_idx_ = padding_idx.guard_int(__FILE__, __LINE__);

    at::Tensor flat_idx = indices.reshape({-1});
    at::Tensor flat_grad = grad_output.reshape({flat_idx.size(0), grad_output.size(-1)});
    at::Tensor vocab = at::arange(num_weights_, flat_idx.options());
    at::Tensor one_hot = at::eq(flat_idx.unsqueeze(1), vocab);

    if (padding_idx_ >= 0) {
        at::Tensor keep = at::ne(flat_idx, padding_idx_).unsqueeze(1);
        one_hot = at::logical_and(one_hot, keep);
    }

    return at::matmul(one_hot.to(grad_output.scalar_type()).transpose(0, 1), flat_grad);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> torchvulkan::native_layer_norm_vulkan(
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

    at::Tensor mean = at::mean(x, reduce_dims, /*keepdim=*/true);
    at::Tensor var = at::var(x, reduce_dims, /*unbiased=*/false, /*keepdim=*/true);
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

std::tuple<at::Tensor, at::Tensor, at::Tensor> torchvulkan::native_layer_norm_backward_vulkan(
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

at::Tensor& torchvulkan::normal_vulkan_(at::Tensor& self, double mean, double stddev, c10::optional<at::Generator> generator)
{
    at::Tensor cpu = at::empty(self.sizes(), self.options().device(at::kCPU));
    cpu.normal_(mean, stddev, generator);
    self.copy_(cpu);
    return self;
}

at::Tensor& torchvulkan::uniform_vulkan_(at::Tensor& self, double from, double to, c10::optional<at::Generator> generator)
{
    at::Tensor cpu = at::empty(self.sizes(), self.options().device(at::kCPU));
    cpu.uniform_(from, to, generator);
    self.copy_(cpu);
    return self;
}
