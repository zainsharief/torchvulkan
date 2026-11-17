#include <torch/extension.h>
#include <ATen/ExpandUtils.h>
#include <ATen/ops/view_as_complex_native.h>
#include <ATen/ops/view_as_real_native.h>
#include <cmath>
#include <vector>
#include "vulkan/vulkan_context.h"
#include "api/ops/helpers.h"
#include "api/ops/internal.h"

namespace torchvulkan {

ComplexParts complex_parts(const at::Tensor& z)
{
    at::Tensor w = z.resolve_conj().resolve_neg();
    if (!w.is_complex()) return {w, at::zeros_like(w)};
    if (w.scalar_type() == at::kComplexHalf) w = w.to(at::kComplexFloat);
    at::Tensor pairs = at::view_as_real(w);
    return {pairs.select(-1, 0), pairs.select(-1, 1)};
}

at::Tensor complex_from(const ComplexParts& z, at::ScalarType dtype)
{
    std::vector<int64_t> shape = at::infer_size(z.re.sizes(), z.im.sizes());
    at::Tensor out = at::complex(z.re.expand(shape), z.im.expand(shape));
    return out.scalar_type() == dtype ? out : out.to(dtype);
}

ComplexParts complex_scalar_parts(const at::Scalar& s, const at::Tensor& like)
{
    c10::complex<double> v = s.toComplexDouble();
    return {at::full({}, v.real(), like.options()), at::full({}, v.imag(), like.options())};
}

namespace {

at::Tensor copysign_of(const at::Tensor& magnitude, const at::Tensor& sign) { return at::copysign(magnitude, sign); }
at::Tensor full_like_of(const at::Tensor& like, double v) { return at::full_like(like, v); }
at::Tensor nonfinite(const at::Tensor& t) { return at::logical_not(at::isfinite(t)); }

} // namespace

ComplexParts complex_mul(const ComplexParts& a, const ComplexParts& b)
{
    return {at::sub(at::mul(a.re, b.re), at::mul(a.im, b.im)), at::add(at::mul(a.re, b.im), at::mul(a.im, b.re))};
}

ComplexParts complex_div(const ComplexParts& a, const ComplexParts& b)
{
    at::Tensor abs_c = at::abs(b.re), abs_d = at::abs(b.im);
    // |c| >= |d|; c10 multiplies by 1 / (c + d rat), divided here instead so a GPU that flushes subnormals keeps tiny quotients
    at::Tensor rat1 = at::div(b.im, b.re);
    at::Tensor den1 = at::add(b.re, at::mul(b.im, rat1));
    at::Tensor re1 = at::div(at::add(a.re, at::mul(a.im, rat1)), den1);
    at::Tensor im1 = at::div(at::sub(a.im, at::mul(a.re, rat1)), den1);
    // both zero: a complex inf or nan
    at::Tensor zero = at::logical_and(at::eq(abs_c, 0), at::eq(abs_d, 0));
    re1 = at::where(zero, at::div(a.re, abs_c), re1);
    im1 = at::where(zero, at::div(a.im, abs_d), im1);
    // |c| < |d| (and NaNs)
    at::Tensor rat2 = at::div(b.re, b.im);
    at::Tensor den2 = at::add(b.im, at::mul(b.re, rat2));
    at::Tensor re2 = at::div(at::add(at::mul(a.re, rat2), a.im), den2);
    at::Tensor im2 = at::div(at::sub(at::mul(a.im, rat2), a.re), den2);
    at::Tensor first = at::ge(abs_c, abs_d);
    return {at::where(first, re1, re2), at::where(first, im1, im2)};
}

at::Tensor complex_abs(const ComplexParts& z)
{
    return at::hypot(z.re, z.im);
}

ComplexParts complex_exp(const ComplexParts& z)
{
    at::Tensor ex = at::exp(z.re);
    at::Tensor re = at::mul(ex, at::cos(z.im)), im = at::mul(ex, at::sin(z.im));
    // libc++: an imaginary zero stays exact; -inf + i(non-finite) is 0
    at::Tensor real_axis = at::eq(z.im, 0);
    re = at::where(real_axis, ex, re);
    im = at::where(real_axis, z.im, im);
    at::Tensor vanish = at::logical_and(at::isneginf(z.re), nonfinite(z.im));
    re = at::where(vanish, at::zeros_like(re), re);
    im = at::where(vanish, at::zeros_like(im), im);
    at::Tensor blowup = at::logical_and(at::isposinf(z.re), nonfinite(z.im));
    re = at::where(blowup, z.re, re);
    im = at::where(blowup, full_like_of(im, std::numeric_limits<double>::quiet_NaN()), im);
    return {re, im};
}

ComplexParts complex_log(const ComplexParts& z)
{
    return {at::log(complex_abs(z)), at::atan2(z.im, z.re)};
}

ComplexParts complex_sqrt(const ComplexParts& z)
{
    // libc++: polar(sqrt(|z|), arg(z) / 2), with its infinities
    at::Tensor r = at::sqrt(complex_abs(z));
    at::Tensor t = at::div(at::atan2(z.im, z.re), 2);
    at::Tensor re = at::mul(r, at::cos(t)), im = at::mul(r, at::sin(t));
    at::Tensor zero = at::zeros_like(re);
    at::Tensor inf_re_pos = at::isposinf(z.re), inf_re_neg = at::isneginf(z.re);
    at::Tensor im_nan = at::isnan(z.im);
    re = at::where(inf_re_pos, z.re, re);
    im = at::where(inf_re_pos, at::where(im_nan, z.im, copysign_of(zero, z.im)), im);
    re = at::where(inf_re_neg, at::where(im_nan, z.im, zero), re);
    im = at::where(inf_re_neg, copysign_of(z.re, z.im), im);
    at::Tensor inf_im = at::isinf(z.im);
    re = at::where(inf_im, full_like_of(re, std::numeric_limits<double>::infinity()), re);
    im = at::where(inf_im, z.im, im);
    return {re, im};
}

ComplexParts complex_sinh(const ComplexParts& z)
{
    ComplexParts r{at::mul(at::sinh(z.re), at::cos(z.im)), at::mul(at::cosh(z.re), at::sin(z.im))};
    at::Tensor nan = full_like_of(r.re, std::numeric_limits<double>::quiet_NaN());
    at::Tensor im_nonfinite = nonfinite(z.im);
    at::Tensor a = at::logical_and(at::logical_or(at::isinf(z.re), at::eq(z.re, 0)), im_nonfinite);
    at::Tensor b = at::logical_and(at::eq(z.im, 0), nonfinite(z.re));
    r.re = at::where(a, z.re, at::where(b, z.re, r.re));
    r.im = at::where(a, nan, at::where(b, z.im, r.im));
    return r;
}

ComplexParts complex_cosh(const ComplexParts& z)
{
    ComplexParts r{at::mul(at::cosh(z.re), at::cos(z.im)), at::mul(at::sinh(z.re), at::sin(z.im))};
    at::Tensor nan = full_like_of(r.re, std::numeric_limits<double>::quiet_NaN());
    at::Tensor im_nonfinite = nonfinite(z.im), re_zero = at::eq(z.re, 0);
    at::Tensor a = at::logical_and(at::isinf(z.re), im_nonfinite);
    at::Tensor b = at::logical_and(re_zero, im_nonfinite);
    at::Tensor c = at::logical_and(re_zero, at::eq(z.im, 0));
    at::Tensor d = at::logical_and(at::eq(z.im, 0), nonfinite(z.re));
    r.re = at::where(a, at::abs(z.re), at::where(b, nan, at::where(c, at::ones_like(r.re), at::where(d, at::abs(z.re), r.re))));
    r.im = at::where(a, nan, at::where(b, z.re, at::where(c, z.im, at::where(d, z.im, r.im))));
    return r;
}

ComplexParts complex_tanh(const ComplexParts& z)
{
    // Kahan's form, as glibc and NumPy compute it: no cancellation near x = 0, y = pi/2. Past |x| = 9 (22 for float64)
    // tanh(x) is 1 to the last bit, so the limit (sign(x), 4 sin(y) cos(y) e^(-2|x|)) takes over before beta s^2 overflows
    at::Tensor t = at::tan(z.im);
    at::Tensor beta = at::add(at::mul(t, t), 1);
    at::Tensor sh = at::sinh(z.re);
    at::Tensor rho = at::sqrt(at::add(at::mul(sh, sh), 1));
    at::Tensor denominator = at::add(at::mul(beta, at::mul(sh, sh)), 1);
    at::Tensor re = at::div(at::mul(at::mul(beta, rho), sh), denominator), im = at::div(t, denominator);
    at::Tensor zero = at::zeros_like(re), one = at::ones_like(re);
    double limit = z.re.scalar_type() == at::kDouble ? 22.0 : 9.0;
    at::Tensor large = at::logical_and(at::gt(at::abs(z.re), limit), at::isfinite(z.im));
    at::Tensor tail = at::mul(at::mul(at::mul(at::sin(z.im), at::cos(z.im)), 4), at::exp(at::mul(at::abs(z.re), -2)));
    re = at::where(large, copysign_of(one, z.re), re);
    im = at::where(large, tail, im);
    at::Tensor two_im = at::mul(z.im, 2);
    // NaN + 0i stays
    at::Tensor nan_real = at::logical_and(at::isnan(z.re), at::eq(z.im, 0));
    re = at::where(nan_real, z.re, re);
    im = at::where(nan_real, z.im, im);
    // infinite real part
    at::Tensor inf_re = at::isinf(z.re);
    re = at::where(inf_re, copysign_of(one, z.re), re);
    im = at::where(inf_re, at::where(nonfinite(z.im), zero, copysign_of(zero, at::sin(two_im))), im);
    return {re, im};
}

// c10's expm1: Re = expm1(x) cos(y) - 2 sin(y/2)^2, Im = e^x sin(y)
ComplexParts complex_expm1(const ComplexParts& z)
{
    at::Tensor half_sin = at::sin(at::div(z.im, 2));
    return {at::sub(at::mul(at::expm1(z.re), at::cos(z.im)), at::mul(at::mul(half_sin, half_sin), 2)), at::mul(at::exp(z.re), at::sin(z.im))};
}

// c10's log1p: near 0 from x (2 + x) + y^2, else log|1 + z|
ComplexParts complex_log1p(const ComplexParts& z)
{
    at::Tensor theta = at::atan2(z.im, at::add(z.re, 1));
    at::Tensor r = at::add(at::mul(z.re, at::add(z.re, 2)), at::mul(z.im, z.im));
    at::Tensor near = at::where(at::eq(r, 0), z.re, at::mul(at::log1p(r), 0.5));
    at::Tensor far = at::log(at::hypot(at::add(z.re, 1), z.im));
    return {at::where(at::lt(complex_abs(z), 0.5), near, far), theta};
}

// i z and -i z
ComplexParts times_i(const ComplexParts& z) { return {at::neg(z.im), z.re}; }
ComplexParts times_minus_i(const ComplexParts& z) { return {z.im, at::neg(z.re)}; }

ComplexParts complex_sin(const ComplexParts& z) { return times_minus_i(complex_sinh(times_i(z))); }
ComplexParts complex_cos(const ComplexParts& z) { return complex_cosh(times_i(z)); }
ComplexParts complex_tan(const ComplexParts& z) { return times_minus_i(complex_tanh(times_i(z))); }

ComplexParts complex_square(const ComplexParts& z)
{
    return {at::mul(at::sub(z.re, z.im), at::add(z.re, z.im)), at::mul(at::mul(z.re, z.im), 2)};
}

ComplexParts complex_asinh(const ComplexParts& z)
{
    ComplexParts sq = complex_square(z);
    ComplexParts inner = complex_sqrt({at::add(sq.re, 1), sq.im});
    ComplexParts l = complex_log({at::add(z.re, inner.re), at::add(z.im, inner.im)});
    at::Tensor re = copysign_of(l.re, z.re), im = copysign_of(l.im, z.im);
    // libc++'s non-finite cases
    at::Tensor zero = at::zeros_like(re);
    at::Tensor inf_re = at::isinf(z.re), nan_re = at::isnan(z.re), inf_im = at::isinf(z.im), nan_im = at::isnan(z.im);
    re = at::where(inf_im, copysign_of(z.im, z.re), re);
    im = at::where(inf_im, copysign_of(full_like_of(im, M_PI / 2), z.im), im);
    at::Tensor nan_re_inf_im = at::logical_and(nan_re, inf_im), nan_re_zero_im = at::logical_and(nan_re, at::eq(z.im, 0));
    re = at::where(nan_re, at::where(nan_re_inf_im, z.im, z.re), re);
    im = at::where(nan_re, at::where(nan_re_inf_im, z.re, at::where(nan_re_zero_im, z.im, z.re)), im);
    re = at::where(inf_re, z.re, re);
    im = at::where(inf_re, at::where(nan_im, z.im, at::where(inf_im, copysign_of(full_like_of(im, M_PI / 4), z.im), copysign_of(zero, z.im))), im);
    return {re, im};
}

// log(z + sqrt(z^2 - 1)), the core of libc++'s acos and acosh
ComplexParts acosh_core(const ComplexParts& z)
{
    ComplexParts sq = complex_square(z);
    ComplexParts root = complex_sqrt({at::sub(sq.re, 1), sq.im});
    return complex_log({at::add(z.re, root.re), at::add(z.im, root.im)});
}

ComplexParts complex_acosh(const ComplexParts& z)
{
    ComplexParts l = acosh_core(z);
    at::Tensor re = at::abs(l.re), im = copysign_of(l.im, z.im);
    // libc++'s non-finite cases, innermost first
    at::Tensor pi = full_like_of(re, M_PI);
    at::Tensor inf_re = at::isinf(z.re), nan_re = at::isnan(z.re), inf_im = at::isinf(z.im), nan_im = at::isnan(z.im);
    re = at::where(inf_im, at::abs(z.im), re);
    im = at::where(inf_im, copysign_of(at::div(pi, 2), z.im), im);
    re = at::where(nan_re, at::where(inf_im, at::abs(z.im), z.re), re);
    im = at::where(nan_re, z.re, im);
    at::Tensor positive = at::gt(z.re, 0);
    at::Tensor inf_re_re = at::where(nan_im, at::abs(z.re), at::where(inf_im, at::abs(z.re), at::abs(z.re)));
    at::Tensor inf_re_im = at::where(nan_im, z.im, at::where(inf_im, copysign_of(at::where(positive, at::mul(pi, 0.25), at::mul(pi, 0.75)), z.im),
                                                             at::where(positive, copysign_of(at::zeros_like(im), z.im), copysign_of(pi, z.im))));
    re = at::where(inf_re, inf_re_re, re);
    im = at::where(inf_re, inf_re_im, im);
    return {re, im};
}

ComplexParts complex_atanh(const ComplexParts& z)
{
    // libc++: log((1 + z) / (1 - z)) / 2, real part signed like z's
    ComplexParts q = complex_div({at::add(z.re, 1), z.im}, {at::rsub(z.re, 1), at::neg(z.im)});
    ComplexParts l = complex_log(q);
    at::Tensor re = copysign_of(at::div(l.re, 2), z.re), im = copysign_of(at::div(l.im, 2), z.im);
    // the branch points ±1 + 0i
    at::Tensor pole = at::logical_and(at::eq(at::abs(z.re), 1), at::eq(z.im, 0));
    re = at::where(pole, copysign_of(full_like_of(re, std::numeric_limits<double>::infinity()), z.re), re);
    im = at::where(pole, copysign_of(at::zeros_like(im), z.im), im);
    // C99 Annex G: an infinite part sends the result to (+-0, +-pi/2) (NaN imaginary for inf + i NaN)
    at::Tensor zero = copysign_of(at::zeros_like(re), z.re), half_pi = copysign_of(full_like_of(im, M_PI / 2), z.im);
    at::Tensor inf_im = at::isinf(z.im), inf_re = at::logical_and(at::isinf(z.re), at::logical_not(inf_im));
    re = at::where(at::logical_or(inf_im, inf_re), zero, re);
    im = at::where(inf_im, half_pi, at::where(inf_re, at::where(at::isnan(z.im), z.im, half_pi), im));
    at::Tensor zero_nan = at::logical_and(at::eq(z.re, 0), at::isnan(z.im));
    re = at::where(zero_nan, z.re, re);
    return {re, im};
}

ComplexParts complex_asin(const ComplexParts& z) { return times_minus_i(complex_asinh(times_i(z))); }
ComplexParts complex_atan(const ComplexParts& z) { return times_minus_i(complex_atanh(times_i(z))); }

ComplexParts complex_acos(const ComplexParts& z)
{
    ComplexParts l = acosh_core(z);
    at::Tensor negative_im = at::signbit(z.im);
    at::Tensor re = at::abs(l.im), im = at::where(negative_im, at::abs(l.re), at::neg(at::abs(l.re)));
    // libc++'s non-finite and imaginary-axis cases, innermost first
    at::Tensor pi = full_like_of(re, M_PI);
    at::Tensor inf_re = at::isinf(z.re), nan_re = at::isnan(z.re), inf_im = at::isinf(z.im), nan_im = at::isnan(z.im);
    at::Tensor axis = at::logical_and(at::eq(z.re, 0), at::logical_or(at::eq(z.im, 0), nan_im));
    re = at::where(axis, at::div(pi, 2), re);
    im = at::where(axis, at::neg(z.im), im);
    re = at::where(inf_im, at::div(pi, 2), re);
    im = at::where(inf_im, at::neg(z.im), im);
    re = at::where(nan_re, z.re, re);
    im = at::where(nan_re, at::where(inf_im, at::neg(z.im), z.re), im);
    at::Tensor negative_re = at::lt(z.re, 0);
    at::Tensor inf_re_re = at::where(nan_im, z.im, at::where(inf_im, at::where(negative_re, at::mul(pi, 0.75), at::mul(pi, 0.25)),
                                                             at::where(negative_re, pi, at::zeros_like(re))));
    at::Tensor inf_re_im = at::where(nan_im, z.re, at::where(inf_im, at::neg(z.im),
                                                             at::where(negative_re, at::where(negative_im, at::neg(z.re), z.re),
                                                                       at::where(negative_im, z.re, at::neg(z.re)))));
    re = at::where(inf_re, inf_re_re, re);
    im = at::where(inf_re, inf_re_im, im);
    return {re, im};
}

// c10's logaddexp: log1p(exp(min - max)) + max, min and max by real part, with its NaN and infinity cases
ComplexParts complex_logaddexp(const ComplexParts& x, const ComplexParts& y)
{
    at::Tensor x_min = at::lt(x.re, y.re);
    ComplexParts lo{at::where(x_min, x.re, y.re), at::where(x_min, x.im, y.im)};
    ComplexParts hi{at::where(x_min, y.re, x.re), at::where(x_min, y.im, x.im)};
    ComplexParts e = complex_exp({at::sub(lo.re, hi.re), at::sub(lo.im, hi.im)});
    ComplexParts l = complex_log1p(e);
    ComplexParts r{at::add(l.re, hi.re), at::add(l.im, hi.im)};
    // both +inf: log(exp(min) + exp(max)); both -inf: min
    ComplexParts el = complex_exp(lo), eh = complex_exp(hi);
    ComplexParts both_pos = complex_log({at::add(el.re, eh.re), at::add(el.im, eh.im)});
    at::Tensor same_inf = at::logical_and(at::isinf(lo.re), at::eq(lo.re, hi.re));
    at::Tensor neg = at::logical_and(same_inf, at::lt(lo.re, 0)), pos = at::logical_and(same_inf, at::gt(lo.re, 0));
    r.re = at::where(neg, lo.re, at::where(pos, both_pos.re, r.re));
    r.im = at::where(neg, lo.im, at::where(pos, both_pos.im, r.im));
    at::Tensor nan = at::logical_or(at::isnan(lo.re), at::isnan(lo.im));
    at::Tensor quiet = at::full_like(r.re, std::numeric_limits<double>::quiet_NaN());
    return {at::where(nan, quiet, r.re), at::where(nan, quiet, r.im)};
}

} // namespace torchvulkan


namespace {

at::Tensor& fill_complex(at::Tensor& out, const at::Tensor& real, const at::Tensor& imag)
{
    TORCH_CHECK(out.is_privateuseone(), "Expected out tensor to have device ", real.device(), ", but got ", out.device(), " instead");
    at::ScalarType expected = c10::toComplexType(real.scalar_type());
    TORCH_CHECK(out.scalar_type() == expected, "Expected object of scalar type ", expected, " but got scalar type ", out.scalar_type(), " for argument 'out'");
    std::vector<int64_t> shape = at::infer_size(real.sizes(), imag.sizes());
    resize_output_like(out, shape, {real, imag});
    at::Tensor pairs = at::view_as_real(out);
    pairs.select(-1, 0).copy_(real.expand(shape));
    pairs.select(-1, 1).copy_(imag.expand(shape));
    return out;
}

at::Tensor& complex_out_vulkan(const at::Tensor& real, const at::Tensor& imag, at::Tensor& out)
{
    TORCH_CHECK(real.scalar_type() == imag.scalar_type(), "Expected object of scalar type ", real.scalar_type(), " but got scalar type ", imag.scalar_type(),
        " for second argument"
    );
    TORCH_CHECK(real.scalar_type() == at::kFloat || real.scalar_type() == at::kDouble || real.scalar_type() == at::kHalf,
        "Expected both inputs to be Half, Float or Double tensors but got ", real.scalar_type(), " and ", imag.scalar_type()
    );
    return fill_complex(out, real, imag);
}

at::Tensor& polar_out_vulkan(const at::Tensor& abs, const at::Tensor& angle, at::Tensor& out)
{
    TORCH_CHECK(abs.scalar_type() == angle.scalar_type(), "Expected object of scalar type ", abs.scalar_type(), " but got scalar type ", angle.scalar_type(),
        " for second argument"
    );
    return fill_complex(out, at::mul(abs, at::cos(angle)), at::mul(abs, at::sin(angle)));
}

at::Tensor angle_vulkan(const at::Tensor& self)
{
    if (self.is_complex()) {
        at::Tensor pairs = at::view_as_real(self.resolve_conj());
        return at::atan2(pairs.select(-1, 1), pairs.select(-1, 0));
    }
    at::Tensor x = promote_to_float(self);
    at::Tensor result = at::where(at::lt(x, 0), at::full_like(x, M_PI), at::zeros_like(x));
    return at::where(at::isnan(x), x, result);
}

at::Tensor& angle_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return fill_out(out, angle_vulkan(self));
}

at::Tensor& conj_physical_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    if (!self.is_complex()) return fill_out(out, self);
    at::Tensor result = self.resolve_conj().clone();
    at::view_as_real(result).select(-1, 1).neg_();
    return fill_out(out, result);
}

// in place through conj_physical.out (torch's kernel is a CPU/CUDA stub)
at::Tensor& conj_physical_vulkan_(at::Tensor& self)
{
    if (!self.is_complex()) return self;
    return self.copy_(at::conj_physical(self));
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("view_as_real", &at::native::view_as_real);
    m.impl("view_as_complex", &at::native::view_as_complex);
    m.impl("complex.out", &complex_out_vulkan);
    m.impl("polar.out", &polar_out_vulkan);
    m.impl("angle", &angle_vulkan);
    m.impl("angle.out", &angle_out_vulkan);
    m.impl("conj_physical.out", &conj_physical_out_vulkan);
    m.impl("conj_physical_", &conj_physical_vulkan_);
}
