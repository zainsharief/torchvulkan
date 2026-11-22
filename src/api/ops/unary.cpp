#include <torch/extension.h>
#include <c10/core/MemoryFormat.h>
#include <functional>
#include <limits>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "api/ops/helpers.h"
#include "api/ops/internal.h"
#include "api/fallback.h"

namespace {

enum class UnaryOp {
    RELU = 0,
    EXP = 1,
    LOG = 2,
    SQRT = 3,
    NEG = 4,
    RECIPROCAL = 5,
    SIN = 6,
    COS = 7,
    TAN = 8,
    ASIN = 9,
    ACOS = 10,
    ATAN = 11,
    SINH = 12,
    COSH = 13,
    TANH = 14,
    EXP2 = 15,
    LOG2 = 16,
    LOG10 = 17,
    EXPM1 = 18,
    LOG1P = 19,
    RSQRT = 20,
    SIGMOID = 21,
    ASINH = 22,
    ACOSH = 23,
    ATANH = 24,
    DEG2RAD = 25,
    RAD2DEG = 26,
    FLOOR = 27,
    CEIL = 28,
    TRUNC = 29,
    ROUND = 30,
    FRAC = 31,
    ABS = 32,
    SIGN = 33,
    ERF = 34,
    ERFINV = 35,
    SINC = 36,
    ENTR = 37,
    LGAMMA = 38,
    DIGAMMA = 39,
    I0 = 40,
    I1 = 41,
    ERFC = 42,
    ERFCX = 43,
    LOG_NDTR = 44,
    I0E = 45,
    I1E = 46,
    GELU = 47,
    GELU_TANH = 48
};

enum class ComplexArcOp : uint32_t
{ 
    ASINH = 0, 
    ASIN = 1, 
    ACOS = 2, 
    ACOSH = 3, 
    ATANH = 4, 
    ATAN = 5 
};

at::Tensor unary_op_vulkan(
    const at::Tensor& src,
    UnaryOp operation,
    const std::function<at::Tensor(const at::Tensor&)>& fallback)
{
    DeviceContext* device = get_device_context(src);
    at::ScalarType dtype = src.scalar_type();
    bool supported = is_dtype_supported(device, dtype);

    at::Tensor dst = OutputTarget::take(dtype, src.sizes(), {src});
    if (!dst.defined()) dst = at::empty_like(src);
    at::TensorIterator iter = at::TensorIteratorConfig()
        .set_check_mem_overlap(true)
        .add_output(dst)
        .add_const_input(src)
        .build();

    uint64_t numel = iter.numel();
    if (numel == 0) return dst;

    int32_t out_dims = static_cast<int32_t>(iter.ndim());
    if (!supported || out_dims > MAX_DIMS || !fits_32bit_indexing(iter)) {
        TORCHVULKAN_FALLBACK("A unary op on ", dtype, " can't run on the Vulkan device. Falling back to CPU.");
        return fallback(src.cpu()).to(src.device());
    }

    uint32_t vecSize = get_dtype_vec_size(dtype);
    uint32_t workgroupSizeX = get_dtype_workgroup_size(dtype, vecSize, device);
    uint32_t contiguous = iter.is_contiguous() ? 1 : 0;

    SpecializationBuilder spd{};
    spd.push(static_cast<uint32_t>(operation))
        .push(contiguous)
        .push(out_dims)
        .push(workgroupSizeX);

    MetadataBuilder metadataBuilder = get_iter_metadata(iter, {1, 0});
    Metadata metadata = metadataBuilder.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(src))
        .push(get_output_address(dst))
        .push_metadata(contiguous ? nullptr : &metadata)
        .push(numel);

    uint64_t threads = contiguous ? (numel + (vecSize - 1)) / vecSize : numel;
    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::UNARY, dtype},
        device->default_subgroup_size,
        spd.build(),
        pcs.build(),
        /* read = */ {src},
        /* write = */ {dst},
        get_group_count(threads, workgroupSizeX), 1, 1
    );

    return dst;
}

at::Tensor complex_unary(const at::Tensor& z, torchvulkan::ComplexParts (*fn)(const torchvulkan::ComplexParts&))
{
    return torchvulkan::complex_from(fn(torchvulkan::complex_parts(z)), z.scalar_type());
}

at::Tensor complex_arc(const at::Tensor& self, ComplexArcOp op, torchvulkan::ComplexParts (*fn)(const torchvulkan::ComplexParts&))
{
    if (self.scalar_type() != at::kComplexFloat) return complex_unary(self, fn);
    at::Tensor input = self.resolve_conj().contiguous();
    at::Tensor out = at::empty(input.sizes(), input.options());
    uint64_t numel = static_cast<uint64_t>(input.numel());
    if (numel == 0) return out;

    DeviceContext* device = get_device_context(input);
    uint32_t workgroupSizeX = get_dtype_workgroup_size(at::kFloat, 1, device);

    SpecializationBuilder spd{};
    spd.push(static_cast<uint32_t>(op))
       .push(workgroupSizeX);

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(input))
       .push(get_output_address(out))
       .push(numel);

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::COMPLEX_ARC, at::kFloat},
        device->default_subgroup_size,
        spd.build(),
        pcs.build(),
        /* read = */ {input},
        /* write = */ {out},
        get_group_count(numel, workgroupSizeX), 1, 1
    );
    return out;
}

torchvulkan::ComplexParts complex_neg(const torchvulkan::ComplexParts& z)
{
    return {at::neg(z.re), at::neg(z.im)};
}

torchvulkan::ComplexParts complex_reciprocal(const torchvulkan::ComplexParts& z)
{
    torchvulkan::ComplexParts r = torchvulkan::complex_div({at::ones_like(z.re), at::zeros_like(z.im)}, z);
    at::Tensor zero = at::logical_and(at::eq(z.re, 0), at::eq(z.im, 0));
    return {at::where(zero, at::reciprocal(z.re), r.re), at::where(zero, at::full_like(r.im, std::numeric_limits<double>::quiet_NaN()), r.im)};
}

torchvulkan::ComplexParts complex_rsqrt(const torchvulkan::ComplexParts& z)
{
    torchvulkan::ComplexParts root = torchvulkan::complex_sqrt(z);
    return torchvulkan::complex_div({at::ones_like(root.re), at::zeros_like(root.im)}, root);
}

torchvulkan::ComplexParts complex_exp2(const torchvulkan::ComplexParts& z) 
{
    return torchvulkan::complex_exp({at::mul(z.re, M_LN2), at::mul(z.im, M_LN2)});
}

torchvulkan::ComplexParts complex_log_scaled(const torchvulkan::ComplexParts& z, double scale)
{
    torchvulkan::ComplexParts l = torchvulkan::complex_log(z);
    return {at::div(l.re, scale), at::div(l.im, scale)};
}

torchvulkan::ComplexParts complex_log2(const torchvulkan::ComplexParts& z)
{
    torchvulkan::ComplexParts l = torchvulkan::complex_log(z);
    return torchvulkan::complex_div(l, {at::full_like(l.re, M_LN2), at::zeros_like(l.im)});
}

torchvulkan::ComplexParts complex_log10(const torchvulkan::ComplexParts& z)
{
    return complex_log_scaled(z, M_LN10);
}

torchvulkan::ComplexParts complex_sigmoid(const torchvulkan::ComplexParts& z)
{
    torchvulkan::ComplexParts e = torchvulkan::complex_exp(complex_neg(z));
    return complex_reciprocal({at::add(e.re, 1), e.im});
}

torchvulkan::ComplexParts complex_sgn(const torchvulkan::ComplexParts& z)
{
    at::Tensor a = torchvulkan::complex_abs(z);
    torchvulkan::ComplexParts q = torchvulkan::complex_div(z, {a, at::zeros_like(a)});
    at::Tensor zero = at::eq(a, 0);
    return {at::where(zero, at::zeros_like(z.re), q.re), at::where(zero, at::zeros_like(z.im), q.im)};
}

at::Tensor gelu_vulkan(const at::Tensor& self, c10::string_view approximate)
{
    auto fallback = [approximate](const at::Tensor& a) { return at::gelu(a, approximate); };
    if (!self.is_floating_point() || (approximate != "none" && approximate != "tanh")) return fallback(self.cpu()).to(self.device());
    return unary_op_vulkan(self, approximate == "tanh" ? UnaryOp::GELU_TANH : UnaryOp::GELU, fallback);
}

at::Tensor relu_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(self, UnaryOp::RELU, [](const at::Tensor& a) { return at::relu(a); });
}

at::Tensor exp_vulkan(const at::Tensor& self)
{
    if (self.is_complex()) return complex_unary(self, torchvulkan::complex_exp);
    return unary_op_vulkan(promote_to_float(self), UnaryOp::EXP, [](const at::Tensor& a) { return at::exp(a); });
}

at::Tensor log_vulkan(const at::Tensor& self)
{
    if (self.is_complex()) return complex_unary(self, torchvulkan::complex_log);
    return unary_op_vulkan(promote_to_float(self), UnaryOp::LOG, [](const at::Tensor& a) { return at::log(a); });
}

at::Tensor sqrt_vulkan(const at::Tensor& self)
{
    if (self.is_complex()) return complex_unary(self, torchvulkan::complex_sqrt);
    return unary_op_vulkan(promote_to_float(self), UnaryOp::SQRT, [](const at::Tensor& a) { return at::sqrt(a); });
}

at::Tensor neg_vulkan(const at::Tensor& self)
{
    TORCH_CHECK(self.scalar_type() != at::kBool, "Negation, the `-` operator, on a bool tensor is not supported. "
        "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");
    if (self.is_complex()) return complex_unary(self, complex_neg);
    return unary_op_vulkan(self, UnaryOp::NEG, [](const at::Tensor& a) { return at::neg(a); });
}

at::Tensor reciprocal_vulkan(const at::Tensor& self)
{
    if (self.is_complex()) return complex_unary(self, complex_reciprocal);
    return unary_op_vulkan(promote_to_float(self), UnaryOp::RECIPROCAL, [](const at::Tensor& a) { return at::reciprocal(a); });
}

at::Tensor sin_vulkan(const at::Tensor& self)
{
    if (self.is_complex()) return complex_unary(self, torchvulkan::complex_sin);
    return unary_op_vulkan(promote_to_float(self), UnaryOp::SIN, [](const at::Tensor& a) { return at::sin(a); });
}

at::Tensor cos_vulkan(const at::Tensor& self)
{
    if (self.is_complex()) return complex_unary(self, torchvulkan::complex_cos);
    return unary_op_vulkan(promote_to_float(self), UnaryOp::COS, [](const at::Tensor& a) { return at::cos(a); });
}

at::Tensor tan_vulkan(const at::Tensor& self)
{
    if (self.is_complex()) return complex_unary(self, torchvulkan::complex_tan);
    return unary_op_vulkan(promote_to_float(self), UnaryOp::TAN, [](const at::Tensor& a) { return at::tan(a); });
}

at::Tensor asin_vulkan(const at::Tensor& self)
{
    if (self.is_complex()) return complex_arc(self, ComplexArcOp::ASIN, torchvulkan::complex_asin);
    return unary_op_vulkan(promote_to_float(self), UnaryOp::ASIN, [](const at::Tensor& a) { return at::asin(a); });
}

at::Tensor acos_vulkan(const at::Tensor& self)
{
    if (self.is_complex()) return complex_arc(self, ComplexArcOp::ACOS, torchvulkan::complex_acos);
    return unary_op_vulkan(promote_to_float(self), UnaryOp::ACOS, [](const at::Tensor& a) { return at::acos(a); });
}

at::Tensor atan_vulkan(const at::Tensor& self)
{
    if (self.is_complex()) return complex_arc(self, ComplexArcOp::ATAN, torchvulkan::complex_atan);
    return unary_op_vulkan(promote_to_float(self), UnaryOp::ATAN, [](const at::Tensor& a) { return at::atan(a); });
}

at::Tensor sinh_vulkan(const at::Tensor& self)
{
    if (self.is_complex()) return complex_unary(self, torchvulkan::complex_sinh);
    return unary_op_vulkan(promote_to_float(self), UnaryOp::SINH, [](const at::Tensor& a) { return at::sinh(a); });
}

at::Tensor cosh_vulkan(const at::Tensor& self)
{
    if (self.is_complex()) return complex_unary(self, torchvulkan::complex_cosh);
    return unary_op_vulkan(promote_to_float(self), UnaryOp::COSH, [](const at::Tensor& a) { return at::cosh(a); });
}

at::Tensor tanh_vulkan(const at::Tensor& self)
{
    if (self.is_complex()) return complex_unary(self, torchvulkan::complex_tanh);
    return unary_op_vulkan(promote_to_float(self), UnaryOp::TANH, [](const at::Tensor& a) { return at::tanh(a); });
}

at::Tensor exp2_vulkan(const at::Tensor& self)
{
    if (self.is_complex()) return complex_unary(self, complex_exp2);
    return unary_op_vulkan(promote_to_float(self), UnaryOp::EXP2, [](const at::Tensor& a) { return at::exp2(a); });
}

at::Tensor log2_vulkan(const at::Tensor& self)
{
    if (self.is_complex()) return complex_unary(self, complex_log2);
    return unary_op_vulkan(promote_to_float(self), UnaryOp::LOG2, [](const at::Tensor& a) { return at::log2(a); });
}

at::Tensor log10_vulkan(const at::Tensor& self)
{
    if (self.is_complex()) return complex_unary(self, complex_log10);
    return unary_op_vulkan(promote_to_float(self), UnaryOp::LOG10, [](const at::Tensor& a) { return at::log10(a); });
}

at::Tensor expm1_vulkan(const at::Tensor& self)
{
    if (self.is_complex()) return complex_unary(self, torchvulkan::complex_expm1);
    return unary_op_vulkan(promote_to_float(self), UnaryOp::EXPM1, [](const at::Tensor& a) { return at::expm1(a); });
}

at::Tensor log1p_vulkan(const at::Tensor& self)
{
    if (self.is_complex()) return complex_unary(self, torchvulkan::complex_log1p);
    return unary_op_vulkan(promote_to_float(self), UnaryOp::LOG1P, [](const at::Tensor& a) { return at::log1p(a); });
}

at::Tensor rsqrt_vulkan(const at::Tensor& self)
{
    if (self.is_complex()) return complex_unary(self, complex_rsqrt);
    return unary_op_vulkan(promote_to_float(self), UnaryOp::RSQRT, [](const at::Tensor& a) { return at::rsqrt(a); });
}

at::Tensor sigmoid_vulkan(const at::Tensor& self)
{
    if (self.is_complex()) return complex_unary(self, complex_sigmoid);
    return unary_op_vulkan(promote_to_float(self), UnaryOp::SIGMOID, [](const at::Tensor& a) { return at::sigmoid(a); });
}

at::Tensor asinh_vulkan(const at::Tensor& self)
{
    if (self.is_complex()) return complex_arc(self, ComplexArcOp::ASINH, torchvulkan::complex_asinh);
    return unary_op_vulkan(promote_to_float(self), UnaryOp::ASINH, [](const at::Tensor& a) { return at::asinh(a); });
}

at::Tensor acosh_vulkan(const at::Tensor& self)
{
    if (self.is_complex()) return complex_arc(self, ComplexArcOp::ACOSH, torchvulkan::complex_acosh);
    return unary_op_vulkan(promote_to_float(self), UnaryOp::ACOSH, [](const at::Tensor& a) { return at::acosh(a); });
}

at::Tensor atanh_vulkan(const at::Tensor& self)
{
    if (self.is_complex()) return complex_arc(self, ComplexArcOp::ATANH, torchvulkan::complex_atanh);
    return unary_op_vulkan(promote_to_float(self), UnaryOp::ATANH, [](const at::Tensor& a) { return at::atanh(a); });
}

at::Tensor deg2rad_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::DEG2RAD, [](const at::Tensor& a) { return at::deg2rad(a); });
}

at::Tensor rad2deg_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::RAD2DEG, [](const at::Tensor& a) { return at::rad2deg(a); });
}

at::Tensor floor_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(self, UnaryOp::FLOOR, [](const at::Tensor& a) { return at::floor(a); });
}

at::Tensor ceil_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(self, UnaryOp::CEIL, [](const at::Tensor& a) { return at::ceil(a); });
}

at::Tensor trunc_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(self, UnaryOp::TRUNC, [](const at::Tensor& a) { return at::trunc(a); });
}

at::Tensor round_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(self, UnaryOp::ROUND, [](const at::Tensor& a) { return at::round(a); });
}

at::Tensor frac_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(self, UnaryOp::FRAC, [](const at::Tensor& a) { return at::frac(a); });
}

at::Tensor abs_vulkan(const at::Tensor& self)
{
    if (self.is_complex()) return torchvulkan::complex_abs(torchvulkan::complex_parts(self)).to(c10::toRealValueType(self.scalar_type()));
    return unary_op_vulkan(self, UnaryOp::ABS, [](const at::Tensor& a) { return at::abs(a); });
}

at::Tensor sign_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(self, UnaryOp::SIGN, [](const at::Tensor& a) { return at::sign(a); });
}

at::Tensor sgn_vulkan(const at::Tensor& self)
{
    if (self.is_complex()) return complex_unary(self, complex_sgn);
    return sign_vulkan(self);
}

at::Tensor erf_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::ERF, [](const at::Tensor& a) { return at::erf(a); });
}

at::Tensor erfinv_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::ERFINV, [](const at::Tensor& a) { return at::erfinv(a); });
}

torchvulkan::ComplexParts complex_sinc(const torchvulkan::ComplexParts& z)
{
    torchvulkan::ComplexParts pz{at::mul(z.re, M_PI), at::mul(z.im, M_PI)};
    torchvulkan::ComplexParts r = torchvulkan::complex_div(torchvulkan::complex_sin(pz), pz);
    at::Tensor zero = at::logical_and(at::eq(z.re, 0), at::eq(z.im, 0));
    return {at::where(zero, at::ones_like(r.re), r.re), at::where(zero, at::zeros_like(r.im), r.im)};
}

at::Tensor sinc_vulkan(const at::Tensor& self)
{
    if (self.is_complex()) return complex_unary(self, complex_sinc);
    return unary_op_vulkan(promote_to_float(self), UnaryOp::SINC, [](const at::Tensor& a) { return at::sinc(a); });
}

at::Tensor entr_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::ENTR, [](const at::Tensor& a) { return at::special_entr(a); });
}

at::Tensor lgamma_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::LGAMMA, [](const at::Tensor& a) { return at::lgamma(a); });
}

at::Tensor digamma_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::DIGAMMA, [](const at::Tensor& a) { return at::digamma(a); });
}

at::Tensor i0_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::I0, [](const at::Tensor& a) { return at::i0(a); });
}

at::Tensor i1_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::I1, [](const at::Tensor& a) { return at::special_i1(a); });
}

at::Tensor erfc_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::ERFC, [](const at::Tensor& a) { return at::erfc(a); });
}

at::Tensor erfcx_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::ERFCX, [](const at::Tensor& a) { return at::special_erfcx(a); });
}

at::Tensor log_ndtr_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::LOG_NDTR, [](const at::Tensor& a) { return at::special_log_ndtr(a); });
}

at::Tensor i0e_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::I0E, [](const at::Tensor& a) { return at::special_i0e(a); });
}

at::Tensor i1e_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::I1E, [](const at::Tensor& a) { return at::special_i1e(a); });
}

at::Tensor& relu_vulkan_(at::Tensor& self)
{
    return copy_inplace_result(self, [&] { return relu_vulkan(self); });
}

// the out= variants, which torch also routes the in-place forms (exp_, sigmoid_, ...) through
at::Tensor& exp_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return exp_vulkan(self); }); }
at::Tensor& log_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return log_vulkan(self); }); }
at::Tensor& sqrt_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return sqrt_vulkan(self); }); }
at::Tensor& neg_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_exact_out(out, [&] { return neg_vulkan(self); }); }
at::Tensor& reciprocal_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return reciprocal_vulkan(self); }); }
at::Tensor& sin_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return sin_vulkan(self); }); }
at::Tensor& cos_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return cos_vulkan(self); }); }
at::Tensor& tan_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return tan_vulkan(self); }); }
at::Tensor& asin_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return asin_vulkan(self); }); }
at::Tensor& acos_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return acos_vulkan(self); }); }
at::Tensor& atan_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return atan_vulkan(self); }); }
at::Tensor& sinh_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return sinh_vulkan(self); }); }
at::Tensor& cosh_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return cosh_vulkan(self); }); }
at::Tensor& tanh_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return tanh_vulkan(self); }); }
at::Tensor& asinh_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return asinh_vulkan(self); }); }
at::Tensor& acosh_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return acosh_vulkan(self); }); }
at::Tensor& atanh_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return atanh_vulkan(self); }); }
at::Tensor& exp2_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return exp2_vulkan(self); }); }
at::Tensor& log2_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return log2_vulkan(self); }); }
at::Tensor& log10_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return log10_vulkan(self); }); }
at::Tensor& expm1_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return expm1_vulkan(self); }); }
at::Tensor& log1p_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return log1p_vulkan(self); }); }
at::Tensor& rsqrt_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return rsqrt_vulkan(self); }); }
at::Tensor& sigmoid_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return sigmoid_vulkan(self); }); }
at::Tensor& deg2rad_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return deg2rad_vulkan(self); }); }
at::Tensor& rad2deg_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return rad2deg_vulkan(self); }); }
at::Tensor& floor_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_exact_out(out, [&] { return floor_vulkan(self); }); }
at::Tensor& ceil_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_exact_out(out, [&] { return ceil_vulkan(self); }); }
at::Tensor& trunc_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_exact_out(out, [&] { return trunc_vulkan(self); }); }
at::Tensor& round_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_exact_out(out, [&] { return round_vulkan(self); }); }
at::Tensor& frac_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_exact_out(out, [&] { return frac_vulkan(self); }); }
at::Tensor& abs_out_vulkan(const at::Tensor& self, at::Tensor& out) { return self.is_complex() ? fill_out(out, [&] { return abs_vulkan(self); }) : fill_exact_out(out, [&] { return abs_vulkan(self); }); }
at::Tensor& sign_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_exact_out(out, [&] { return sign_vulkan(self); }); }
at::Tensor& sgn_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_exact_out(out, [&] { return sgn_vulkan(self); }); }
at::Tensor& erf_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return erf_vulkan(self); }); }
at::Tensor& erfc_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return erfc_vulkan(self); }); }
at::Tensor& erfinv_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return erfinv_vulkan(self); }); }
at::Tensor& sinc_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return sinc_vulkan(self); }); }
at::Tensor& entr_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return entr_vulkan(self); }); }
at::Tensor& lgamma_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return lgamma_vulkan(self); }); }
at::Tensor& digamma_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return digamma_vulkan(self); }); }
at::Tensor& i0_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return i0_vulkan(self); }); }
at::Tensor& i1_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return i1_vulkan(self); }); }
at::Tensor& erfcx_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return erfcx_vulkan(self); }); }
at::Tensor& log_ndtr_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return log_ndtr_vulkan(self); }); }
at::Tensor& i0e_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return i0e_vulkan(self); }); }
at::Tensor& i1e_out_vulkan(const at::Tensor& self, at::Tensor& out) { return fill_out(out, [&] { return i1e_vulkan(self); }); }
at::Tensor& gelu_out_vulkan(const at::Tensor& self, c10::string_view approximate, at::Tensor& out) { return fill_out(out, [&] { return gelu_vulkan(self, approximate); }); }

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("relu", &relu_vulkan);
    m.impl("gelu", &gelu_vulkan);
    m.impl("gelu.out", &gelu_out_vulkan);
    m.impl("exp", &exp_vulkan);
    m.impl("log", &log_vulkan);
    m.impl("sqrt", &sqrt_vulkan);
    m.impl("neg", &neg_vulkan);
    m.impl("reciprocal", &reciprocal_vulkan);
    m.impl("sin", &sin_vulkan);
    m.impl("cos", &cos_vulkan);
    m.impl("tan", &tan_vulkan);
    m.impl("asin", &asin_vulkan);
    m.impl("acos", &acos_vulkan);
    m.impl("atan", &atan_vulkan);
    m.impl("sinh", &sinh_vulkan);
    m.impl("cosh", &cosh_vulkan);
    m.impl("tanh", &tanh_vulkan);
    m.impl("asinh", &asinh_vulkan);
    m.impl("acosh", &acosh_vulkan);
    m.impl("atanh", &atanh_vulkan);
    m.impl("exp2", &exp2_vulkan);
    m.impl("log2", &log2_vulkan);
    m.impl("log10", &log10_vulkan);
    m.impl("expm1", &expm1_vulkan);
    m.impl("log1p", &log1p_vulkan);
    m.impl("rsqrt", &rsqrt_vulkan);
    m.impl("sigmoid", &sigmoid_vulkan);
    m.impl("deg2rad", &deg2rad_vulkan);
    m.impl("rad2deg", &rad2deg_vulkan);
    m.impl("floor", &floor_vulkan);
    m.impl("ceil", &ceil_vulkan);
    m.impl("trunc", &trunc_vulkan);
    m.impl("round", &round_vulkan);
    m.impl("frac", &frac_vulkan);
    m.impl("abs", &abs_vulkan);
    m.impl("sign", &sign_vulkan);
    m.impl("sgn", &sgn_vulkan);
    m.impl("erf", &erf_vulkan);
    m.impl("erfinv", &erfinv_vulkan);
    m.impl("sinc", &sinc_vulkan);
    m.impl("special_entr", &entr_vulkan);
    m.impl("lgamma", &lgamma_vulkan);
    m.impl("digamma", &digamma_vulkan);
    m.impl("i0", &i0_vulkan);
    m.impl("special_i1", &i1_vulkan);
    m.impl("erfc", &erfc_vulkan);
    m.impl("special_erfcx", &erfcx_vulkan);
    m.impl("special_log_ndtr", &log_ndtr_vulkan);
    m.impl("relu_", &relu_vulkan_);
    m.impl("exp.out", &exp_out_vulkan);
    m.impl("log.out", &log_out_vulkan);
    m.impl("sqrt.out", &sqrt_out_vulkan);
    m.impl("neg.out", &neg_out_vulkan);
    m.impl("reciprocal.out", &reciprocal_out_vulkan);
    m.impl("sin.out", &sin_out_vulkan);
    m.impl("cos.out", &cos_out_vulkan);
    m.impl("tan.out", &tan_out_vulkan);
    m.impl("asin.out", &asin_out_vulkan);
    m.impl("acos.out", &acos_out_vulkan);
    m.impl("atan.out", &atan_out_vulkan);
    m.impl("sinh.out", &sinh_out_vulkan);
    m.impl("cosh.out", &cosh_out_vulkan);
    m.impl("tanh.out", &tanh_out_vulkan);
    m.impl("asinh.out", &asinh_out_vulkan);
    m.impl("acosh.out", &acosh_out_vulkan);
    m.impl("atanh.out", &atanh_out_vulkan);
    m.impl("exp2.out", &exp2_out_vulkan);
    m.impl("log2.out", &log2_out_vulkan);
    m.impl("log10.out", &log10_out_vulkan);
    m.impl("expm1.out", &expm1_out_vulkan);
    m.impl("log1p.out", &log1p_out_vulkan);
    m.impl("rsqrt.out", &rsqrt_out_vulkan);
    m.impl("sigmoid.out", &sigmoid_out_vulkan);
    m.impl("deg2rad.out", &deg2rad_out_vulkan);
    m.impl("rad2deg.out", &rad2deg_out_vulkan);
    m.impl("floor.out", &floor_out_vulkan);
    m.impl("ceil.out", &ceil_out_vulkan);
    m.impl("trunc.out", &trunc_out_vulkan);
    m.impl("round.out", &round_out_vulkan);
    m.impl("frac.out", &frac_out_vulkan);
    m.impl("abs.out", &abs_out_vulkan);
    m.impl("sign.out", &sign_out_vulkan);
    m.impl("sgn.out", &sgn_out_vulkan);
    m.impl("erf.out", &erf_out_vulkan);
    m.impl("erfc.out", &erfc_out_vulkan);
    m.impl("erfinv.out", &erfinv_out_vulkan);
    m.impl("sinc.out", &sinc_out_vulkan);
    m.impl("special_entr.out", &entr_out_vulkan);
    m.impl("lgamma.out", &lgamma_out_vulkan);
    m.impl("digamma.out", &digamma_out_vulkan);
    m.impl("i0.out", &i0_out_vulkan);
    m.impl("special_i1.out", &i1_out_vulkan);
    m.impl("special_erfcx.out", &erfcx_out_vulkan);
    m.impl("special_log_ndtr.out", &log_ndtr_out_vulkan);
    m.impl("special_i0e", &i0e_vulkan);
    m.impl("special_i0e.out", &i0e_out_vulkan);
    m.impl("special_i1e", &i1e_vulkan);
    m.impl("special_i1e.out", &i1e_out_vulkan);
}
