#include <torch/extension.h>
#include <ATen/TensorIterator.h>
#include <c10/core/MemoryFormat.h>
#include <cmath>
#include <functional>
#include <optional>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "api/ops/helpers.h"
#include "api/ops/internal.h"
#include <ATen/native/TypeProperties.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/ops/_foreach_add_native.h>
#include <ATen/ops/_foreach_mul_native.h>
#include "api/fallback.h"

namespace {

enum class BinaryOp {
    ADD = 0,
    SUB = 1,
    RSUB = 2,
    MUL = 3,
    DIV = 4,
    MAX = 5,
    MIN = 6,
    POW = 7,
    RPOW = 8,
    ATAN2 = 9,
    THRESHOLD_BACKWARD = 10,
    FMAX = 11,
    FMIN = 12,
    FMOD = 13,
    REMAINDER = 14,
    HYPOT = 15,
    XLOGY = 16,
    LOGADDEXP = 17,
    LOGADDEXP2 = 18,
    FLOOR_DIVIDE = 19,
    GELU_BACKWARD = 20,
    GELU_TANH_BACKWARD = 21
};

// at::native::alpha_check
void alpha_check(at::ScalarType dtype, const at::Scalar& alpha)
{
    TORCH_CHECK(!alpha.isBoolean() || dtype == at::kBool, "Boolean alpha only supported for Boolean results.");
    TORCH_CHECK(at::isFloatingType(dtype) || at::isComplexType(dtype) || alpha.isIntegral(true),
        "For integral input tensors, argument alpha must not be a floating point number.");
    TORCH_CHECK(at::isComplexType(dtype) || !alpha.isComplex(), "For non-complex input tensors, argument alpha must not be a complex number.");
}

// a 0-dim CPU tensor (or a wrapped Python number) mixed with vulkan tensors acts as a scalar
bool is_cpu_scalar(const at::Tensor& t)
{
    return t.unsafeGetTensorImpl()->is_wrapped_number() || (t.dim() == 0 && t.device().is_cpu());
}

std::optional<at::Tensor> dispatch_binary_shader(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& scalar,
    const at::Scalar& alpha,
    BinaryOp operation,
    c10::ScalarType promoted_type)
{
    DeviceContext* device = get_device_context(self);
    if (!is_dtype_supported(device, promoted_type)) return std::nullopt;

    bool use_scalar = !other.defined();
    at::Tensor a = self.to(promoted_type);
    at::Tensor b = use_scalar ? at::Tensor() : other.to(a.options());

    at::Tensor target = OutputTarget::take(promoted_type, use_scalar ? a.sizes() : at::IntArrayRef(at::infer_size(a.sizes(), b.sizes())), {a, b});
    // with a scalar the CPU's iterator also holds the 0-dim scalar, which makes a dense output take self's strides
    // exactly (size-1 dims included): allocated so here
    if (!target.defined() && use_scalar && a.is_non_overlapping_and_dense()) target = at::empty_strided(a.sizes(), a.strides(), a.options());
    at::TensorIteratorConfig config;
    config.set_check_mem_overlap(true).add_output(target).add_const_input(a);
    if (!use_scalar) config.add_const_input(b);
    at::TensorIterator iter = config.build();

    at::Tensor out = iter.output();
    uint64_t numel = iter.numel();
    if (numel == 0) return out;

    int32_t out_dims = static_cast<int32_t>(iter.ndim());
    if (out_dims > MAX_DIMS || !fits_32bit_indexing(iter)) return std::nullopt;

    uint32_t vecSize = get_dtype_vec_size(promoted_type);
    uint32_t workgroupSizeX = get_dtype_workgroup_size(promoted_type, vecSize, device);
    uint32_t contiguous = iter.is_contiguous() ? 1 : 0;
    uint32_t has_alpha = (alpha.toDouble() != 1.0) ? 1u : 0u;

    SpecializationBuilder spd{};
    spd.push(static_cast<uint32_t>(operation))
       .push(contiguous)
       .push(use_scalar ? 1u : 0u)
       .push(out_dims)
       .push(workgroupSizeX)
       .push(has_alpha);

    MetadataBuilder metadataBuilder = get_iter_metadata(iter, {1, use_scalar ? -1 : 2, 0});
    Metadata metadata = metadataBuilder.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(a))
       .push(use_scalar ? uint64_t(0) : get_tensor_address(b))
       .push(get_output_address(out))
       .push_metadata(contiguous ? nullptr : &metadata)
       .push(numel)
       .push_scalar(alpha, promoted_type)
       .push_operand_scalar(use_scalar ? scalar : at::Scalar(0), promoted_type);

    std::vector<at::Tensor> reads = {a};
    if (!use_scalar) reads.push_back(b);

    uint64_t threads = contiguous ? (numel + (vecSize - 1)) / vecSize : numel;
    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::BINARY, promoted_type},
        device->default_subgroup_size,
        spd.build(),
        pcs.build(),
        /* read = */ reads,
        /* write = */ {out},
        get_group_count(threads, workgroupSizeX), 1, 1
    );

    return out;
}

bool fits_dtype(const at::Scalar& s, at::ScalarType dtype)
{
    if (s.isComplex() || !s.isFloatingPoint()) return true;
    double v = s.toDouble();
    if (!std::isfinite(v)) return true;
    double limit = dtype == at::kHalf ? 65504.0 : 3.3895313892515355e38;
    return std::abs(v) <= limit;
}

torchvulkan::ComplexParts complex_pow(const torchvulkan::ComplexParts& base, const torchvulkan::ComplexParts& exponent)
{
    return torchvulkan::complex_exp(torchvulkan::complex_mul(torchvulkan::complex_log(base), exponent));  // base first: broadcast errors name it as a
}

std::optional<torchvulkan::ComplexParts> complex_pow_scalar(const torchvulkan::ComplexParts& z, const at::Scalar& exponent)
{
    if (exponent.isComplex() && exponent.toComplexDouble().imag() != 0) return std::nullopt;
    double e = exponent.isComplex() ? exponent.toComplexDouble().real() : exponent.toDouble();
    torchvulkan::ComplexParts one{at::ones_like(z.re), at::zeros_like(z.im)};
    if (e == 0.5) return torchvulkan::complex_sqrt(z);
    if (e == 2.0) return torchvulkan::complex_mul(z, z);
    if (e == 3.0) return torchvulkan::complex_mul(torchvulkan::complex_mul(z, z), z);
    if (e == -0.5) return torchvulkan::complex_div(one, torchvulkan::complex_sqrt(z));
    if (e == -1.0) return torchvulkan::complex_div(one, z);
    if (e == -2.0) return torchvulkan::complex_div(one, torchvulkan::complex_mul(z, z));
    return std::nullopt;
}

std::optional<at::Tensor> complex_binary(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar* scalar,
    const at::Scalar& alpha,
    BinaryOp op,
    at::ScalarType promoted)
{
    if (!at::isComplexType(promoted)) return std::nullopt;
    bool supported = op == BinaryOp::ADD || op == BinaryOp::SUB || op == BinaryOp::RSUB || op == BinaryOp::MUL || op == BinaryOp::DIV ||
                     op == BinaryOp::POW || op == BinaryOp::RPOW || op == BinaryOp::LOGADDEXP;
    if (!supported) return std::nullopt;

    at::ScalarType real = c10::toRealValueType(promoted == at::kComplexHalf ? at::kComplexFloat : promoted);
    auto parts_of = [real](const at::Tensor& t) {
        torchvulkan::ComplexParts p = torchvulkan::complex_parts(t);
        return torchvulkan::ComplexParts{p.re.to(real), p.im.to(real)};
    };
    torchvulkan::ComplexParts a = parts_of(self);
    torchvulkan::ComplexParts b = scalar != nullptr ? torchvulkan::complex_scalar_parts(*scalar, a.re) : parts_of(other);
    bool scaled = op == BinaryOp::SUB || op == BinaryOp::RSUB ||
                  (alpha.isComplex() ? alpha.toComplexDouble() != c10::complex<double>(1.0) : alpha.toDouble() != 1.0);
    torchvulkan::ComplexParts alpha_parts = scaled ? torchvulkan::complex_scalar_parts(alpha, a.re) : torchvulkan::ComplexParts{};

    torchvulkan::ComplexParts r;
    switch (op) {
        case BinaryOp::ADD:
        case BinaryOp::SUB: {
            if (scaled) b = torchvulkan::complex_mul(b, alpha_parts);
            r = op == BinaryOp::ADD ? torchvulkan::ComplexParts{at::add(a.re, b.re), at::add(a.im, b.im)} : torchvulkan::ComplexParts{at::sub(a.re, b.re), at::sub(a.im, b.im)};
            break;
        }
        case BinaryOp::RSUB: {
            if (scaled) a = torchvulkan::complex_mul(a, alpha_parts);
            r = {at::sub(b.re, a.re), at::sub(b.im, a.im)};
            break;
        }
        case BinaryOp::MUL: r = torchvulkan::complex_mul(a, b); break;
        case BinaryOp::DIV: r = torchvulkan::complex_div(a, b); break;
        case BinaryOp::POW: {
            std::optional<torchvulkan::ComplexParts> shortcut = scalar != nullptr ? complex_pow_scalar(a, *scalar) : std::nullopt;
            r = shortcut.has_value() ? *shortcut : complex_pow(a, b);
            break;
        }
        case BinaryOp::LOGADDEXP: r = torchvulkan::complex_logaddexp(a, b); break;
        default: r = complex_pow(b, a); break;  // RPOW: scalar ** self
    }
    return torchvulkan::complex_from(r, promoted);
}

at::Tensor binary_op_vulkan(
    const at::Tensor& self_, 
    const at::Scalar& other, 
    const at::Scalar& alpha, 
    BinaryOp operation, 
    const std::function<at::Tensor(const at::Tensor&, const at::Scalar&)>& fallback)
{
    // a bool result computes on the raw bytes; other dtypes cast (and so normalize) bools on the way in
    at::Tensor self = at::result_type(self_, other) == at::kBool ? normalize_bool(self_) : self_;
    at::ScalarType promoted = at::result_type(self, other);
    // the shader takes scalars in the tensor's dtype: one that half or bfloat16 can't hold computes in float32, as the CPU's opmath does
    if (is_low_prec(promoted) && (!fits_dtype(other, promoted) || !fits_dtype(alpha, promoted))) {
        return binary_op_vulkan(self.to(at::kFloat), other, alpha, operation, fallback).to(promoted);
    }
    std::optional<at::Tensor> out = complex_binary(self, at::Tensor(), &other, alpha, operation, promoted);
    if (!out.has_value()) out = dispatch_binary_shader(self, at::Tensor(), other, alpha, operation, promoted);
    if (out.has_value()) return *out;

    TORCHVULKAN_FALLBACK("A binary op on ", self.scalar_type(), " can't run on the Vulkan device. Falling back to CPU.");
    return fallback(self.cpu(), other).to(self.device());
}

at::Tensor binary_op_vulkan(
    const at::Tensor& self_, 
    const at::Tensor& other_, 
    const at::Scalar& alpha, 
    BinaryOp operation, 
    const std::function<at::Tensor(const at::Tensor&, const at::Tensor&)>& fallback)
{
    bool bool_result = at::result_type(self_, other_) == at::kBool;
    at::Tensor self = bool_result ? normalize_bool(self_) : self_, other = bool_result ? normalize_bool(other_) : other_;
    if (is_cpu_scalar(other) && self.device().type() == c10::DeviceType::PrivateUse1) 
    {
        return binary_op_vulkan(
            self, other.item(), alpha, operation,
            [&fallback, &other](const at::Tensor& a, const at::Scalar&) { return fallback(a, other); }
        );
    }

    if (self.device() != other.device() && !is_cpu_scalar(self) && !is_cpu_scalar(other)) {
        const at::Tensor& on_vulkan = self.device().type() == c10::DeviceType::PrivateUse1 ? self : other;
        const at::Tensor& elsewhere = self.device().type() == c10::DeviceType::PrivateUse1 ? other : self;
        TORCH_CHECK(false, "Expected all tensors to be on the same device, but found at least two devices, ", on_vulkan.device(), " and ",
            elsewhere.device(), "!");
    }
    at::Tensor lhs = is_cpu_scalar(self) ? self.to(other.device()) : self;
    at::Tensor rhs = other.device() == lhs.device() ? other : other.to(lhs.device());

    std::optional<at::Tensor> out = complex_binary(lhs, rhs, nullptr, alpha, operation, at::result_type(self, other));
    if (!out.has_value()) out = dispatch_binary_shader(lhs, rhs, at::Scalar(0), alpha, operation, at::result_type(self, other));
    if (out.has_value()) return *out;

    TORCHVULKAN_FALLBACK("A binary op on ", at::result_type(self, other), " can't run on the Vulkan device. Falling back to CPU.");
    return fallback(lhs.cpu(), rhs.cpu()).to(lhs.device());
}

} // namespace

at::Tensor torchvulkan::add_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) 
{    
    alpha_check(at::result_type(self, other), alpha);
    if (self.scalar_type() == at::kBool) {
        if (alpha.to<bool>() == false) return self.clone();
        return binary_op_vulkan(self, other, alpha, BinaryOp::MAX, [alpha](const at::Tensor& a, const at::Tensor& b) { return at::add(a, b, alpha); });
    }
    return binary_op_vulkan(self, other, alpha, BinaryOp::ADD, [alpha](const at::Tensor& a, const at::Tensor& b) { return at::add(a, b, alpha); });
}

at::Tensor torchvulkan::multiply_vulkan(const at::Tensor& self, const at::Tensor& other) {
    return binary_op_vulkan(self, other, (int)1, BinaryOp::MUL, [](const at::Tensor& a, const at::Tensor& b) { return at::mul(a, b); });
}

at::Tensor torchvulkan::divide_vulkan(const at::Tensor& self, const at::Tensor& other) {
    at::Tensor self_vulkan = self;
    at::Tensor other_vulkan = other;

    // true division of integers gives the default dtype
    if (!self.is_floating_point() && !other.is_floating_point() && !self.is_complex() && !other.is_complex()) {
        self_vulkan = promote_to_float(self);
        other_vulkan = promote_to_float(other);
    }

    return binary_op_vulkan(self_vulkan, other_vulkan, (int)1, BinaryOp::DIV, [](const at::Tensor& a, const at::Tensor& b) { return at::div(a, b); });
}

at::Tensor torchvulkan::divide_scalar_vulkan(const at::Tensor& self, const at::Scalar& other) {
    at::Tensor self_vulkan = self;

    if (at::isIntegralType(self.scalar_type(), /* includeBool = */ true)) self_vulkan = promote_to_float(self);

    return binary_op_vulkan(self_vulkan, other, (int)1, BinaryOp::DIV, [](const at::Tensor& a, const at::Scalar& b) { return at::div(a, b); });
}

namespace {

at::Tensor add_scalar_vulkan(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) 
{
    alpha_check(at::result_type(self, other), alpha);
    if (self.scalar_type() == at::kBool) {
        if (alpha.to<bool>() == false) return self.clone();
        return binary_op_vulkan(self, other, alpha, BinaryOp::MAX, [alpha](const at::Tensor& a, const at::Scalar& b) { return at::add(a, b, alpha); });
    }
    return binary_op_vulkan(self, other, alpha, BinaryOp::ADD, [alpha](const at::Tensor& a, const at::Scalar& b) { return at::add(a, b, alpha); });
}

// an in-place op writes self, so a CPU self (even 0-dim) can't take a vulkan result
void check_inplace_device(const at::Tensor& self, const at::Tensor& other)
{
    TORCH_CHECK(self.device() == other.device() || other.device().is_cpu(), "Expected all tensors to be on the same device, but found at least two devices, ",
        other.device(), " and ", self.device(), "!");
}

void sub_check(at::ScalarType a, at::ScalarType b)
{
    TORCH_CHECK_NOT_IMPLEMENTED(a != at::kBool || b != at::kBool,
        "Subtraction, the `-` operator, with two bool tensors is not supported. Use the `^` or `logical_xor()` operator instead.");
    TORCH_CHECK_NOT_IMPLEMENTED(a != at::kBool && b != at::kBool,
        "Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");
}

at::Tensor subtract_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    sub_check(self.scalar_type(), other.scalar_type());
    alpha_check(at::result_type(self, other), alpha);
    return binary_op_vulkan(self, other, alpha, BinaryOp::SUB, [alpha](const at::Tensor& a, const at::Tensor& b) { return at::sub(a, b, alpha); });
}

at::Tensor subtract_scalar_vulkan(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
    sub_check(self.scalar_type(), other.isBoolean() ? at::kBool : at::kLong);
    alpha_check(at::result_type(self, other), alpha);
    return binary_op_vulkan(self, other, alpha, BinaryOp::SUB, [alpha](const at::Tensor& a, const at::Scalar& b) { return at::sub(a, b, alpha); });
}

at::Tensor rsub_scalar_vulkan(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
    sub_check(self.scalar_type(), other.isBoolean() ? at::kBool : at::kLong);
    alpha_check(at::result_type(self, other), alpha);
    return binary_op_vulkan(self, other, alpha, BinaryOp::RSUB, [alpha](const at::Tensor& a, const at::Scalar& b) { return at::rsub(a, b, alpha); });
}

at::Tensor multiply_scalar_vulkan(const at::Tensor& self, const at::Scalar& other) {
    return binary_op_vulkan(self, other, (int)1, BinaryOp::MUL, [](const at::Tensor& a, const at::Scalar& b) { return at::mul(a, b); });
}

at::Tensor maximum_vulkan(const at::Tensor& self, const at::Tensor& other) {
    TORCH_CHECK_TYPE(!self.is_complex() && !other.is_complex(), "maximum not implemented for complex tensors.");
    return binary_op_vulkan(self, other, (int)1, BinaryOp::MAX, [](const at::Tensor& a, const at::Tensor& b) { return at::max(a, b); });
}

at::Tensor minimum_vulkan(const at::Tensor& self, const at::Tensor& other) {
    TORCH_CHECK_TYPE(!self.is_complex() && !other.is_complex(), "minimum not implemented for complex tensors.");
    return binary_op_vulkan(self, other, (int)1, BinaryOp::MIN, [](const at::Tensor& a, const at::Tensor& b) { return at::min(a, b); });
}

at::Tensor pow_vulkan(const at::Tensor& self, const at::Tensor& other) {
    return binary_op_vulkan(self, other, (int)1, BinaryOp::POW, [](const at::Tensor& a, const at::Tensor& b) { return at::pow(a, b); });
}

at::Tensor pow_tensor_scalar_vulkan(const at::Tensor& self, const at::Scalar& other) {
    TORCH_CHECK(!at::isIntegralType(self.scalar_type(), /*includeBool=*/true) || !other.isIntegral(/*includeBool=*/true) || other.toLong() >= 0,
        "Integers to negative integer powers are not allowed.");
    return binary_op_vulkan(self, other, (int)1, BinaryOp::POW, [](const at::Tensor& a, const at::Scalar& b) { return at::pow(a, b); });
}

at::Tensor pow_scalar_vulkan(const at::Scalar& self, const at::Tensor& other) {
    return binary_op_vulkan(other, self, (int)1, BinaryOp::RPOW, [](const at::Tensor& a, const at::Scalar& b) { return at::pow(b, a); });
}

at::Tensor atan2_vulkan(const at::Tensor& self, const at::Tensor& other) {
    at::Tensor self_vulkan = self;
    at::Tensor other_vulkan = other;

    if (!self.is_floating_point() && !other.is_floating_point() && !self.is_complex() && !other.is_complex()) {
        self_vulkan = self.to(c10::kFloat);
        other_vulkan = other.to(c10::kFloat);
    }

    return binary_op_vulkan(self_vulkan, other_vulkan, (int)1, BinaryOp::ATAN2, [](const at::Tensor& a, const at::Tensor& b) { return at::atan2(a, b); });
}

at::Tensor threshold_backward_vulkan(const at::Tensor& grad_output, const at::Tensor& self, const at::Scalar& threshold) {
    return binary_op_vulkan(grad_output, self, threshold, BinaryOp::THRESHOLD_BACKWARD, [threshold](const at::Tensor& a, const at::Tensor& b) { return at::threshold_backward(a, b, threshold); });
}

at::Tensor gelu_backward_vulkan(const at::Tensor& grad_output, const at::Tensor& self, c10::string_view approximate) {
    auto fallback = [approximate](const at::Tensor& a, const at::Tensor& b) { return at::gelu_backward(a, b, approximate); };
    if (!self.is_floating_point() || (approximate != "none" && approximate != "tanh")) return fallback(grad_output.cpu(), self.cpu()).to(self.device());
    return binary_op_vulkan(grad_output, self, (int)1, approximate == "tanh" ? BinaryOp::GELU_TANH_BACKWARD : BinaryOp::GELU_BACKWARD, fallback);
}

at::Tensor fmax_vulkan(const at::Tensor& self, const at::Tensor& other) {
    TORCH_CHECK_TYPE(!self.is_complex() && !other.is_complex(), "fmax not implemented for complex tensors.");
    return binary_op_vulkan(self, other, (int)1, BinaryOp::FMAX, [](const at::Tensor& a, const at::Tensor& b) { return at::fmax(a, b); });
}

at::Tensor fmin_vulkan(const at::Tensor& self, const at::Tensor& other) {
    TORCH_CHECK_TYPE(!self.is_complex() && !other.is_complex(), "fmin not implemented for complex tensors.");
    return binary_op_vulkan(self, other, (int)1, BinaryOp::FMIN, [](const at::Tensor& a, const at::Tensor& b) { return at::fmin(a, b); });
}

// an integer divided by zero gives all bits set (as CUDA): 32 of them for int64, so 2^32 - 1 for non-negative x and -1 otherwise
at::Tensor integer_by_zero(const at::Tensor& self, const at::Tensor& other, const at::Tensor& result)
{
    if (!at::isIntegralType(result.scalar_type(), /*includeBool=*/false)) return result;
    at::Tensor pattern = result.scalar_type() == at::kLong
        ? at::where(at::ge(self, 0), at::full_like(result, 4294967295LL), at::full_like(result, -1))
        : at::full_like(result, result.scalar_type() == at::kByte ? 255 : -1);
    return at::where(at::eq(other, 0), pattern, result);
}

at::Tensor fmod_vulkan(const at::Tensor& self, const at::Tensor& other) {
    at::Tensor r = binary_op_vulkan(self, other, (int)1, BinaryOp::FMOD, [](const at::Tensor& a, const at::Tensor& b) { return at::fmod(a, b); });
    return integer_by_zero(self, other, r);
}

at::Tensor fmod_scalar_vulkan(const at::Tensor& self, const at::Scalar& other) {
    return binary_op_vulkan(self, other, (int)1, BinaryOp::FMOD, [](const at::Tensor& a, const at::Scalar& b) { return at::fmod(a, b); });
}

at::Tensor remainder_vulkan(const at::Tensor& self, const at::Tensor& other) {
    at::Tensor r = binary_op_vulkan(self, other, (int)1, BinaryOp::REMAINDER, [](const at::Tensor& a, const at::Tensor& b) { return at::remainder(a, b); });
    return integer_by_zero(self, other, r);
}

at::Tensor remainder_scalar_vulkan(const at::Tensor& self, const at::Scalar& other) {
    return binary_op_vulkan(self, other, (int)1, BinaryOp::REMAINDER, [](const at::Tensor& a, const at::Scalar& b) { return at::remainder(a, b); });
}

at::Tensor hypot_vulkan(const at::Tensor& self, const at::Tensor& other) {
    at::Tensor a = self, b = other;
    if (!self.is_floating_point() && !other.is_floating_point()) {
        a = self.to(c10::kFloat);
        b = other.to(c10::kFloat);
    }
    return binary_op_vulkan(a, b, (int)1, BinaryOp::HYPOT, [](const at::Tensor& x, const at::Tensor& y) { return at::hypot(x, y); });
}

at::Tensor xlogy_vulkan(const at::Tensor& self, const at::Tensor& other) {
    at::Tensor a = self, b = other;
    if (!self.is_floating_point() && !other.is_floating_point()) {
        a = self.to(c10::kFloat);
        b = other.to(c10::kFloat);
    }
    return binary_op_vulkan(a, b, (int)1, BinaryOp::XLOGY, [](const at::Tensor& x, const at::Tensor& y) { return at::xlogy(x, y); });
}

at::Tensor logaddexp_vulkan(const at::Tensor& self, const at::Tensor& other) {
    return binary_op_vulkan(self, other, (int)1, BinaryOp::LOGADDEXP, [](const at::Tensor& a, const at::Tensor& b) { return at::logaddexp(a, b); });
}

at::Tensor logaddexp2_vulkan(const at::Tensor& self, const at::Tensor& other) {
    return binary_op_vulkan(self, other, (int)1, BinaryOp::LOGADDEXP2, [](const at::Tensor& a, const at::Tensor& b) { return at::logaddexp2(a, b); });
}

// an integer bound outside an integer tensor's range saturates to it (clamp_min(uint8, -1) leaves the tensor alone)
// torch's prepare_clamp_bound: an integral bound past the dtype's range on the side it can't reach changes nothing and
// is dropped; on the other side every element would be unrepresentable, which raises
std::optional<at::Scalar> clamp_bound(at::ScalarType dtype, const at::Scalar& bound, bool lower)
{
    if (!at::isIntegralType(dtype, /*includeBool=*/false)) return bound;
    int64_t lowest = 0;
    uint64_t highest = 0;
    switch (dtype) {
        case at::kByte: highest = std::numeric_limits<uint8_t>::max(); break;
        case at::kUInt16: highest = std::numeric_limits<uint16_t>::max(); break;
        case at::kUInt32: highest = std::numeric_limits<uint32_t>::max(); break;
        case at::kUInt64: highest = std::numeric_limits<uint64_t>::max(); break;
        case at::kChar: lowest = std::numeric_limits<int8_t>::lowest(); highest = std::numeric_limits<int8_t>::max(); break;
        case at::kShort: lowest = std::numeric_limits<int16_t>::lowest(); highest = std::numeric_limits<int16_t>::max(); break;
        case at::kInt: lowest = std::numeric_limits<int32_t>::lowest(); highest = std::numeric_limits<int32_t>::max(); break;
        default: lowest = std::numeric_limits<int64_t>::lowest(); highest = std::numeric_limits<int64_t>::max(); break;
    }
    bool below = false, above = false;
    if (bound.isFloatingPoint()) {
        double v = bound.toDouble();
        below = v < static_cast<double>(lowest);
        above = v > static_cast<double>(highest);
    } else if (bound.type() == at::ScalarType::UInt64) {
        above = bound.toUInt64() > highest;
    } else {
        int64_t v = bound.toLong();
        below = v < lowest;
        above = v > 0 && static_cast<uint64_t>(v) > highest;
    }
    if (!below && !above) return bound;
    TORCH_CHECK(lower ? below : above, "Clamp ", lower ? "min" : "max", " value ", bound.toDouble(), " is outside the representable range of ", dtype);
    return std::nullopt;
}

at::Tensor clamp_min_scalar_vulkan(const at::Tensor& self, const at::Scalar& min) {
    std::optional<at::Scalar> bound = clamp_bound(at::result_type(self, min), min, /*lower=*/true);
    if (!bound.has_value()) return self.clone();
    return binary_op_vulkan(self, *bound, (int)1, BinaryOp::MAX, [](const at::Tensor& a, const at::Scalar& b) { return at::clamp_min(a, b); });
}

at::Tensor clamp_max_scalar_vulkan(const at::Tensor& self, const at::Scalar& max) {
    std::optional<at::Scalar> bound = clamp_bound(at::result_type(self, max), max, /*lower=*/false);
    if (!bound.has_value()) return self.clone();
    return binary_op_vulkan(self, *bound, (int)1, BinaryOp::MIN, [](const at::Tensor& a, const at::Scalar& b) { return at::clamp_max(a, b); });
}

at::Tensor clamp_min_tensor_vulkan(const at::Tensor& self, const at::Tensor& min) {
    return maximum_vulkan(self, min);
}

at::Tensor clamp_max_tensor_vulkan(const at::Tensor& self, const at::Tensor& max) {
    return minimum_vulkan(self, max);
}

at::Tensor clamp_scalar_vulkan(const at::Tensor& self, const c10::optional<at::Scalar>& min, const c10::optional<at::Scalar>& max) {
    TORCH_CHECK(min.has_value() || max.has_value(), "torch.clamp: At least one of 'min' or 'max' must not be None");
    at::ScalarType dtype = self.scalar_type();
    if (min.has_value()) dtype = at::promote_types(dtype, at::result_type(self, *min));
    if (max.has_value()) dtype = at::promote_types(dtype, at::result_type(self, *max));
    at::Tensor result = self.to(dtype);
    if (min.has_value()) result = clamp_min_scalar_vulkan(result, *min);
    if (max.has_value()) result = clamp_max_scalar_vulkan(result, *max);
    return result;
}

at::Tensor clamp_tensor_vulkan(const at::Tensor& self, const c10::optional<at::Tensor>& min, const c10::optional<at::Tensor>& max) {
    bool has_min = min.has_value() && min->defined();
    bool has_max = max.has_value() && max->defined();
    TORCH_CHECK(has_min || has_max, "torch.clamp: At least one of 'min' or 'max' must not be None");
    // one promotion over self and both bounds, as the CPU (not max in self's dtype, then min)
    at::native::ResultTypeState state = at::native::update_result_type_state(self, at::native::ResultTypeState{});
    if (has_min) state = at::native::update_result_type_state(*min, state);
    if (has_max) state = at::native::update_result_type_state(*max, state);
    at::ScalarType dtype = at::native::result_type(state);
    at::Tensor result = self.to(dtype);
    if (has_min) result = maximum_vulkan(result, min->to(dtype));
    if (has_max) result = minimum_vulkan(result, max->to(dtype));
    return result;
}

at::Tensor& clamp_min_out_vulkan(const at::Tensor& self, const at::Scalar& min, at::Tensor& out) {
    return fill_out(out, [&] { return clamp_min_scalar_vulkan(self, min); });
}

at::Tensor& clamp_max_out_vulkan(const at::Tensor& self, const at::Scalar& max, at::Tensor& out) {
    return fill_out(out, [&] { return clamp_max_scalar_vulkan(self, max); });
}

at::Tensor& clamp_min_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& min, at::Tensor& out) {
    return fill_out(out, [&] { return clamp_min_tensor_vulkan(self, min); });
}

at::Tensor& clamp_max_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& max, at::Tensor& out) {
    return fill_out(out, [&] { return clamp_max_tensor_vulkan(self, max); });
}

at::Tensor& clamp_out_vulkan(const at::Tensor& self, const c10::optional<at::Scalar>& min, const c10::optional<at::Scalar>& max, at::Tensor& out) {
    return fill_out(out, [&] { return clamp_scalar_vulkan(self, min, max); });
}

at::Tensor& clamp_tensor_out_vulkan(const at::Tensor& self, const c10::optional<at::Tensor>& min, const c10::optional<at::Tensor>& max, at::Tensor& out) {
    return fill_out(out, [&] { return clamp_tensor_vulkan(self, min, max); });
}

at::Tensor& add_out_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha, at::Tensor& out) {
    return fill_out(out, [&] { return torchvulkan::add_vulkan(self, other, alpha); });
}

at::Tensor& subtract_out_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha, at::Tensor& out) {
    return fill_out(out, [&] { return subtract_vulkan(self, other, alpha); });
}

at::Tensor& multiply_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    check_inplace_broadcast(out, self, other);
    return fill_out(out, [&] { return torchvulkan::multiply_vulkan(self, other); });
}

at::Tensor& divide_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    check_inplace_broadcast(out, self, other);
    return fill_out(out, [&] { return torchvulkan::divide_vulkan(self, other); });
}

at::Tensor& atan2_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    check_inplace_broadcast(out, self, other);
    return fill_out(out, [&] { return atan2_vulkan(self, other); });
}

at::Tensor& fmax_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    check_inplace_broadcast(out, self, other);
    return fill_out(out, [&] { return fmax_vulkan(self, other); });
}

at::Tensor& fmin_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    check_inplace_broadcast(out, self, other);
    return fill_out(out, [&] { return fmin_vulkan(self, other); });
}

at::Tensor& fmod_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    check_inplace_broadcast(out, self, other);
    return fill_out(out, [&] { return fmod_vulkan(self, other); });
}

at::Tensor& hypot_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    check_inplace_broadcast(out, self, other);
    return fill_out(out, [&] { return hypot_vulkan(self, other); });
}

at::Tensor& logaddexp_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    check_inplace_broadcast(out, self, other);
    return fill_out(out, [&] { return logaddexp_vulkan(self, other); });
}

at::Tensor& logaddexp2_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    check_inplace_broadcast(out, self, other);
    return fill_out(out, [&] { return logaddexp2_vulkan(self, other); });
}

at::Tensor& maximum_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    check_inplace_broadcast(out, self, other);
    return fill_out(out, [&] { return maximum_vulkan(self, other); });
}

at::Tensor& minimum_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    check_inplace_broadcast(out, self, other);
    return fill_out(out, [&] { return minimum_vulkan(self, other); });
}

at::Tensor& remainder_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    check_inplace_broadcast(out, self, other);
    return fill_out(out, [&] { return remainder_vulkan(self, other); });
}

at::Tensor remainder_scalar_tensor_vulkan(const at::Scalar& self, const at::Tensor& other) {
    at::Tensor wrapped = at::scalar_tensor(self, other.options().dtype(at::result_type(other, self)));
    wrapped.unsafeGetTensorImpl()->set_wrapped_number(true);
    return remainder_vulkan(wrapped, other);
}

at::Tensor& xlogy_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    check_inplace_broadcast(out, self, other);
    return fill_out(out, [&] { return xlogy_vulkan(self, other); });
}

at::Tensor& threshold_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& self, const at::Scalar& threshold, at::Tensor& grad_input) {
    return fill_out(grad_input, [&] { return threshold_backward_vulkan(grad_output, self, threshold); });
}

at::Tensor& gelu_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& self, c10::string_view approximate, at::Tensor& grad_input) {
    return fill_out(grad_input, [&] { return gelu_backward_vulkan(grad_output, self, approximate); });
}

at::Tensor add_relu_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    return at::relu(torchvulkan::add_vulkan(self, other, alpha));
}

at::Tensor add_relu_scalar_vulkan(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
    return at::relu(add_scalar_vulkan(self, other, alpha));
}

at::Tensor& add_relu_out_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha, at::Tensor& out) {
    return fill_out(out, [&] { return add_relu_vulkan(self, other, alpha); });
}

at::Tensor& add_relu_vulkan_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    return copy_inplace_result(self, [&] { return add_relu_vulkan(self, other, alpha); });
}

at::Tensor& add_relu_scalar_vulkan_(at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
    return copy_inplace_result(self, [&] { return add_relu_scalar_vulkan(self, other, alpha); });
}

at::Tensor& add_vulkan_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    check_inplace_device(self, other);
    return copy_inplace_result(self, [&] { return torchvulkan::add_vulkan(self, other, alpha); });
}

at::Tensor& add_scalar_vulkan_(at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
    return copy_inplace_result(self, [&] { return add_scalar_vulkan(self, other, alpha); });
}

at::Tensor& subtract_vulkan_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    check_inplace_device(self, other);
    return copy_inplace_result(self, [&] { return subtract_vulkan(self, other, alpha); });
}

at::Tensor& subtract_scalar_vulkan_(at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
    return copy_inplace_result(self, [&] { return subtract_scalar_vulkan(self, other, alpha); });
}

at::Tensor& multiply_vulkan_(at::Tensor& self, const at::Tensor& other) {
    check_inplace_device(self, other);
    return copy_inplace_result(self, [&] { return torchvulkan::multiply_vulkan(self, other); });
}

at::Tensor& multiply_scalar_vulkan_(at::Tensor& self, const at::Scalar& other) {
    return copy_inplace_result(self, [&] { return multiply_scalar_vulkan(self, other); });
}

at::Tensor& divide_vulkan_(at::Tensor& self, const at::Tensor& other) {
    check_inplace_device(self, other);
    return copy_inplace_result(self, [&] { return torchvulkan::divide_vulkan(self, other); });
}

at::Tensor& divide_scalar_vulkan_(at::Tensor& self, const at::Scalar& other) {
    return copy_inplace_result(self, [&] { return torchvulkan::divide_scalar_vulkan(self, other); });
}

std::optional<at::Tensor> dispatch_addc_shader(const at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2, const at::Scalar& value,
                                               bool divide, at::ScalarType dtype)
{
    DeviceContext* device = get_device_context(self.is_privateuseone() ? self : tensor1.is_privateuseone() ? tensor1 : tensor2);
    if (!is_dtype_supported(device, dtype)) return std::nullopt;
    at::Device where = self.is_privateuseone() ? self.device() : tensor1.is_privateuseone() ? tensor1.device() : tensor2.device();
    at::Tensor s = self.to(where, dtype), a = tensor1.to(where, dtype), b = tensor2.to(where, dtype);
    at::Tensor out = at::empty(at::infer_size(at::infer_size(s.sizes(), a.sizes()), b.sizes()), s.options());
    at::TensorIterator iter = at::TensorIteratorConfig()
        .add_output(out)
        .add_const_input(s)
        .add_const_input(a)
        .add_const_input(b)
        .build();
    uint64_t numel = iter.numel();
    if (numel == 0) return out;
    int32_t out_dims = static_cast<int32_t>(iter.ndim());
    if (out_dims > MAX_DIMS || !fits_32bit_indexing(iter)) return std::nullopt;

    uint32_t contiguous = iter.is_contiguous() ? 1 : 0;
    uint32_t workgroupSizeX = get_dtype_workgroup_size(dtype, 1, device);
    SpecializationBuilder spd{};
    spd.push(contiguous)
       .push(out_dims)
       .push(workgroupSizeX)
       .push(divide ? 1u : 0u)
       .push(value.toComplexDouble() != c10::complex<double>(1.0) ? 1u : 0u);

    MetadataBuilder metadataBuilder = get_iter_metadata(iter, {1, 2, 3});
    Metadata metadata = metadataBuilder.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(s))
       .push(get_tensor_address(a))
       .push(get_tensor_address(b))
       .push(get_output_address(out))
       .push_metadata(contiguous ? nullptr : &metadata)
       .push(numel)
       .push_scalar(value, is_low_prec(dtype) ? at::kFloat : dtype);  // the kernel's accumulate type

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::ADDC, dtype},
        device->default_subgroup_size,
        spd.build(),
        pcs.build(),
        /* read = */ {s, a, b},
        /* write = */ {out},
        get_group_count(numel, workgroupSizeX), 1, 1
    );
    return out;
}

// self + value * (tensor1 op tensor2); half and bfloat16 compute it all in float32 and round once, as the CPU
at::Tensor addc_vulkan(const at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2, const at::Scalar& value, bool divide)
{
    at::native::ResultTypeState state{};
    for (const at::Tensor* t : {&self, &tensor1, &tensor2}) state = at::native::update_result_type_state(*t, state);
    at::ScalarType dtype = at::native::result_type(state);
    if (!at::isComplexType(dtype) && dtype != at::kBool) {
        if (std::optional<at::Tensor> fused = dispatch_addc_shader(self, tensor1, tensor2, value, divide, dtype)) return *fused;
    }
    bool low = is_low_prec(dtype);
    auto widen = [low](const at::Tensor& t) { return low ? t.to(at::kFloat) : t; };
    at::Tensor combined = divide ? torchvulkan::divide_vulkan(widen(tensor1), widen(tensor2)) : torchvulkan::multiply_vulkan(widen(tensor1), widen(tensor2));
    // a bool value (value=True) scales like 1, without add's boolean-alpha check
    at::Scalar scale = value.isBoolean() ? at::Scalar(static_cast<int64_t>(value.toBool())) : value;
    at::Tensor result = torchvulkan::add_vulkan(widen(self), combined, scale);
    return low ? result.to(dtype) : result;
}

at::Tensor& addcmul_vulkan_(at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2, const at::Scalar& value) {
    return copy_inplace_result(self, addc_vulkan(self, tensor1, tensor2, value, /*divide=*/false));
}

at::Tensor& addcdiv_vulkan_(at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2, const at::Scalar& value) {
    TORCH_CHECK(!(at::isIntegralType(tensor1.scalar_type(), true) && at::isIntegralType(tensor2.scalar_type(), true)),
        "Integer division with addcdiv is no longer supported, and in a future release addcdiv will perform a true division of tensor1 and tensor2."
    );
    return copy_inplace_result(self, addc_vulkan(self, tensor1, tensor2, value, /*divide=*/true));
}

// as torch computes it: from whichever end is nearer, so weight 1 gives exactly `end`
// the CPU's lerp iterator puts self first, so a dense self of the result's shape sets its layout
at::Tensor in_layout_of(const at::Tensor& self, const at::Tensor& result)
{
    if (self.sizes() != result.sizes() || !self.is_non_overlapping_and_dense() || self.strides() == result.strides()) return result;
    return at::empty_strided(self.sizes(), self.strides(), result.options()).copy_(result);
}

at::Tensor lerp_scalar_layout(const at::Tensor& self, const at::Tensor& end, const at::Scalar& weight);

at::Tensor lerp_scalar_vulkan(const at::Tensor& self, const at::Tensor& end, const at::Scalar& weight) {
    return in_layout_of(self, lerp_scalar_layout(self, end, weight));
}

at::Tensor lerp_scalar_layout(const at::Tensor& self, const at::Tensor& end, const at::Scalar& weight) {
    if (is_low_prec(self.scalar_type())) {
        return lerp_scalar_vulkan(self.to(at::kFloat), end.to(at::kFloat), weight).to(self.scalar_type());  // in float32, as the CPU
    }
    at::Tensor diff = subtract_vulkan(end, self, (int)1);
    
    if (weight.isComplex()) 
    {
        c10::complex<double> w = weight.toComplexDouble();
        if (std::abs(w) < 0.5) return torchvulkan::add_vulkan(self, diff, weight);
        return subtract_vulkan(end, diff, c10::complex<double>(1.0) - w);
    }

    double w = weight.toDouble();
    if (std::abs(w) < 0.5) return torchvulkan::add_vulkan(self, diff, weight);
    return subtract_vulkan(end, diff, 1.0 - w);
}

at::Tensor& lerp_scalar_vulkan_(at::Tensor& self, const at::Tensor& end, const at::Scalar& weight) {
    return copy_inplace_result(self, lerp_scalar_vulkan(self, end, weight));
}

at::Tensor& lerp_scalar_vulkan_out(const at::Tensor& self, const at::Tensor& end, const at::Scalar& weight, at::Tensor& out) {
    return fill_out(out, lerp_scalar_vulkan(self, end, weight));
}

at::Tensor floor_divide_vulkan(const at::Tensor& self, const at::Tensor& other) {
    // an empty result never divides, so never raises
    bool empty = self.numel() == 0 || other.numel() == 0;
    if (!empty && at::isIntegralType(at::result_type(self, other), /*includeBool=*/true)) {
        TORCH_CHECK(!at::eq(other, 0).any().item<bool>(), "ZeroDivisionError");
    }
    return binary_op_vulkan(self, other, (int)1, BinaryOp::FLOOR_DIVIDE, [](const at::Tensor& a, const at::Tensor& b) { return at::floor_divide(a, b); });
}

at::Tensor& floor_divide_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    check_inplace_broadcast(out, self, other);
    return fill_out(out, [&] { return floor_divide_vulkan(self, other); });
}

at::Tensor& floor_divide_vulkan_(at::Tensor& self, const at::Tensor& other) {
    return copy_inplace_result(self, [&] { return floor_divide_vulkan(self, other); });
}

at::Tensor rsub_tensor_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    return binary_op_vulkan(self, other, alpha, BinaryOp::RSUB, [alpha](const at::Tensor& a, const at::Tensor& b) { return at::rsub(a, b, alpha); });
}

// the CPU's two-sided formula, in float32 for half and bfloat16
at::Tensor lerp_tensor_vulkan(const at::Tensor& self, const at::Tensor& end, const at::Tensor& weight) {
    at::ScalarType dtype = self.scalar_type();
    TORCH_CHECK(dtype == end.scalar_type(), "expected dtype ", dtype, " for `end` but got dtype ", end.scalar_type());
    // a 0-dim weight is promoted like a scalar
    TORCH_CHECK(weight.dim() == 0 || dtype == weight.scalar_type(), "expected dtype ", dtype, " for `weight` but got dtype ", weight.scalar_type());
    bool low = is_low_prec(dtype);
    at::Tensor s = low ? self.to(at::kFloat) : self;
    at::Tensor e = low ? end.to(at::kFloat) : end;
    at::Tensor w = weight.to(self.device(), s.scalar_type());
    at::Tensor diff = subtract_vulkan(e, s, (int)1);
    at::Tensor near_self = torchvulkan::add_vulkan(s, torchvulkan::multiply_vulkan(w, diff), (int)1);
    at::Tensor near_end = subtract_vulkan(e, torchvulkan::multiply_vulkan(diff, rsub_scalar_vulkan(w, 1, 1)), (int)1);
    at::Tensor r = at::where(at::lt(at::abs(w), 0.5), near_self, near_end);
    return in_layout_of(self, low ? r.to(dtype) : r);
}

at::Tensor& lerp_tensor_vulkan_out(const at::Tensor& self, const at::Tensor& end, const at::Tensor& weight, at::Tensor& out) {
    return fill_out(out, lerp_tensor_vulkan(self, end, weight));
}

at::Tensor& pow_tensor_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& exponent, at::Tensor& out) {
    return fill_out(out, [&] { return pow_vulkan(self, exponent); });
}

at::Tensor& pow_tensor_scalar_out_vulkan(const at::Tensor& self, const at::Scalar& exponent, at::Tensor& out) {
    return fill_out(out, [&] { return pow_tensor_scalar_vulkan(self, exponent); });
}

at::Tensor& pow_scalar_out_vulkan(const at::Scalar& self, const at::Tensor& exponent, at::Tensor& out) {
    return fill_out(out, [&] { return pow_scalar_vulkan(self, exponent); });
}

at::ScalarType bits_dtype(at::ScalarType dtype) {
    switch (c10::elementSize(dtype)) {
        case 2: return at::kShort;
        case 4: return at::kInt;
        default: return at::kLong;
    }
}

// sign bits read through an integer view, so -0.0 and negative NaNs count as negative
at::Tensor signbit_vulkan(const at::Tensor& self) {
    if (at::isFloatingType(self.scalar_type())) return at::lt(self.contiguous().view(bits_dtype(self.scalar_type())), 0);
    if (at::isSignedType(self.scalar_type())) return at::lt(self, 0);
    return at::zeros(self.sizes(), self.options().dtype(at::kBool));
}

at::Tensor& signbit_out_vulkan(const at::Tensor& self, at::Tensor& out) {
    return fill_exact_out(out, signbit_vulkan(self));
}

at::Tensor& copysign_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    at::ScalarType dtype = at::result_type(self, other);
    if (!at::isFloatingType(dtype)) dtype = c10::typeMetaToScalarType(at::get_default_dtype());
    at::Tensor magnitude = at::abs(self.to(dtype));
    at::Tensor sign_source = other.to(self.device(), dtype);
    if (sign_source.dim() == 0 && magnitude.dim() > 0) sign_source = sign_source.expand(magnitude.sizes());
    at::Tensor negative = signbit_vulkan(sign_source);
    return fill_out(out, at::where(negative, at::neg(magnitude), magnitude));
}

// from zero, the first step is the smallest subnormal with b's sign
at::Tensor& nextafter_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    at::ScalarType dtype = at::result_type(self, other);
    TORCH_CHECK_NOT_IMPLEMENTED(at::isFloatingType(dtype), "\"nextafter_cpu\" not implemented for '", dtype, "'");
    std::vector<int64_t> shape = at::infer_size(self.sizes(), other.sizes());
    at::Tensor a = self.to(dtype).expand(shape).contiguous();
    at::Tensor b = other.to(self.device(), dtype).expand(shape).contiguous();
    at::ScalarType bits = bits_dtype(dtype);

    at::Tensor a_bits = a.view(bits);
    at::Tensor toward_zero = at::ne(at::lt(a, b), at::gt(a, 0));
    at::Tensor stepped = at::where(toward_zero, at::sub(a_bits, 1), at::add(a_bits, 1)).view(dtype);
    int64_t sign_bit = std::numeric_limits<int64_t>::min() >> (64 - 8 * c10::elementSize(dtype));
    at::Tensor smallest = at::where(signbit_vulkan(b), at::full_like(a_bits, sign_bit | 1), at::full_like(a_bits, 1)).view(dtype);

    at::Tensor r = at::where(at::eq(a, 0), smallest, stepped);
    r = at::where(at::eq(a, b), b, r);
    r = at::where(at::logical_or(at::isnan(a), at::isnan(b)), at::add(a, b), r);
    return fill_out(out, r);
}

// as torch: round(x * 10^d) / 10^d, or round(x / 10^-d) * 10^-d for negative d
at::Tensor& round_decimals_out_vulkan(const at::Tensor& self, int64_t decimals, at::Tensor& out) {
    TORCH_CHECK(out.scalar_type() == self.scalar_type(), "Found dtype ", out.scalar_type(), " but expected ", self.scalar_type());
    if (decimals == 0) return fill_out(out, at::round(self));
    TORCH_CHECK_NOT_IMPLEMENTED(at::isFloatingType(self.scalar_type()), "\"round_cpu\" not implemented for '", self.scalar_type(), "'");
    double ten_pow = std::pow(10.0, static_cast<double>(std::abs(decimals)));
    if (decimals > 0) return fill_out(out, at::div(at::round(at::mul(self, ten_pow)), ten_pow));
    return fill_out(out, at::mul(at::round(at::div(self, ten_pow)), ten_pow));
}

// works on the bit pattern alone: Metal flushes subnormals to zero in float arithmetic and comparisons
std::tuple<at::Tensor&, at::Tensor&> frexp_out_vulkan(const at::Tensor& self, at::Tensor& mantissa, at::Tensor& exponent) {
    at::ScalarType dtype = self.scalar_type();
    TORCH_CHECK(at::isFloatingType(dtype), "torch.frexp() only supports floating-point dtypes");

    int64_t mantissa_bits = 0, exponent_mask = 0, bias = 0;
    switch (dtype) {
        case at::kHalf: mantissa_bits = 10; exponent_mask = 0x1f; bias = 15; break;
        case at::kBFloat16: mantissa_bits = 7; exponent_mask = 0xff; bias = 127; break;
        case at::kFloat: mantissa_bits = 23; exponent_mask = 0xff; bias = 127; break;
        default: mantissa_bits = 52; exponent_mask = 0x7ff; bias = 1023; break;
    }
    int64_t mantissa_mask = (int64_t(1) << mantissa_bits) - 1;
    int64_t half_exponent = (bias - 1) << mantissa_bits;

    at::Tensor bits = self.contiguous().view(bits_dtype(dtype)).to(at::kLong);
    at::Tensor field = at::bitwise_and(at::bitwise_right_shift(bits, mantissa_bits), exponent_mask);
    at::Tensor fraction = at::bitwise_and(bits, mantissa_mask);
    at::Tensor sign = at::bitwise_and(bits, ~(exponent_mask << mantissa_bits | mantissa_mask));

    // a subnormal's leading bit, from the exponent of float(fraction); rounding up can overshoot by one
    at::Tensor lead = at::sub(at::bitwise_and(at::bitwise_right_shift(fraction.to(at::kFloat).view(at::kInt).to(at::kLong), 23), 0xff), 127);
    lead = at::where(at::gt(at::bitwise_left_shift(at::ones_like(lead), lead), fraction), at::sub(lead, 1), lead);

    at::Tensor normal_exponent = at::sub(field, bias - 1);
    at::Tensor normal_bits = at::bitwise_or(at::bitwise_or(sign, half_exponent), fraction);
    at::Tensor subnormal_exponent = at::add(lead, 2 - bias - mantissa_bits);
    at::Tensor subnormal_fraction = at::bitwise_and(at::bitwise_left_shift(fraction, at::rsub(lead, mantissa_bits)), mantissa_mask);
    at::Tensor subnormal_bits = at::bitwise_or(at::bitwise_or(sign, half_exponent), subnormal_fraction);

    at::Tensor is_subnormal = at::logical_and(at::eq(field, 0), at::ne(fraction, 0));
    at::Tensor is_special = at::logical_or(at::logical_and(at::eq(field, 0), at::eq(fraction, 0)), at::eq(field, exponent_mask));

    at::Tensor m_bits = at::where(is_subnormal, subnormal_bits, normal_bits);
    m_bits = at::where(is_special, bits, m_bits);
    at::Tensor e = at::where(is_subnormal, subnormal_exponent, normal_exponent);
    e = at::where(is_special, at::zeros_like(e), e);

    fill_out(mantissa, m_bits.to(bits_dtype(dtype)).view(dtype).view(self.sizes()));
    fill_out(exponent, e.to(at::kInt).view(self.sizes()));
    return std::forward_as_tuple(mantissa, exponent);
}

at::Tensor div_mode_vulkan(const at::Tensor& self, const at::Tensor& other, c10::optional<c10::string_view> rounding_mode)
{
    if (!rounding_mode.has_value()) return torchvulkan::divide_vulkan(self, other);
    TORCH_CHECK(*rounding_mode == "trunc" || *rounding_mode == "floor",
        "div expected rounding_mode to be one of None, 'trunc', or 'floor' but found '", *rounding_mode, "'"
    );
    if (*rounding_mode == "floor") return floor_divide_vulkan(self, other);
    at::ScalarType dtype = at::result_type(self, other);
    if (is_low_prec(dtype)) {
        // the float32 quotient is rounded to the dtype, then truncated, as the CPU does
        return at::trunc(torchvulkan::divide_vulkan(self.to(at::kFloat), other.to(at::kFloat)).to(dtype));
    }
    if (!at::isIntegralType(dtype, /*includeBool=*/true)) return at::trunc(torchvulkan::divide_vulkan(self, other));
    at::Tensor floored = floor_divide_vulkan(self, other);
    at::Tensor inexact = at::ne(fmod_vulkan(self, other), 0);
    at::Tensor signs_differ = at::ne(at::lt(self, 0), at::lt(other, 0));
    return at::where(at::logical_and(inexact, signs_differ), at::add(floored, 1), floored);
}

at::Tensor& div_out_mode_vulkan(const at::Tensor& self, const at::Tensor& other, c10::optional<c10::string_view> rounding_mode, at::Tensor& out)
{
    return fill_out(out, div_mode_vulkan(self, other, rounding_mode));
}

at::Tensor& addcmul_out_vulkan(const at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2, const at::Scalar& value, at::Tensor& out)
{
    return fill_out(out, addc_vulkan(self, tensor1, tensor2, value, /*divide=*/false));
}

at::Tensor& addcdiv_out_vulkan(const at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2, const at::Scalar& value, at::Tensor& out)
{
    TORCH_CHECK(!(at::isIntegralType(tensor1.scalar_type(), true) && at::isIntegralType(tensor2.scalar_type(), true)),
        "Integer division with addcdiv is no longer supported, and in a future release addcdiv will perform a true division of tensor1 and tensor2."
    );
    return fill_out(out, addc_vulkan(self, tensor1, tensor2, value, /*divide=*/true));
}

// torch's foreach ops with a scalar tensor run its slow kernels here, so make the checks its accelerator kernels make
// (mul and div also take a CPU scalar tensor)
void check_foreach_scalar_tensor(at::TensorList tensors, const at::Tensor& scalar, bool cpu_ok)
{
    TORCH_CHECK(scalar.dim() == 0 && scalar.numel() == 1, "scalar tensor expected to be 0 dim but it has ", scalar.dim(), " dimensions and ",
        scalar.numel(), " elements.");
    bool one_device = std::all_of(tensors.begin(), tensors.end(), [&](const at::Tensor& t) { return t.device() == tensors[0].device(); });
    if (!tensors.empty() && one_device && tensors[0].is_privateuseone() && !(cpu_ok && scalar.is_cpu())) {
        TORCH_CHECK(scalar.device() == tensors[0].device(), "scalar tensor expected to be on ", tensors[0].device(), " but is on ", scalar.device());
    }
}

std::vector<at::Tensor> foreach_add_tensor_vulkan(at::TensorList tensors, const at::Tensor& scalar, const at::Scalar& alpha)
{
    check_foreach_scalar_tensor(tensors, scalar, /*cpu_ok=*/false);
    return at::native::foreach_tensor_add_tensor_kernel_slow(tensors, scalar, alpha);
}

void foreach_add_tensor_vulkan_(at::TensorList tensors, const at::Tensor& scalar, const at::Scalar& alpha)
{
    check_foreach_scalar_tensor(tensors, scalar, /*cpu_ok=*/false);
    at::native::foreach_tensor_add_tensor_kernel_slow_(tensors, scalar, alpha);
}

std::vector<at::Tensor> foreach_mul_tensor_vulkan(at::TensorList tensors, const at::Tensor& scalar)
{
    check_foreach_scalar_tensor(tensors, scalar, /*cpu_ok=*/true);
    return at::native::foreach_tensor_mul_tensor_kernel_slow(tensors, scalar);
}

void foreach_mul_tensor_vulkan_(at::TensorList tensors, const at::Tensor& scalar)
{
    check_foreach_scalar_tensor(tensors, scalar, /*cpu_ok=*/true);
    at::native::foreach_tensor_mul_tensor_kernel_slow_(tensors, scalar);
}

void ldexp_vulkan_kernel(at::TensorIteratorBase& iter)
{
    at::Tensor out = iter.tensor(0);
    const at::Tensor& self = iter.tensor(1);
    bool wide = self.scalar_type() == at::kDouble;
    int64_t limit = wide ? 2200 : 300;
    at::Tensor x = self.to(wide ? at::kDouble : at::kFloat);
    at::Tensor e = at::clamp(iter.tensor(2).to(at::kLong), -limit, limit);
    at::Tensor first = at::div(e, 3, "trunc");
    at::Tensor second = at::div(at::sub(e, first), 2, "trunc");
    at::Tensor third = at::sub(at::sub(e, first), second);
    at::Tensor two = at::scalar_tensor(2.0, x.options());
    at::Tensor result = at::mul(at::mul(at::mul(x, at::pow(two, first)), at::pow(two, second)), at::pow(two, third));
    out.copy_(result);
}

} // namespace

namespace at::native {
REGISTER_PRIVATEUSE1_DISPATCH(ldexp_stub, &ldexp_vulkan_kernel)
} // namespace at::native

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("_foreach_add.Tensor", &foreach_add_tensor_vulkan);
    m.impl("_foreach_add_.Tensor", &foreach_add_tensor_vulkan_);
    m.impl("_foreach_mul.Tensor", &foreach_mul_tensor_vulkan);
    m.impl("_foreach_mul_.Tensor", &foreach_mul_tensor_vulkan_);
    m.impl("add.Tensor", &torchvulkan::add_vulkan);
    m.impl("add.Scalar", &add_scalar_vulkan);
    m.impl("add_.Tensor", &add_vulkan_);
    m.impl("add_.Scalar", &add_scalar_vulkan_);
    m.impl("add.out", &add_out_vulkan);
    m.impl("sub.out", &subtract_out_vulkan);
    m.impl("mul.out", &multiply_out_vulkan);
    m.impl("div.out", &divide_out_vulkan);
    m.impl("atan2.out", &atan2_out_vulkan);
    m.impl("fmax.out", &fmax_out_vulkan);
    m.impl("fmin.out", &fmin_out_vulkan);
    m.impl("fmod.Tensor_out", &fmod_out_vulkan);
    m.impl("hypot.out", &hypot_out_vulkan);
    m.impl("logaddexp.out", &logaddexp_out_vulkan);
    m.impl("logaddexp2.out", &logaddexp2_out_vulkan);
    m.impl("maximum.out", &maximum_out_vulkan);
    m.impl("minimum.out", &minimum_out_vulkan);
    m.impl("remainder.Tensor_out", &remainder_out_vulkan);
    m.impl("remainder.Scalar_Tensor", &remainder_scalar_tensor_vulkan);
    m.impl("xlogy.OutTensor", &xlogy_out_vulkan);
    m.impl("threshold_backward.grad_input", &threshold_backward_out_vulkan);
    m.impl("_add_relu.Tensor", &add_relu_vulkan);
    m.impl("_add_relu.Scalar", &add_relu_scalar_vulkan);
    m.impl("_add_relu.out", &add_relu_out_vulkan);
    m.impl("_add_relu_.Tensor", &add_relu_vulkan_);
    m.impl("_add_relu_.Scalar", &add_relu_scalar_vulkan_);
    m.impl("sub.Tensor", &subtract_vulkan);
    m.impl("sub.Scalar", &subtract_scalar_vulkan);
    m.impl("sub_.Tensor", &subtract_vulkan_);
    m.impl("sub_.Scalar", &subtract_scalar_vulkan_);
    m.impl("rsub.Scalar", &rsub_scalar_vulkan);
    m.impl("mul.Tensor", &torchvulkan::multiply_vulkan);
    m.impl("mul.Scalar", &multiply_scalar_vulkan);
    m.impl("mul_.Scalar", &multiply_scalar_vulkan_);
    m.impl("div.Tensor", &torchvulkan::divide_vulkan);
    m.impl("div.Scalar", &torchvulkan::divide_scalar_vulkan);
    m.impl("div_.Tensor", &divide_vulkan_);
    m.impl("div_.Scalar", &divide_scalar_vulkan_);
    m.impl("maximum", &maximum_vulkan);
    m.impl("minimum", &minimum_vulkan);
    m.impl("pow.Tensor_Tensor", &pow_vulkan);
    m.impl("pow.Tensor_Scalar", &pow_tensor_scalar_vulkan);
    m.impl("pow.Scalar", &pow_scalar_vulkan);
    m.impl("atan2", &atan2_vulkan);
    m.impl("fmax", &fmax_vulkan);
    m.impl("fmin", &fmin_vulkan);
    m.impl("fmod.Tensor", &fmod_vulkan);
    m.impl("fmod.Scalar", &fmod_scalar_vulkan);
    m.impl("remainder.Tensor", &remainder_vulkan);
    m.impl("remainder.Scalar", &remainder_scalar_vulkan);
    m.impl("hypot", &hypot_vulkan);
    m.impl("xlogy.Tensor", &xlogy_vulkan);
    m.impl("logaddexp", &logaddexp_vulkan);
    m.impl("logaddexp2", &logaddexp2_vulkan);
    m.impl("clamp", &clamp_scalar_vulkan);
    m.impl("clamp.Tensor", &clamp_tensor_vulkan);
    m.impl("clamp_min", &clamp_min_scalar_vulkan);
    m.impl("clamp_min.Tensor", &clamp_min_tensor_vulkan);
    m.impl("clamp_max", &clamp_max_scalar_vulkan);
    m.impl("clamp_max.Tensor", &clamp_max_tensor_vulkan);
    m.impl("clamp_min.out", &clamp_min_out_vulkan);
    m.impl("clamp_max.out", &clamp_max_out_vulkan);
    m.impl("clamp_min.Tensor_out", &clamp_min_tensor_out_vulkan);
    m.impl("clamp_max.Tensor_out", &clamp_max_tensor_out_vulkan);
    m.impl("clamp.out", &clamp_out_vulkan);
    m.impl("clamp.Tensor_out", &clamp_tensor_out_vulkan);
    m.impl("addcmul_", &addcmul_vulkan_);
    m.impl("addcdiv_", &addcdiv_vulkan_);
    m.impl("lerp.Scalar", &lerp_scalar_vulkan);
    m.impl("lerp_.Scalar", &lerp_scalar_vulkan_);
    m.impl("lerp.Scalar_out", &lerp_scalar_vulkan_out);
    m.impl("threshold_backward", &threshold_backward_vulkan);
    m.impl("gelu_backward", &gelu_backward_vulkan);
    m.impl("gelu_backward.grad_input", &gelu_backward_out_vulkan);
    m.impl("floor_divide", &floor_divide_vulkan);
    m.impl("div.Tensor_mode", &div_mode_vulkan);
    m.impl("div.out_mode", &div_out_mode_vulkan);
    m.impl("addcmul.out", &addcmul_out_vulkan);
    m.impl("addcdiv.out", &addcdiv_out_vulkan);
    m.impl("floor_divide.out", &floor_divide_out_vulkan);
    m.impl("floor_divide_.Tensor", &floor_divide_vulkan_);
    m.impl("rsub.Tensor", &rsub_tensor_vulkan);
    m.impl("lerp.Tensor", &lerp_tensor_vulkan);
    m.impl("lerp.Tensor_out", &lerp_tensor_vulkan_out);
    m.impl("pow.Tensor_Tensor_out", &pow_tensor_tensor_out_vulkan);
    m.impl("pow.Tensor_Scalar_out", &pow_tensor_scalar_out_vulkan);
    m.impl("pow.Scalar_out", &pow_scalar_out_vulkan);
    m.impl("signbit.out", &signbit_out_vulkan);
    m.impl("copysign.out", &copysign_out_vulkan);
    m.impl("nextafter.out", &nextafter_out_vulkan);
    m.impl("round.decimals_out", &round_decimals_out_vulkan);
    m.impl("frexp.Tensor_out", &frexp_out_vulkan);
}
