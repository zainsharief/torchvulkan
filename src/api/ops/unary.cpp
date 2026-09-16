#include <torch/extension.h>
#include <c10/core/MemoryFormat.h>
#include <functional>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "shaders/shader_registry.h"
#include "api/ops/helpers.h"

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
    I1 = 41
};

at::Tensor unary_op_vulkan(
    const at::Tensor& src,
    UnaryOp operation,
    const std::function<at::Tensor(const at::Tensor&)>& fallback)
{
    if (!is_dtype_supported(src.scalar_type())) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Vulkan device does not support ", src.scalar_type(), ". Falling back to CPU.");
        return fallback(src.cpu()).to(src.device());
    }

    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();

    uint32_t alignment = device->properties.limits.minStorageBufferOffsetAlignment;
    at::Tensor src_in = src;
    if ((src_in.storage_offset() * src_in.element_size()) % alignment != 0) {
        src_in = src_in.clone();
    }

    at::Tensor dst = at::empty_like(src_in);

    at::TensorIterator iter = at::TensorIteratorConfig()
        .set_check_mem_overlap(true)
        .add_output(dst)
        .add_input(src_in)
        .build();

    uint64_t numel = iter.numel();
    if (numel == 0) return dst;

    int32_t out_dims = static_cast<int32_t>(iter.ndim());
    if (out_dims > MAX_DIMS) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Coalesced dimensions (", out_dims, ") exceed maximum supported (", MAX_DIMS, "). Falling back to CPU.");
        return fallback(src.cpu()).to(src.device());
    }

    uint32_t vecSize = get_dtype_vec_size(dst.scalar_type());
    uint32_t workgroupSizeX = get_dtype_workgroup_size(dst.scalar_type(), vecSize);

    uint32_t contiguous = iter.is_contiguous() ? 1 : 0;
    torchvulkan::ShaderID shader_id = torchvulkan::get_shader_id_unaryop(dst.scalar_type());
    uint32_t op = static_cast<uint32_t>(operation);

    SpecializationBuilder spd{};
    spd.push(op)
        .push(contiguous)
        .push(out_dims)
        .push(workgroupSizeX);
    uint32_t key = (workgroupSizeX << 11) | (out_dims << 7) | (contiguous << 6) | op;
    SpecializationArgs specialization = {spd.data(), spd.offsets(), spd.sizes(), spd.numConstants(), key};

    IntDivider sizes;
    uint32_t strides_in[MAX_DIMS] = {0};
    uint32_t strides_out[MAX_DIMS] = {0};

    uint64_t metadata_address = 0;
    if (!contiguous) {
        int64_t el_size = iter.element_size(0);
        at::IntArrayRef iter_shape = iter.shape();
        at::IntArrayRef iter_strides_out = iter.strides(0);
        at::IntArrayRef iter_strides_in = iter.strides(1);

        for (int i = 0; i < out_dims; i++) {
            sizes.set(i, iter_shape[i]);
            strides_in[i] = iter_strides_in[i] / el_size;
            strides_out[i] = iter_strides_out[i] / el_size;
        }

        MetadataBuilder metadataBuilder{};
        metadataBuilder.push(sizes, out_dims)
                       .push_array(strides_in, out_dims)
                       .push_array(strides_out, out_dims);
        Metadata metadata = metadataBuilder.build();
        metadata_address = device->shader_manager->registerMetadata(metadata);
    }

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(src_in))
        .push(get_tensor_address(dst))
        .push(metadata_address)
        .push(numel);

    uint64_t numel_vec = !contiguous ? numel : (numel + (vecSize - 1)) / vecSize;
    uint32_t groupX = (numel_vec + (workgroupSizeX - 1)) / workgroupSizeX;

    PushConstants pushConstants = { const_cast<void*>(pcs.data()), pcs.size() };
    device->shader_manager->dispatchShader(
        shader_id,
        specialization,
        pushConstants,
        /* read = */ {src_in},
        /* write = */ {dst},
        groupX, 1, 1
    );

    return dst;
}

at::Tensor relu_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(self, UnaryOp::RELU, [](const at::Tensor& a) { return at::relu(a); });
}

at::Tensor exp_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::EXP, [](const at::Tensor& a) { return at::exp(a); });
}

at::Tensor log_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::LOG, [](const at::Tensor& a) { return at::log(a); });
}

at::Tensor sqrt_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::SQRT, [](const at::Tensor& a) { return at::sqrt(a); });
}

at::Tensor neg_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(self, UnaryOp::NEG, [](const at::Tensor& a) { return at::neg(a); });
}

at::Tensor reciprocal_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::RECIPROCAL, [](const at::Tensor& a) { return at::reciprocal(a); });
}

at::Tensor sin_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::SIN, [](const at::Tensor& a) { return at::sin(a); });
}

at::Tensor cos_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::COS, [](const at::Tensor& a) { return at::cos(a); });
}

at::Tensor tan_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::TAN, [](const at::Tensor& a) { return at::tan(a); });
}

at::Tensor asin_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::ASIN, [](const at::Tensor& a) { return at::asin(a); });
}

at::Tensor acos_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::ACOS, [](const at::Tensor& a) { return at::acos(a); });
}

at::Tensor atan_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::ATAN, [](const at::Tensor& a) { return at::atan(a); });
}

at::Tensor sinh_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::SINH, [](const at::Tensor& a) { return at::sinh(a); });
}

at::Tensor cosh_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::COSH, [](const at::Tensor& a) { return at::cosh(a); });
}

at::Tensor tanh_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::TANH, [](const at::Tensor& a) { return at::tanh(a); });
}

at::Tensor exp2_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::EXP2, [](const at::Tensor& a) { return at::exp2(a); });
}

at::Tensor log2_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::LOG2, [](const at::Tensor& a) { return at::log2(a); });
}

at::Tensor log10_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::LOG10, [](const at::Tensor& a) { return at::log10(a); });
}

at::Tensor expm1_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::EXPM1, [](const at::Tensor& a) { return at::expm1(a); });
}

at::Tensor log1p_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::LOG1P, [](const at::Tensor& a) { return at::log1p(a); });
}

at::Tensor rsqrt_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::RSQRT, [](const at::Tensor& a) { return at::rsqrt(a); });
}

at::Tensor sigmoid_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::SIGMOID, [](const at::Tensor& a) { return at::sigmoid(a); });
}

at::Tensor asinh_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::ASINH, [](const at::Tensor& a) { return at::asinh(a); });
}

at::Tensor acosh_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::ACOSH, [](const at::Tensor& a) { return at::acosh(a); });
}

at::Tensor atanh_vulkan(const at::Tensor& self)
{
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
    return unary_op_vulkan(self, UnaryOp::ABS, [](const at::Tensor& a) { return at::abs(a); });
}

at::Tensor sign_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(self, UnaryOp::SIGN, [](const at::Tensor& a) { return at::sign(a); });
}

at::Tensor erf_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::ERF, [](const at::Tensor& a) { return at::erf(a); });
}

at::Tensor erfinv_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::ERFINV, [](const at::Tensor& a) { return at::erfinv(a); });
}

at::Tensor sinc_vulkan(const at::Tensor& self)
{
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

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("relu", &relu_vulkan);
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
    m.impl("sgn", &sign_vulkan);
    m.impl("erf", &erf_vulkan);
    m.impl("erfinv", &erfinv_vulkan);
    m.impl("sinc", &sinc_vulkan);
    m.impl("special_entr", &entr_vulkan);
    m.impl("lgamma", &lgamma_vulkan);
    m.impl("digamma", &digamma_vulkan);
    m.impl("i0", &i0_vulkan);
    m.impl("special_i1", &i1_vulkan);
}
