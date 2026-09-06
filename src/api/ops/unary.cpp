#include <torch/extension.h>
#include "api/ops/unary.h"
#include "api/ops/helpers.h"

at::Tensor& torchvulkan::fill_scalar_vulkan(
    at::Tensor& self, 
    const at::Scalar& value)
{
    at::TensorIterator iter = at::TensorIteratorConfig()
        .add_output(self)
        .add_input(self)
        .build();

    uint64_t numel = iter.numel();
    if (numel == 0) return self;
    
    if (!is_dtype_supported(iter.dtype())) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Vulkan device does not support ", iter.dtype(), ". Falling back to CPU.");
        at::Tensor cpu_temp = self.to(at::kCPU);
        cpu_temp.fill_(value);        
        self.copy_(cpu_temp); 
        return self;
    }

    int32_t out_dims = static_cast<int32_t>(iter.ndim());
    if (out_dims > MAX_DIMS) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Coalesced dimensions (", out_dims, ") exceed maximum supported (", MAX_DIMS, "). Falling back to CPU.");
        at::Tensor cpu_temp = self.to(at::kCPU);
        cpu_temp.fill_(value);
        self.copy_(cpu_temp);
        return self;
    }

    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    torchvulkan::ShaderID shader_id = torchvulkan::get_shader_id_fill(self.scalar_type());
    uint32_t vecSize = get_dtype_vec_size(self.scalar_type()); // our workgroup must match the shader workgroup
    uint32_t workgroupSizeX = get_dtype_workgroup_size(self.scalar_type(), vecSize);
    uint32_t contiguous = iter.is_contiguous() ? 1 : 0;

    SpecializationBuilder spd{};
    spd.push(out_dims)
        .push(contiguous)
        .push(workgroupSizeX);
    uint32_t key = (workgroupSizeX << 8) | (contiguous << 4) | out_dims;
    SpecializationArgs specialization = {spd.data(), spd.offsets(), spd.sizes(), spd.numConstants(), key};

    IntDivider sizes; 
    uint32_t strides_in[MAX_DIMS] = {0};
    
    int64_t el_size = iter.element_size(0);
    at::IntArrayRef iter_shape = iter.shape();
    at::IntArrayRef iter_strides_in = iter.strides(0);

    uint64_t metadata_address = 0;
    if (!contiguous) 
    {
        for (int i = 0; i < out_dims; i++) {
            sizes.set(i, iter_shape[i]);
            strides_in[i] = iter_strides_in[i] / el_size;
        }

        MetadataBuilder metadataBuilder{};
        metadataBuilder.push(sizes, out_dims)
                       .push_array(strides_in, out_dims);
        Metadata metadata = metadataBuilder.build();
        metadata_address = device->shader_manager->registerMetadata(metadata);
    }

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(self))
        .push(metadata_address)
        .push(numel)
        .push_scalar(value, self.scalar_type());

    uint32_t groupX = (numel + (workgroupSizeX - 1)) / workgroupSizeX;

    PushConstants pushConstants = { const_cast<void*>(pcs.data()), pcs.size() };
    device->shader_manager->dispatchShader(
        shader_id,
        specialization,
        pushConstants,
        /* read = */ {},
        /* write = */ {self},
        groupX, 1, 1
    );

    return self;
}

at::Tensor& torchvulkan::fill_tensor_vulkan(at::Tensor& self, const at::Tensor& value) 
{
    return torchvulkan::fill_scalar_vulkan(self, value.item());
}

at::Tensor& torchvulkan::zero_vulkan(at::Tensor& self)
{
    return fill_scalar_vulkan(self, 0);
}

at::Tensor torchvulkan::unary_op_vulkan(
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

namespace {

at::Tensor promote_to_float(const at::Tensor& self)
{
    if (self.is_floating_point() || self.is_complex()) return self;
    return self.to(c10::typeMetaToScalarType(at::get_default_dtype()));
}

} // namespace

at::Tensor torchvulkan::relu_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(self, UnaryOp::RELU, [](const at::Tensor& a) { return at::relu(a); });
}

at::Tensor torchvulkan::exp_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::EXP, [](const at::Tensor& a) { return at::exp(a); });
}

at::Tensor torchvulkan::log_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::LOG, [](const at::Tensor& a) { return at::log(a); });
}

at::Tensor torchvulkan::sqrt_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::SQRT, [](const at::Tensor& a) { return at::sqrt(a); });
}

at::Tensor torchvulkan::neg_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(self, UnaryOp::NEG, [](const at::Tensor& a) { return at::neg(a); });
}

at::Tensor torchvulkan::reciprocal_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::RECIPROCAL, [](const at::Tensor& a) { return at::reciprocal(a); });
}

at::Tensor torchvulkan::sin_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::SIN, [](const at::Tensor& a) { return at::sin(a); });
}

at::Tensor torchvulkan::cos_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::COS, [](const at::Tensor& a) { return at::cos(a); });
}

at::Tensor torchvulkan::tan_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::TAN, [](const at::Tensor& a) { return at::tan(a); });
}

at::Tensor torchvulkan::asin_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::ASIN, [](const at::Tensor& a) { return at::asin(a); });
}

at::Tensor torchvulkan::acos_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::ACOS, [](const at::Tensor& a) { return at::acos(a); });
}

at::Tensor torchvulkan::atan_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::ATAN, [](const at::Tensor& a) { return at::atan(a); });
}

at::Tensor torchvulkan::sinh_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::SINH, [](const at::Tensor& a) { return at::sinh(a); });
}

at::Tensor torchvulkan::cosh_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::COSH, [](const at::Tensor& a) { return at::cosh(a); });
}

at::Tensor torchvulkan::tanh_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::TANH, [](const at::Tensor& a) { return at::tanh(a); });
}

at::Tensor torchvulkan::exp2_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::EXP2, [](const at::Tensor& a) { return at::exp2(a); });
}

at::Tensor torchvulkan::log2_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::LOG2, [](const at::Tensor& a) { return at::log2(a); });
}

at::Tensor torchvulkan::log10_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::LOG10, [](const at::Tensor& a) { return at::log10(a); });
}

at::Tensor torchvulkan::expm1_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::EXPM1, [](const at::Tensor& a) { return at::expm1(a); });
}

at::Tensor torchvulkan::log1p_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::LOG1P, [](const at::Tensor& a) { return at::log1p(a); });
}

at::Tensor torchvulkan::rsqrt_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::RSQRT, [](const at::Tensor& a) { return at::rsqrt(a); });
}

at::Tensor torchvulkan::sigmoid_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::SIGMOID, [](const at::Tensor& a) { return at::sigmoid(a); });
}

at::Tensor torchvulkan::asinh_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::ASINH, [](const at::Tensor& a) { return at::asinh(a); });
}

at::Tensor torchvulkan::acosh_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::ACOSH, [](const at::Tensor& a) { return at::acosh(a); });
}

at::Tensor torchvulkan::atanh_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::ATANH, [](const at::Tensor& a) { return at::atanh(a); });
}

at::Tensor torchvulkan::deg2rad_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::DEG2RAD, [](const at::Tensor& a) { return at::deg2rad(a); });
}

at::Tensor torchvulkan::rad2deg_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::RAD2DEG, [](const at::Tensor& a) { return at::rad2deg(a); });
}

at::Tensor torchvulkan::floor_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(self, UnaryOp::FLOOR, [](const at::Tensor& a) { return at::floor(a); });
}

at::Tensor torchvulkan::ceil_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(self, UnaryOp::CEIL, [](const at::Tensor& a) { return at::ceil(a); });
}

at::Tensor torchvulkan::trunc_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(self, UnaryOp::TRUNC, [](const at::Tensor& a) { return at::trunc(a); });
}

at::Tensor torchvulkan::round_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(self, UnaryOp::ROUND, [](const at::Tensor& a) { return at::round(a); });
}

at::Tensor torchvulkan::frac_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(self, UnaryOp::FRAC, [](const at::Tensor& a) { return at::frac(a); });
}

at::Tensor torchvulkan::abs_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(self, UnaryOp::ABS, [](const at::Tensor& a) { return at::abs(a); });
}

at::Tensor torchvulkan::sign_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(self, UnaryOp::SIGN, [](const at::Tensor& a) { return at::sign(a); });
}

at::Tensor torchvulkan::erf_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::ERF, [](const at::Tensor& a) { return at::erf(a); });
}

at::Tensor torchvulkan::erfinv_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::ERFINV, [](const at::Tensor& a) { return at::erfinv(a); });
}

at::Tensor torchvulkan::sinc_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::SINC, [](const at::Tensor& a) { return at::sinc(a); });
}

at::Tensor torchvulkan::entr_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::ENTR, [](const at::Tensor& a) { return at::special_entr(a); });
}

at::Tensor torchvulkan::lgamma_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::LGAMMA, [](const at::Tensor& a) { return at::lgamma(a); });
}

at::Tensor torchvulkan::digamma_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::DIGAMMA, [](const at::Tensor& a) { return at::digamma(a); });
}

at::Tensor torchvulkan::i0_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::I0, [](const at::Tensor& a) { return at::i0(a); });
}

at::Tensor torchvulkan::i1_vulkan(const at::Tensor& self)
{
    return unary_op_vulkan(promote_to_float(self), UnaryOp::I1, [](const at::Tensor& a) { return at::special_i1(a); });
}

at::Tensor torchvulkan::silu_vulkan(const at::Tensor& self)
{
    return at::mul(self, at::sigmoid(self));
}

at::Tensor torchvulkan::hardtanh_vulkan(const at::Tensor& self, const at::Scalar& min_val, const at::Scalar& max_val)
{
    return at::clamp(self, min_val, max_val);
}

at::Tensor torchvulkan::hardsigmoid_vulkan(const at::Tensor& self)
{
    return at::div(at::clamp(at::add(self, 3), 0, 6), 6);
}

at::Tensor torchvulkan::hardswish_vulkan(const at::Tensor& self)
{
    return at::mul(self, at::div(at::clamp(at::add(self, 3), 0, 6), 6));
}

at::Tensor torchvulkan::leaky_relu_vulkan(const at::Tensor& self, const at::Scalar& negative_slope)
{
    at::Tensor pos = at::clamp_min(self, 0);
    at::Tensor neg = at::clamp_max(self, 0);
    return at::add(pos, at::mul(neg, negative_slope));
}

at::Tensor torchvulkan::elu_vulkan(const at::Tensor& self, const at::Scalar& alpha, const at::Scalar& scale, const at::Scalar& input_scale)
{
    at::Tensor pos = at::clamp_min(self, 0);
    at::Tensor inner = at::mul(at::expm1(at::mul(self, input_scale)), alpha);
    at::Tensor neg = at::clamp_max(inner, 0);
    return at::mul(at::add(pos, neg), scale);
}

at::Tensor torchvulkan::softplus_vulkan(const at::Tensor& self, const at::Scalar& beta, const at::Scalar& threshold)
{
    at::Tensor bx = at::mul(self, beta);
    at::Tensor sp = at::div(at::add(at::clamp_min(bx, 0), at::log1p(at::exp(at::neg(at::abs(bx))))), beta);
    at::Tensor mask = at::clamp(at::sign(at::sub(bx, threshold)), 0, 1);
    return at::add(at::mul(mask, self), at::mul(at::rsub(mask, 1), sp));
}

at::Tensor torchvulkan::mish_vulkan(const at::Tensor& self)
{
    return at::mul(self, at::tanh(softplus_vulkan(self, 1, 20)));
}

at::Tensor torchvulkan::tanhshrink_vulkan(const at::Tensor& self)
{
    return at::sub(self, at::tanh(self));
}

at::Tensor torchvulkan::square_vulkan(const at::Tensor& self)
{
    if (self.scalar_type() == at::kBool) {
        at::Tensor t = self.to(at::kLong);
        return at::mul(t, t);
    }
    return at::mul(self, self);
}

at::Tensor torchvulkan::logit_vulkan(const at::Tensor& self, c10::optional<double> eps)
{
    at::Tensor x = promote_to_float(self);
    if (eps.has_value()) x = at::clamp(x, *eps, 1.0 - *eps);
    return at::log(at::div(x, at::rsub(x, 1)));
}