#include <torch/extension.h>
#include <ATen/TensorIterator.h>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "api/ops/helpers.h"
#include "api/fallback.h"

namespace {

enum class IntegerBinaryOp {
    AND = 0,
    OR = 1,
    XOR = 2,
    LSHIFT = 3,
    RSHIFT = 4,
    GCD = 5,
    LCM = 6
};

bool is_cpu_scalar(const at::Tensor& t)
{
    return t.unsafeGetTensorImpl()->is_wrapped_number() || (t.dim() == 0 && t.device().is_cpu());
}

bool dispatch_integer_binary_shader(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& scalar,
    at::Tensor& out,
    IntegerBinaryOp operation)
{
    DeviceContext* device = get_device_context(out);
    at::ScalarType dtype = out.scalar_type();
    if (!is_dtype_supported(device, dtype)) return false;

    bool use_scalar = !other.defined();
    at::Tensor a = self.to(out.device(), dtype);
    at::Tensor b = use_scalar ? at::Tensor() : other.to(out.device(), dtype);

    at::TensorIteratorConfig config;
    config.set_check_mem_overlap(true).add_output(out).add_input(a);
    if (!use_scalar) config.add_input(b);
    at::TensorIterator iter = config.build();

    uint64_t numel = iter.numel();
    if (numel == 0) return true;
    int32_t out_dims = static_cast<int32_t>(iter.ndim());
    if (out_dims > MAX_DIMS || !fits_32bit_indexing(iter)) return false;

    uint32_t workgroupSizeX = get_dtype_workgroup_size(dtype, 1, device);
    uint32_t contiguous = iter.is_contiguous() ? 1 : 0;

    SpecializationBuilder spd{};
    spd.push(static_cast<uint32_t>(operation))
       .push(contiguous)
       .push(use_scalar ? 1u : 0u)
       .push(out_dims)
       .push(workgroupSizeX);

    MetadataBuilder metadataBuilder = get_iter_metadata(iter, {1, use_scalar ? -1 : 2, 0});
    Metadata metadata = metadataBuilder.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(a))
       .push(use_scalar ? uint64_t(0) : get_tensor_address(b))
       .push(get_tensor_address(iter.output()))
       .push_metadata(contiguous ? nullptr : &metadata)
       .push(numel)
       .push_scalar(use_scalar ? scalar : at::Scalar(0), dtype);

    std::vector<at::Tensor> reads = {a};
    if (!use_scalar) reads.push_back(b);

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::INTEGER_BINARY, dtype},
        spd.build(),
        pcs.build(),
        /* read = */ reads,
        /* write = */ {iter.output()},
        get_group_count(numel, workgroupSizeX), 1, 1
    );
    return true;
}

at::Tensor cpu_integer_binary(const at::Tensor& a, const at::Tensor& b, IntegerBinaryOp operation)
{
    switch (operation) {
        case IntegerBinaryOp::AND:    return at::bitwise_and(a, b);
        case IntegerBinaryOp::OR:     return at::bitwise_or(a, b);
        case IntegerBinaryOp::XOR:    return at::bitwise_xor(a, b);
        case IntegerBinaryOp::LSHIFT: return at::bitwise_left_shift(a, b);
        case IntegerBinaryOp::RSHIFT: return at::bitwise_right_shift(a, b);
        case IntegerBinaryOp::GCD:    return at::gcd(a, b);
        case IntegerBinaryOp::LCM:    return at::lcm(a, b);
    }
    return at::bitwise_and(a, b);
}

at::Tensor& integer_binary_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out,
    IntegerBinaryOp operation,
    bool allow_bool,
    const char* name)
{
    at::ScalarType dtype = at::result_type(self, other);
    TORCH_CHECK_NOT_IMPLEMENTED(at::isIntegralType(dtype, allow_bool), "\"", name, "\" not implemented for '", dtype, "'");
    TORCH_CHECK(at::canCast(dtype, out.scalar_type()), "result type ", dtype, " can't be cast to the desired output type ", out.scalar_type());

    bool scalar_other = is_cpu_scalar(other) && !is_cpu_scalar(self);
    std::vector<int64_t> shape = scalar_other ? self.sizes().vec() : at::infer_size(self.sizes(), other.sizes());
    at::Tensor result = out.scalar_type() == dtype ? out : at::empty({0}, out.options().dtype(dtype));
    if (result.sizes() != shape) result.resize_(shape);

    bool ran = scalar_other
        ? dispatch_integer_binary_shader(self, at::Tensor(), other.item(), result, operation)
        : dispatch_integer_binary_shader(self, other, at::Scalar(0), result, operation);
    if (!ran) {
        TORCHVULKAN_CPU_FALLBACK("", name, " on ", dtype, " can't run on the Vulkan device. Falling back to CPU.");
        result.copy_(cpu_integer_binary(self.cpu(), other.cpu(), operation));
    }
    if (!result.is_same(out)) fill_out(out, result);
    return out;
}

at::Tensor& bitwise_and_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out)
{
    return integer_binary_out(self, other, out, IntegerBinaryOp::AND, true, "bitwise_and");
}

at::Tensor& bitwise_or_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out)
{
    return integer_binary_out(self, other, out, IntegerBinaryOp::OR, true, "bitwise_or");
}

at::Tensor& bitwise_xor_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out)
{
    return integer_binary_out(self, other, out, IntegerBinaryOp::XOR, true, "bitwise_xor");
}

at::Tensor& bitwise_left_shift_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out)
{
    return integer_binary_out(self, other, out, IntegerBinaryOp::LSHIFT, false, "lshift");
}

at::Tensor& bitwise_right_shift_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out)
{
    return integer_binary_out(self, other, out, IntegerBinaryOp::RSHIFT, false, "rshift");
}

at::Tensor& gcd_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out)
{
    return integer_binary_out(self, other, out, IntegerBinaryOp::GCD, false, "gcd");
}

at::Tensor& lcm_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out)
{
    return integer_binary_out(self, other, out, IntegerBinaryOp::LCM, false, "lcm");
}

at::Tensor& bitwise_not_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    at::ScalarType dtype = self.scalar_type();
    TORCH_CHECK_NOT_IMPLEMENTED(at::isIntegralType(dtype, /*includeBool=*/true), "\"bitwise_not\" not implemented for '", dtype, "'");
    at::Tensor ones = dtype == at::kBool ? at::scalar_tensor(true, at::kBool) : at::scalar_tensor(-1, at::kLong).to(dtype);
    return integer_binary_out(self, ones, out, IntegerBinaryOp::XOR, true, "bitwise_not");
}

at::Tensor wrap_scalar(const at::Scalar& value)
{
    at::Tensor t = at::scalar_tensor(value, at::TensorOptions().dtype(value.isFloatingPoint() ? at::kDouble : value.isBoolean() ? at::kBool : at::kLong));
    t.unsafeGetTensorImpl()->set_wrapped_number(true);
    return t;
}

at::Tensor lshift_tensor_vulkan(const at::Tensor& self, const at::Tensor& other)
{
    at::Tensor out = at::empty({0}, self.options().dtype(at::result_type(self, other)));
    return bitwise_left_shift_out_vulkan(self, other, out);
}

at::Tensor lshift_scalar_vulkan(const at::Tensor& self, const at::Scalar& other)
{
    return lshift_tensor_vulkan(self, wrap_scalar(other));
}

at::Tensor rshift_tensor_vulkan(const at::Tensor& self, const at::Tensor& other)
{
    at::Tensor out = at::empty({0}, self.options().dtype(at::result_type(self, other)));
    return bitwise_right_shift_out_vulkan(self, other, out);
}

at::Tensor rshift_scalar_vulkan(const at::Tensor& self, const at::Scalar& other)
{
    return rshift_tensor_vulkan(self, wrap_scalar(other));
}

at::Tensor& ilshift_tensor_vulkan(at::Tensor& self, const at::Tensor& other)
{
    return copy_inplace_result(self, lshift_tensor_vulkan(self, other));
}

at::Tensor& ilshift_scalar_vulkan(at::Tensor& self, const at::Scalar& other)
{
    return copy_inplace_result(self, lshift_scalar_vulkan(self, other));
}

at::Tensor& irshift_tensor_vulkan(at::Tensor& self, const at::Tensor& other)
{
    return copy_inplace_result(self, rshift_tensor_vulkan(self, other));
}

at::Tensor& irshift_scalar_vulkan(at::Tensor& self, const at::Scalar& other)
{
    return copy_inplace_result(self, rshift_scalar_vulkan(self, other));
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("bitwise_and.Tensor_out", &bitwise_and_out_vulkan);
    m.impl("bitwise_or.Tensor_out", &bitwise_or_out_vulkan);
    m.impl("bitwise_xor.Tensor_out", &bitwise_xor_out_vulkan);
    m.impl("bitwise_not.out", &bitwise_not_out_vulkan);
    m.impl("bitwise_left_shift.Tensor_out", &bitwise_left_shift_out_vulkan);
    m.impl("bitwise_right_shift.Tensor_out", &bitwise_right_shift_out_vulkan);
    m.impl("gcd.out", &gcd_out_vulkan);
    m.impl("lcm.out", &lcm_out_vulkan);
    m.impl("__lshift__.Tensor", &lshift_tensor_vulkan);
    m.impl("__lshift__.Scalar", &lshift_scalar_vulkan);
    m.impl("__rshift__.Tensor", &rshift_tensor_vulkan);
    m.impl("__rshift__.Scalar", &rshift_scalar_vulkan);
    m.impl("__ilshift__.Tensor", &ilshift_tensor_vulkan);
    m.impl("__ilshift__.Scalar", &ilshift_scalar_vulkan);
    m.impl("__irshift__.Tensor", &irshift_tensor_vulkan);
    m.impl("__irshift__.Scalar", &irshift_scalar_vulkan);
}
