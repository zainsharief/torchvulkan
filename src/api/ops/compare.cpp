#include <torch/extension.h>
#include <ATen/ExpandUtils.h>
#include <limits>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "api/ops/helpers.h"
#include "api/fallback.h"

namespace {

enum class CompareOp {
    EQ = 0,
    NE = 1,
    LT = 2,
    LE = 3,
    GT = 4,
    GE = 5,
    LAND = 6,
    LOR = 7,
    LXOR = 8
};

at::Tensor cpu_compare(const at::Tensor& a, const at::Tensor& b, CompareOp op) 
{
    switch (op) {
        case CompareOp::EQ:   return at::eq(a, b);
        case CompareOp::NE:   return at::ne(a, b);
        case CompareOp::LT:   return at::lt(a, b);
        case CompareOp::LE:   return at::le(a, b);
        case CompareOp::GT:   return at::gt(a, b);
        case CompareOp::GE:   return at::ge(a, b);
        case CompareOp::LAND: return at::logical_and(a, b);
        case CompareOp::LOR:  return at::logical_or(a, b);
        case CompareOp::LXOR: return at::logical_xor(a, b);
    }
    
    return at::eq(a, b);
}

at::Tensor cpu_compare_scalar(const at::Tensor& a, const at::Scalar& b, CompareOp op) 
{
    return cpu_compare(a, at::scalar_tensor(b, a.options()), op);
}

bool dispatch_compare_shader(
    const at::Tensor& self, 
    const at::Tensor* other, 
    const at::Scalar& scalar_value, 
    at::Tensor& out, 
    CompareOp op, 
    c10::ScalarType promoted)
{
    DeviceContext* device = get_device_context(self);
    if (!is_dtype_supported(device, promoted)) return false;
    bool use_scalar = (other == nullptr);

    at::Tensor a = self.to(promoted);
    at::Tensor b = use_scalar ? at::Tensor() : other->to(a.options());

    at::TensorIteratorConfig config;
    config.check_all_same_dtype(false).set_check_mem_overlap(true).add_output(out).add_input(a);
    if (!use_scalar) config.add_input(b);
    at::TensorIterator iter = config.build();

    uint64_t numel = iter.numel();
    if (numel == 0) return true;
    int32_t out_dims = static_cast<int32_t>(iter.ndim());
    if (out_dims > MAX_DIMS || !fits_32bit_indexing(iter)) return false;

    uint32_t contiguous = iter.is_contiguous() ? 1 : 0;
    uint32_t vecSize = get_dtype_vec_size(promoted);
    uint32_t workgroupSizeX = get_dtype_workgroup_size(promoted, vecSize, device);

    SpecializationBuilder spd{};
    spd.push(static_cast<uint32_t>(op))
       .push(contiguous)
       .push(use_scalar ? 1u : 0u)
       .push(out_dims)
       .push(workgroupSizeX);

    MetadataBuilder metadataBuilder = get_iter_metadata(iter, {1, use_scalar ? -1 : 2, 0});
    Metadata metadata = metadataBuilder.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(a))
       .push(use_scalar ? (uint64_t)0 : get_tensor_address(b)) // the shader never reads the second operand for a scalar
       .push(get_tensor_address(out))
       .push_metadata(contiguous ? nullptr : &metadata)
       .push(numel)
       .push_scalar(scalar_value, promoted);

    std::vector<at::Tensor> reads = {a};
    if (!use_scalar) reads.push_back(b);

    uint64_t threads = contiguous ? (numel + (vecSize - 1)) / vecSize : numel;
    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::COMPARE, promoted},
        spd.build(),
        pcs.build(),
        /* read = */ reads,
        /* write = */ {out},
        get_group_count(threads, workgroupSizeX), 1, 1
    );
    return true;
}

at::Tensor where_vulkan(
    const at::Tensor& condition, 
    const at::Tensor& self, 
    const at::Tensor& other
) {
    c10::ScalarType promoted = at::result_type(self, other);
    c10::Device device = condition.device().type() == c10::DeviceType::PrivateUse1 ? condition.device()
        : self.device().type() == c10::DeviceType::PrivateUse1 ? self.device() : other.device();

    // 0-dim CPU tensors (from scalars) join the vulkan tensors' device
    at::Tensor c = condition.to(device, at::kBool);
    at::Tensor a = self.to(device, promoted);
    at::Tensor b = other.to(device, promoted);

    at::Tensor out = at::empty(at::infer_size(at::infer_size(c.sizes(), a.sizes()), b.sizes()), a.options());
    at::TensorIterator iter = at::TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(out)
        .add_input(c)
        .add_input(a)
        .add_input(b)
        .build();
    uint64_t numel = iter.numel();
    if (numel == 0) return out;

    DeviceContext* context = get_device_context(out);
    int32_t out_dims = static_cast<int32_t>(iter.ndim());
    if (!is_dtype_supported(context, promoted) || out_dims > MAX_DIMS || !fits_32bit_indexing(iter)) {
        TORCHVULKAN_CPU_FALLBACK("This where can't run on the Vulkan device. Falling back to CPU.");
        return at::where(c.cpu(), a.cpu(), b.cpu()).to(device);
    }

    uint32_t contiguous = iter.is_contiguous() ? 1 : 0;
    uint32_t workgroupSizeX = get_dtype_workgroup_size(promoted, 1, context);

    SpecializationBuilder spd{};
    spd.push(contiguous)
       .push(out_dims)
       .push(workgroupSizeX);

    MetadataBuilder metadataBuilder = get_iter_metadata(iter, {1, 2, 3});
    Metadata metadata = metadataBuilder.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(c))
       .push(get_tensor_address(a))
       .push(get_tensor_address(b))
       .push(get_tensor_address(out))
       .push_metadata(contiguous ? nullptr : &metadata)
       .push(numel);

    context->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::WHERE, promoted},
        spd.build(),
        pcs.build(),
        /* read = */ {c, a, b},
        /* write = */ {out},
        get_group_count(numel, workgroupSizeX), 1, 1
    );

    return out;
}

at::Tensor& compare_scalar_out_vulkan(const at::Tensor& self, const at::Scalar& other, at::Tensor& out, CompareOp op) 
{
    if (!out.sizes().equals(self.sizes())) out.resize_(self.sizes());
    c10::ScalarType promoted = at::result_type(self, other);
    if (!dispatch_compare_shader(self, nullptr, other, out, op, promoted)) {
        TORCHVULKAN_CPU_FALLBACK("This comparison can't run on the Vulkan device. Falling back to CPU.");
        out.copy_(cpu_compare_scalar(self.cpu(), other, op));
    }
    return out;
}

bool is_cpu_scalar(const at::Tensor& t)
{
    return t.unsafeGetTensorImpl()->is_wrapped_number() || (t.dim() == 0 && t.device().is_cpu());
}

at::Tensor& compare_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out, CompareOp op) 
{
    if (is_cpu_scalar(other) && self.device().type() == c10::DeviceType::PrivateUse1) {
        return compare_scalar_out_vulkan(self, other.item(), out, op);
    }

    std::vector<int64_t> shape = at::infer_size(self.sizes(), other.sizes());
    if (!out.sizes().equals(shape)) out.resize_(shape);
    c10::ScalarType promoted = at::result_type(self, other);
    at::Tensor lhs = self.to(out.device());
    at::Tensor rhs = other.to(out.device());
    if (!dispatch_compare_shader(lhs, &rhs, at::Scalar(0), out, op, promoted)) {
        TORCHVULKAN_CPU_FALLBACK("This comparison can't run on the Vulkan device. Falling back to CPU.");
        out.copy_(cpu_compare(self.cpu(), other.cpu(), op));
    }
    return out;
}

at::Tensor& isinf_out_vulkan(const at::Tensor& self, at::Tensor& out) 
{
    if (!self.is_floating_point()) { out.fill_(false); return out; }
    return compare_scalar_out_vulkan(at::abs(self), std::numeric_limits<double>::infinity(), out, CompareOp::EQ);
}

at::Tensor& isposinf_out_vulkan(const at::Tensor& self, at::Tensor& out) 
{
    if (!self.is_floating_point()) { out.fill_(false); return out; }
    return compare_scalar_out_vulkan(self, std::numeric_limits<double>::infinity(), out, CompareOp::EQ);
}

at::Tensor& isneginf_out_vulkan(const at::Tensor& self, at::Tensor& out) 
{
    if (!self.is_floating_point()) { out.fill_(false); return out; }
    return compare_scalar_out_vulkan(self, -std::numeric_limits<double>::infinity(), out, CompareOp::EQ);
}

at::Tensor& eq_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) { return compare_out_vulkan(self, other, out, CompareOp::EQ); }

at::Tensor& eq_scalar_out_vulkan(const at::Tensor& self, const at::Scalar& other, at::Tensor& out) { return compare_scalar_out_vulkan(self, other, out, CompareOp::EQ); }

at::Tensor& ne_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) { return compare_out_vulkan(self, other, out, CompareOp::NE); }

at::Tensor& ne_scalar_out_vulkan(const at::Tensor& self, const at::Scalar& other, at::Tensor& out) { return compare_scalar_out_vulkan(self, other, out, CompareOp::NE); }

at::Tensor& lt_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) { return compare_out_vulkan(self, other, out, CompareOp::LT); }

at::Tensor& lt_scalar_out_vulkan(const at::Tensor& self, const at::Scalar& other, at::Tensor& out) { return compare_scalar_out_vulkan(self, other, out, CompareOp::LT); }

at::Tensor& le_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) { return compare_out_vulkan(self, other, out, CompareOp::LE); }

at::Tensor& le_scalar_out_vulkan(const at::Tensor& self, const at::Scalar& other, at::Tensor& out) { return compare_scalar_out_vulkan(self, other, out, CompareOp::LE); }

at::Tensor& gt_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) { return compare_out_vulkan(self, other, out, CompareOp::GT); }

at::Tensor& gt_scalar_out_vulkan(const at::Tensor& self, const at::Scalar& other, at::Tensor& out) { return compare_scalar_out_vulkan(self, other, out, CompareOp::GT); }

at::Tensor& ge_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) { return compare_out_vulkan(self, other, out, CompareOp::GE); }

at::Tensor& ge_scalar_out_vulkan(const at::Tensor& self, const at::Scalar& other, at::Tensor& out) { return compare_scalar_out_vulkan(self, other, out, CompareOp::GE); }

at::Tensor& logical_and_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) { return compare_out_vulkan(self, other, out, CompareOp::LAND); }

at::Tensor& logical_or_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) { return compare_out_vulkan(self, other, out, CompareOp::LOR); }

at::Tensor& logical_xor_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) { return compare_out_vulkan(self, other, out, CompareOp::LXOR); }

at::Tensor& logical_not_out_vulkan(const at::Tensor& self, at::Tensor& out) { return compare_scalar_out_vulkan(self, at::Scalar(0), out, CompareOp::EQ); }

at::Tensor& where_out_vulkan(const at::Tensor& condition, const at::Tensor& self, const at::Tensor& other, at::Tensor& out)
{
    return fill_out(out, where_vulkan(condition, self, other));
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("eq.Tensor_out", &eq_tensor_out_vulkan);
    m.impl("eq.Scalar_out", &eq_scalar_out_vulkan);
    m.impl("ne.Tensor_out", &ne_tensor_out_vulkan);
    m.impl("ne.Scalar_out", &ne_scalar_out_vulkan);
    m.impl("lt.Tensor_out", &lt_tensor_out_vulkan);
    m.impl("lt.Scalar_out", &lt_scalar_out_vulkan);
    m.impl("le.Tensor_out", &le_tensor_out_vulkan);
    m.impl("le.Scalar_out", &le_scalar_out_vulkan);
    m.impl("gt.Tensor_out", &gt_tensor_out_vulkan);
    m.impl("gt.Scalar_out", &gt_scalar_out_vulkan);
    m.impl("ge.Tensor_out", &ge_tensor_out_vulkan);
    m.impl("ge.Scalar_out", &ge_scalar_out_vulkan);
    m.impl("logical_and.out", &logical_and_out_vulkan);
    m.impl("logical_or.out", &logical_or_out_vulkan);
    m.impl("logical_xor.out", &logical_xor_out_vulkan);
    m.impl("logical_not.out", &logical_not_out_vulkan);
    m.impl("isinf.out", &isinf_out_vulkan);
    m.impl("isposinf.out", &isposinf_out_vulkan);
    m.impl("isneginf.out", &isneginf_out_vulkan);
    m.impl("where.self", &where_vulkan);
    m.impl("where.self_out", &where_out_vulkan);
}
