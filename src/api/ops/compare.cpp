#include <torch/extension.h>
#include <ATen/ExpandUtils.h>
#include <limits>
#include "api/ops/compare.h"

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

at::Tensor& dispatch_compare(
    const at::Tensor& self, 
    const at::Tensor* other, 
    const at::Scalar& scalar_value, 
    at::Tensor& out, 
    CompareOp op, 
    c10::ScalarType promoted
) {
    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    uint32_t alignment = device->properties.limits.minStorageBufferOffsetAlignment;
    bool use_scalar = (other == nullptr);

    std::vector<int64_t> out_shape = use_scalar
        ? self.sizes().vec()
        : at::infer_size(self.sizes(), other->sizes());
    if (!out.sizes().equals(out_shape)) out.resize_(out_shape);

    at::Tensor self_p = self.to(promoted);
    if ((self_p.storage_offset() * self_p.element_size()) % alignment != 0) self_p = self_p.clone();
    at::Tensor a = self_p.expand(out.sizes());

    at::Tensor other_p, b;
    if (!use_scalar) {
        other_p = other->to(promoted);
        if ((other_p.storage_offset() * other_p.element_size()) % alignment != 0) other_p = other_p.clone();
        b = other_p.expand(out.sizes());
    } else {
        other_p = self_p;
        b = a;
    }

    uint64_t numel = out.numel();
    if (numel == 0) return out;
    int32_t out_dims = static_cast<int32_t>(out.dim());

    uint32_t contiguous = (a.is_contiguous() && b.is_contiguous() && out.is_contiguous()) ? 1 : 0;
    uint32_t workgroupSizeX = get_dtype_workgroup_size(promoted, 1);
    torchvulkan::ShaderID shader_id = torchvulkan::get_shader_id_compareop(promoted);
    uint32_t opv = static_cast<uint32_t>(op);
    uint32_t usc = use_scalar ? 1u : 0u;

    SpecializationBuilder spd{};
    spd.push(opv)
       .push(contiguous)
       .push(usc)
       .push(out_dims)
       .push(workgroupSizeX);
    uint32_t key = (workgroupSizeX << 12) | (out_dims << 8) | (usc << 7) | (contiguous << 6) | opv;
    SpecializationArgs specialization = {spd.data(), spd.offsets(), spd.sizes(), spd.numConstants(), key};

    IntDivider sizes;
    uint32_t strides_a[MAX_DIMS] = {0};
    uint32_t strides_b[MAX_DIMS] = {0};
    uint32_t strides_out[MAX_DIMS] = {0};

    uint64_t metadata_address = 0;
    if (!contiguous) {
        for (int i = 0; i < out_dims; i++) {
            int td = out_dims - 1 - i;
            sizes.set(i, static_cast<uint32_t>(out.size(td)));
            strides_a[i] = static_cast<uint32_t>(a.stride(td));
            strides_b[i] = static_cast<uint32_t>(b.stride(td));
            strides_out[i] = static_cast<uint32_t>(out.stride(td));
        }

        MetadataBuilder metadataBuilder{};
        metadataBuilder.push(sizes, out_dims)
                       .push_array(strides_a, out_dims)
                       .push_array(strides_b, out_dims)
                       .push_array(strides_out, out_dims);
        Metadata metadata = metadataBuilder.build();
        metadata_address = device->shader_manager->registerMetadata(metadata);
    }

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(a))
       .push(use_scalar ? (uint64_t)0 : get_tensor_address(b)) // the shader never reads the second operand for a scalar
       .push(get_tensor_address(out))
       .push(metadata_address)
       .push(numel)
       .push_scalar(scalar_value, promoted);

    std::vector<at::Tensor> readTensors = {self_p};
    if (!use_scalar) readTensors.push_back(other_p);

    uint32_t groupX = (numel + (workgroupSizeX - 1)) / workgroupSizeX;
    PushConstants pushConstants = { const_cast<void*>(pcs.data()), pcs.size() };
    device->shader_manager->dispatchShader(
        shader_id,
        specialization,
        pushConstants,
        /* read = */ readTensors,
        /* write = */ {out},
        groupX, 1, 1
    );

    return out;
}

at::Tensor torchvulkan::where_vulkan(
    const at::Tensor& condition, 
    const at::Tensor& self, 
    const at::Tensor& other
) {
    c10::ScalarType promoted = at::result_type(self, other);
    std::vector<int64_t> shape = at::infer_size(self.sizes(), other.sizes());
    shape = at::infer_size(shape, condition.sizes());

    at::Tensor out = at::empty(shape, self.options().dtype(promoted));
    if (out.numel() == 0) return out;

    if (!is_dtype_supported(promoted) || out.dim() > MAX_DIMS) {
        return at::where(condition.cpu(), self.cpu(), other.cpu()).to(out.device());
    }

    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    uint32_t alignment = device->properties.limits.minStorageBufferOffsetAlignment;
    auto aligned = [&](at::Tensor t) {
        if ((t.storage_offset() * t.element_size()) % alignment != 0) return t.clone();
        return t;
    };
    at::Tensor c = aligned(condition.to(at::kBool));
    at::Tensor a = aligned(self.to(promoted));
    at::Tensor b = aligned(other.to(promoted));
    at::Tensor ce = c.expand(shape), ae = a.expand(shape), be = b.expand(shape);

    uint64_t numel = out.numel();
    int32_t out_dims = static_cast<int32_t>(out.dim());
    uint32_t contiguous = (ce.is_contiguous() && ae.is_contiguous() && be.is_contiguous()) ? 1 : 0;
    uint32_t workgroupSizeX = get_dtype_workgroup_size(promoted, 1);
    torchvulkan::ShaderID shader_id = torchvulkan::get_shader_id_where(promoted);

    SpecializationBuilder spd{};
    spd.push(contiguous)
       .push(out_dims)
       .push(workgroupSizeX);
    uint32_t key = (workgroupSizeX << 8) | (static_cast<uint32_t>(out_dims) << 4) | contiguous;
    SpecializationArgs specialization = {spd.data(), spd.offsets(), spd.sizes(), spd.numConstants(), key};

    IntDivider sizes;
    uint32_t strides_cond[MAX_DIMS] = {0};
    uint32_t strides_a[MAX_DIMS] = {0};
    uint32_t strides_b[MAX_DIMS] = {0};

    uint64_t metadata_address = 0;
    if (!contiguous) {
        for (int i = 0; i < out_dims; i++) {
            int td = out_dims - 1 - i;
            sizes.set(i, static_cast<uint32_t>(out.size(td)));
            strides_cond[i] = static_cast<uint32_t>(ce.stride(td));
            strides_a[i] = static_cast<uint32_t>(ae.stride(td));
            strides_b[i] = static_cast<uint32_t>(be.stride(td));
        }

        MetadataBuilder metadataBuilder{};
        metadataBuilder.push(sizes, out_dims)
                       .push_array(strides_cond, out_dims)
                       .push_array(strides_a, out_dims)
                       .push_array(strides_b, out_dims);
        Metadata metadata = metadataBuilder.build();
        metadata_address = device->shader_manager->registerMetadata(metadata);
    }

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(c))
       .push(get_tensor_address(a))
       .push(get_tensor_address(b))
       .push(get_tensor_address(out))
       .push(metadata_address)
       .push(numel);

    uint32_t groupX = (numel + (workgroupSizeX - 1)) / workgroupSizeX;
    PushConstants pushConstants = { const_cast<void*>(pcs.data()), pcs.size() };
    device->shader_manager->dispatchShader(
        shader_id,
        specialization,
        pushConstants,
        /* read = */ {c, a, b},
        /* write = */ {out},
        groupX, 1, 1
    );

    return out;
}

at::Tensor& torchvulkan::compare_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out, CompareOp op) 
{
    std::vector<int64_t> shape = at::infer_size(self.sizes(), other.sizes());
    if (!out.sizes().equals(shape)) out.resize_(shape);
    c10::ScalarType promoted = at::result_type(self, other);
    if (!is_dtype_supported(promoted) || out.dim() > MAX_DIMS) {
        out.copy_(cpu_compare(self.cpu(), other.cpu(), op));
        return out;
    }
    return dispatch_compare(self, &other, at::Scalar(0), out, op, promoted);
}

at::Tensor& torchvulkan::compare_scalar_out_vulkan(const at::Tensor& self, const at::Scalar& other, at::Tensor& out, CompareOp op) 
{
    if (!out.sizes().equals(self.sizes())) out.resize_(self.sizes());
    c10::ScalarType promoted = at::result_type(self, other);
    if (!is_dtype_supported(promoted) || out.dim() > MAX_DIMS) {
        out.copy_(cpu_compare_scalar(self.cpu(), other, op));
        return out;
    }
    return dispatch_compare(self, nullptr, other, out, op, promoted);
}

at::Tensor torchvulkan::where_scalar_other_vulkan(const at::Tensor& condition, const at::Tensor& self, const at::Scalar& other) 
{
    c10::ScalarType promoted = at::result_type(self, other);
    return where_vulkan(condition, self, at::scalar_tensor(other, self.options().dtype(promoted)));
}

at::Tensor torchvulkan::where_scalar_self_vulkan(const at::Tensor& condition, const at::Scalar& self, const at::Tensor& other) 
{
    c10::ScalarType promoted = at::result_type(other, self);
    return where_vulkan(condition, at::scalar_tensor(self, other.options().dtype(promoted)), other);
}

at::Tensor torchvulkan::where_scalar_vulkan(const at::Tensor& condition, const at::Scalar& self, const at::Scalar& other) 
{
    c10::ScalarType promoted = at::result_type(self, other);
    at::TensorOptions opts = condition.options().dtype(promoted);
    return where_vulkan(condition, at::scalar_tensor(self, opts), at::scalar_tensor(other, opts));
}

at::Tensor torchvulkan::isnan_vulkan(const at::Tensor& self) 
{
    return at::ne(self, self);
}

at::Tensor& torchvulkan::isinf_out_vulkan(const at::Tensor& self, at::Tensor& out) 
{
    if (!self.is_floating_point()) { out.fill_(false); return out; }
    return compare_scalar_out_vulkan(at::abs(self), std::numeric_limits<double>::infinity(), out, CompareOp::EQ);
}

at::Tensor& torchvulkan::isposinf_out_vulkan(const at::Tensor& self, at::Tensor& out) 
{
    if (!self.is_floating_point()) { out.fill_(false); return out; }
    return compare_scalar_out_vulkan(self, std::numeric_limits<double>::infinity(), out, CompareOp::EQ);
}

at::Tensor& torchvulkan::isneginf_out_vulkan(const at::Tensor& self, at::Tensor& out) 
{
    if (!self.is_floating_point()) { out.fill_(false); return out; }
    return compare_scalar_out_vulkan(self, -std::numeric_limits<double>::infinity(), out, CompareOp::EQ);
}

at::Tensor& torchvulkan::eq_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) { return compare_out_vulkan(self, other, out, CompareOp::EQ); }
at::Tensor& torchvulkan::eq_scalar_out_vulkan(const at::Tensor& self, const at::Scalar& other, at::Tensor& out) { return compare_scalar_out_vulkan(self, other, out, CompareOp::EQ); }
at::Tensor& torchvulkan::ne_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) { return compare_out_vulkan(self, other, out, CompareOp::NE); }
at::Tensor& torchvulkan::ne_scalar_out_vulkan(const at::Tensor& self, const at::Scalar& other, at::Tensor& out) { return compare_scalar_out_vulkan(self, other, out, CompareOp::NE); }
at::Tensor& torchvulkan::lt_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) { return compare_out_vulkan(self, other, out, CompareOp::LT); }
at::Tensor& torchvulkan::lt_scalar_out_vulkan(const at::Tensor& self, const at::Scalar& other, at::Tensor& out) { return compare_scalar_out_vulkan(self, other, out, CompareOp::LT); }
at::Tensor& torchvulkan::le_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) { return compare_out_vulkan(self, other, out, CompareOp::LE); }
at::Tensor& torchvulkan::le_scalar_out_vulkan(const at::Tensor& self, const at::Scalar& other, at::Tensor& out) { return compare_scalar_out_vulkan(self, other, out, CompareOp::LE); }
at::Tensor& torchvulkan::gt_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) { return compare_out_vulkan(self, other, out, CompareOp::GT); }
at::Tensor& torchvulkan::gt_scalar_out_vulkan(const at::Tensor& self, const at::Scalar& other, at::Tensor& out) { return compare_scalar_out_vulkan(self, other, out, CompareOp::GT); }
at::Tensor& torchvulkan::ge_tensor_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) { return compare_out_vulkan(self, other, out, CompareOp::GE); }
at::Tensor& torchvulkan::ge_scalar_out_vulkan(const at::Tensor& self, const at::Scalar& other, at::Tensor& out) { return compare_scalar_out_vulkan(self, other, out, CompareOp::GE); }

at::Tensor& torchvulkan::logical_and_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) { return compare_out_vulkan(self, other, out, CompareOp::LAND); }
at::Tensor& torchvulkan::logical_or_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) { return compare_out_vulkan(self, other, out, CompareOp::LOR); }
at::Tensor& torchvulkan::logical_xor_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) { return compare_out_vulkan(self, other, out, CompareOp::LXOR); }
at::Tensor& torchvulkan::logical_not_out_vulkan(const at::Tensor& self, at::Tensor& out) { return compare_scalar_out_vulkan(self, at::Scalar(0), out, CompareOp::EQ); }