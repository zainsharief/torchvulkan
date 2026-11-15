#include <torch/extension.h>
#include <algorithm>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "api/ops/helpers.h"
#include "api/fallback.h"

namespace {

at::Tensor& fill_scalar_vulkan(
    at::Tensor& self, 
    const at::Scalar& value)
{
    if (self.is_complex() && self.numel() > 0) 
    {
        c10::complex<double> v = value.toComplexDouble();
        at::Tensor pairs = at::view_as_real(self.is_conj() ? self.conj() : self);  // conj() of a conjugate view is the plain one
        pairs.select(-1, 0).fill_(v.real());
        pairs.select(-1, 1).fill_(self.is_conj() ? -v.imag() : v.imag());
        return self;
    }

    if (c10::isFloat8Type(self.scalar_type()) && self.numel() > 0) 
    {
        float v = value.toFloat();
        uint8_t bits = 0;
        switch (self.scalar_type()) {
            case at::kFloat8_e4m3fn: bits = c10::Float8_e4m3fn(v).x; break;
            case at::kFloat8_e5m2: bits = c10::Float8_e5m2(v).x; break;
            case at::kFloat8_e4m3fnuz: bits = c10::Float8_e4m3fnuz(v).x; break;
            case at::kFloat8_e5m2fnuz: bits = c10::Float8_e5m2fnuz(v).x; break;
            case at::kFloat8_e8m0fnu: bits = c10::Float8_e8m0fnu(v).x; break;
            default: TORCH_CHECK(false, "torchvulkan [ERROR]: no fill for ", self.scalar_type());
        }
        self.view(at::kByte).fill_(bits);
        return self;
    }

    for (int64_t d = 0; d < self.dim(); d++) 
    {
        if (self.stride(d) == 0 && self.size(d) > 1) 
        {
            at::Tensor distinct = self.narrow(d, 0, 1);
            fill_scalar_vulkan(distinct, value);
            return self;
        }
    }

    constexpr int64_t chunk = int64_t(1) << 30;
    if (self.is_contiguous() && self.numel() > chunk) 
    {
        at::Tensor flat = self.view({-1});
        for (int64_t start = 0; start < flat.numel(); start += chunk)
        {
            at::Tensor part = flat.narrow(0, start, std::min(chunk, flat.numel() - start));
            fill_scalar_vulkan(part, value);
        }
        return self;
    }

    if (!is_dtype_supported(self, self.scalar_type()) && value.equal(0) && !self.is_complex())
    {
        at::ScalarType bits = c10::elementSize(self.scalar_type()) == 1 ? at::kByte
                            : c10::elementSize(self.scalar_type()) == 2 ? at::kShort
                            : c10::elementSize(self.scalar_type()) == 4 ? at::kInt : at::kLong;
        at::Tensor raw = self.view(bits);
        fill_scalar_vulkan(raw, 0);
        return self;
    }

    at::TensorIterator iter = at::TensorIterator::borrowing_nullary_op(self);

    uint64_t numel = iter.numel();
    if (numel == 0) return self;
    
    int32_t out_dims = static_cast<int32_t>(iter.ndim());
    if (!is_dtype_supported(self, iter.dtype()) || out_dims > MAX_DIMS || !fits_32bit_indexing(iter)) {
        TORCHVULKAN_FALLBACK("Can't fill a ", iter.dtype(), " tensor of this shape on the Vulkan device. Falling back to CPU.");
        self.copy_(at::empty(self.sizes(), self.options().device(at::kCPU)).fill_(value));
        return self;
    }

    DeviceContext* device = get_device_context(self);
    ShaderKey shader_key{torchvulkan::Kernel::FILL, self.scalar_type()};
    uint32_t vecSize = get_dtype_vec_size(self.scalar_type()); // our workgroup must match the shader workgroup
    uint32_t workgroupSizeX = get_dtype_workgroup_size(self.scalar_type(), vecSize, device);
    uint32_t contiguous = iter.is_contiguous() ? 1 : 0;

    SpecializationBuilder spd{};
    spd.push(out_dims)
        .push(contiguous)
        .push(workgroupSizeX);

    MetadataBuilder metadataBuilder = get_iter_metadata(iter, {0});
    Metadata metadata = metadataBuilder.build();

    PushConstantBuilder pcs{};
    pcs.push(get_output_address(self))
        .push_metadata(contiguous ? nullptr : &metadata)
        .push(numel)
        .push_scalar(value, self.scalar_type());

    device->shader_manager->dispatchShader(
        shader_key,
        device->default_subgroup_size,
        spd.build(),
        pcs.build(),
        /* read = */ {},
        /* write = */ {self},
        get_group_count(contiguous ? (numel + vecSize - 1) / vecSize : numel, workgroupSizeX), 1, 1
    );

    return self;
}

at::Tensor& fill_tensor_vulkan(at::Tensor& self, const at::Tensor& value) 
{
    TORCH_CHECK(value.dim() == 0, "fill_ only supports 0-dimension value tensor but got tensor with ", value.dim(), " dimensions.");
    return fill_scalar_vulkan(self, value.item());
}

at::Tensor& zero_vulkan(at::Tensor& self)
{
    return fill_scalar_vulkan(self, 0);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("fill_.Scalar", &fill_scalar_vulkan);
    m.impl("fill_.Tensor", &fill_tensor_vulkan);
    m.impl("zero_", &zero_vulkan);
}
