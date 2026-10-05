#include <torch/extension.h>
#include <ATen/WrapDimUtils.h>
#include <algorithm>
#include <vector>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "api/ops/helpers.h"

namespace {

enum class ScanOp { SUM = 0, PROD = 1 };

at::Tensor cumscan(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype,
    ScanOp op)
{
    c10::ScalarType out_dtype = reduce_compute_dtype(self, dtype);
    c10::ScalarType compute = reduce_accumulate_dtype(out_dtype);

    at::Tensor x = self.to(compute);
    if (x.dim() == 0 || x.numel() == 0) return x.clone().to(out_dtype);

    int64_t d = c10::maybe_wrap_dim(dim, x.dim());
    if (!is_dtype_supported(compute) || x.dim() > MAX_DIMS) {
        at::Tensor cpu = op == ScanOp::SUM ? at::cumsum(x.cpu(), d) : at::cumprod(x.cpu(), d);
        return cpu.to(out_dtype).to(self.device());
    }

    int64_t ndim = x.dim();
    int64_t scan_size = x.size(d);
    int64_t num_lines = scan_size > 0 ? x.numel() / scan_size : 0;
    at::Tensor out = at::empty(x.sizes(), x.options());
    if (num_lines == 0) return out.to(out_dtype);

    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    uint32_t workgroupSizeX = std::min<uint32_t>(256u, device->properties.limits.maxComputeWorkGroupInvocations);
    ShaderKey shader_key{torchvulkan::Kernel::SCAN, compute};
    uint32_t opv = static_cast<uint32_t>(op);
    int32_t ndim32 = static_cast<int32_t>(ndim);

    SpecializationBuilder spd{};
    spd.push(opv)
       .push(ndim32)
       .push(workgroupSizeX);
    SpecializationArgs specialization = spd.build();

    std::vector<int64_t> self_strides(x.strides().begin(), x.strides().end());
    std::vector<int64_t> out_strides(out.strides().begin(), out.strides().end());
    std::vector<int64_t> sizes_kept(x.sizes().begin(), x.sizes().end());
    sizes_kept[d] = 1;

    IntDivider sizes;
    uint32_t strides_in[MAX_DIMS] = {0};
    uint32_t strides_out[MAX_DIMS] = {0};
    for (int64_t i = 0; i < ndim; i++) {
        int64_t actual_dim = ndim - 1 - i;
        sizes.set(i, static_cast<uint32_t>(sizes_kept[actual_dim]));
        strides_in[i] = static_cast<uint32_t>(self_strides[actual_dim]);
        strides_out[i] = static_cast<uint32_t>(out_strides[actual_dim]);
    }

    MetadataBuilder metadataBuilder{};
    metadataBuilder.push(sizes, ndim32)
                   .push_array(strides_in, ndim32)
                   .push_array(strides_out, ndim32);
    Metadata metadata = metadataBuilder.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(x))
       .push(get_tensor_address(out))
       .push(device->shader_manager->registerMetadata(metadata))
       .push(static_cast<uint64_t>(num_lines))
       .push(static_cast<uint32_t>(scan_size))
       .push(static_cast<uint32_t>(self_strides[d]))
       .push(static_cast<uint32_t>(out_strides[d]));

    // one workgroup per line - the shader itself distributes each line across workgroupSizeX threads
    uint32_t groupX = static_cast<uint32_t>(num_lines);

    PushConstants pushConstants = { const_cast<void*>(pcs.data()), pcs.size() };
    device->shader_manager->dispatchShader(
        shader_key,
        specialization,
        pushConstants,
        /* read = */ {x},
        /* write = */ {out},
        groupX, 1, 1
    );

    return out.to(out_dtype);
}

at::Tensor& cumsum_out_vulkan(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype,
    at::Tensor& out)
{
    out.copy_(cumscan(self, dim, dtype, ScanOp::SUM));
    return out;
}

at::Tensor& cumprod_out_vulkan(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype,
    at::Tensor& out)
{
    out.copy_(cumscan(self, dim, dtype, ScanOp::PROD));
    return out;
}

enum class CumArgOp { MAX = 0, MIN = 1 };

void cumscanarg(
    const at::Tensor& self,
    at::Tensor& values,
    at::Tensor& indices,
    int64_t dim,
    CumArgOp op)
{
    if (self.dim() == 0 || self.numel() == 0) {
        values.copy_(self);
        if (indices.numel() > 0) indices.zero_();
        return;
    }
    int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    if (!is_dtype_supported(self.scalar_type()) || self.dim() > MAX_DIMS) {
        auto r = op == CumArgOp::MAX ? at::cummax(self.cpu(), dim) : at::cummin(self.cpu(), dim);
        values.copy_(std::get<0>(r));
        indices.copy_(std::get<1>(r));
        return;
    }

    int64_t ndim = self.dim();
    int64_t scan_size = self.size(d);
    int64_t num_lines = scan_size > 0 ? self.numel() / scan_size : 0;
    at::Tensor val_t = at::empty(self.sizes(), self.options());
    at::Tensor idx_t = at::empty(self.sizes(), self.options().dtype(at::kLong));
    if (num_lines == 0) return;

    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    uint32_t workgroupSizeX = std::min<uint32_t>(256u, device->properties.limits.maxComputeWorkGroupInvocations);
    ShaderKey shader_key{torchvulkan::Kernel::SCAN_ARG, self.scalar_type()};
    uint32_t opv = static_cast<uint32_t>(op);
    int32_t ndim32 = static_cast<int32_t>(ndim);

    SpecializationBuilder spd{};
    spd.push(opv)
       .push(ndim32)
       .push(workgroupSizeX);
    SpecializationArgs specialization = spd.build();

    std::vector<int64_t> self_strides(self.strides().begin(), self.strides().end());
    std::vector<int64_t> out_strides(val_t.strides().begin(), val_t.strides().end());
    std::vector<int64_t> sizes_kept(self.sizes().begin(), self.sizes().end());
    sizes_kept[d] = 1;

    IntDivider sizes;
    uint32_t strides_in[MAX_DIMS] = {0};
    uint32_t strides_out[MAX_DIMS] = {0};
    for (int64_t i = 0; i < ndim; i++) {
        int64_t actual_dim = ndim - 1 - i;
        sizes.set(i, static_cast<uint32_t>(sizes_kept[actual_dim]));
        strides_in[i] = static_cast<uint32_t>(self_strides[actual_dim]);
        strides_out[i] = static_cast<uint32_t>(out_strides[actual_dim]);
    }

    MetadataBuilder metadataBuilder{};
    metadataBuilder.push(sizes, ndim32)
                   .push_array(strides_in, ndim32)
                   .push_array(strides_out, ndim32);
    Metadata metadata = metadataBuilder.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(self))
       .push(get_tensor_address(val_t))
       .push(get_tensor_address(idx_t))
       .push(device->shader_manager->registerMetadata(metadata))
       .push(static_cast<uint64_t>(num_lines))
       .push(static_cast<uint32_t>(scan_size))
       .push(static_cast<uint32_t>(self_strides[d]))
       .push(static_cast<uint32_t>(out_strides[d]));

    // one workgroup per line - the shader itself distributes each line across workgroupSizeX threads
    uint32_t groupX = static_cast<uint32_t>(num_lines);

    PushConstants pushConstants = { const_cast<void*>(pcs.data()), pcs.size() };
    device->shader_manager->dispatchShader(
        shader_key,
        specialization,
        pushConstants,
        /* read = */ {self},
        /* write = */ {val_t, idx_t},
        groupX, 1, 1
    );

    values.copy_(val_t);
    indices.copy_(idx_t);
}

void cummax_helper_vulkan(
    const at::Tensor& self,
    at::Tensor& values,
    at::Tensor& indices,
    int64_t dim)
{
    cumscanarg(self, values, indices, dim, CumArgOp::MAX);
}

void cummin_helper_vulkan(
    const at::Tensor& self,
    at::Tensor& values,
    at::Tensor& indices,
    int64_t dim)
{
    cumscanarg(self, values, indices, dim, CumArgOp::MIN);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("cumsum.out", &cumsum_out_vulkan);
    m.impl("cumprod.out", &cumprod_out_vulkan);
    m.impl("_cummax_helper", &cummax_helper_vulkan);
    m.impl("_cummin_helper", &cummin_helper_vulkan);
}
