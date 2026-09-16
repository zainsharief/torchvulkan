#include <torch/extension.h>
#include <ATen/WrapDimUtils.h>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "shaders/shader_registry.h"
#include "api/ops/helpers.h"

namespace {

enum class ScanOp { SUM = 0, PROD = 1 };

at::Tensor cumscan(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype,
    ScanOp op)
{
    c10::ScalarType compute = reduce_compute_dtype(self, dtype);
    at::Tensor x = self.to(compute);
    if (x.dim() == 0 || x.numel() == 0) return x.clone();

    int64_t d = c10::maybe_wrap_dim(dim, x.dim());
    if (!is_dtype_supported(compute) || x.dim() > MAX_DIMS) {
        at::Tensor cpu = op == ScanOp::SUM ? at::cumsum(x.cpu(), dim) : at::cumprod(x.cpu(), dim);
        return cpu.to(self.device());
    }

    at::Tensor xt = x.transpose(d, -1).contiguous();
    int64_t scan_size = xt.size(-1);
    int64_t num_lines = scan_size > 0 ? xt.numel() / scan_size : 0;
    at::Tensor out_t = at::empty_like(xt);
    if (num_lines == 0) return out_t.transpose(d, -1).contiguous();

    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    uint32_t workgroupSizeX = get_dtype_workgroup_size(compute, 1);
    torchvulkan::ShaderID shader_id = torchvulkan::get_shader_id_scan(compute);
    uint32_t opv = static_cast<uint32_t>(op);

    SpecializationBuilder spd{};
    spd.push(opv)
       .push(workgroupSizeX);
    SpecializationArgs specialization = spd.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(xt))
       .push(get_tensor_address(out_t))
       .push(static_cast<uint64_t>(num_lines))
       .push(static_cast<uint32_t>(scan_size));

    uint32_t groupX = (static_cast<uint32_t>(num_lines) + (workgroupSizeX - 1)) / workgroupSizeX;

    PushConstants pushConstants = { const_cast<void*>(pcs.data()), pcs.size() };
    device->shader_manager->dispatchShader(
        shader_id,
        specialization,
        pushConstants,
        /* read = */ {xt},
        /* write = */ {out_t},
        groupX, 1, 1
    );

    return out_t.transpose(d, -1).contiguous();
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

    at::Tensor xt = self.transpose(d, -1).contiguous();
    int64_t scan_size = xt.size(-1);
    int64_t num_lines = scan_size > 0 ? xt.numel() / scan_size : 0;
    at::Tensor val_t = at::empty_like(xt);
    at::Tensor idx_t = at::empty(xt.sizes(), self.options().dtype(at::kLong));
    if (num_lines == 0) return;

    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    uint32_t workgroupSizeX = get_dtype_workgroup_size(self.scalar_type(), 1);
    torchvulkan::ShaderID shader_id = torchvulkan::get_shader_id_scan_arg(self.scalar_type());

    uint32_t opv = static_cast<uint32_t>(op);
    SpecializationBuilder spd{};
    spd.push(opv)
       .push(workgroupSizeX);
    SpecializationArgs specialization = spd.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(xt))
       .push(get_tensor_address(val_t))
       .push(get_tensor_address(idx_t))
       .push(static_cast<uint64_t>(num_lines))
       .push(static_cast<uint32_t>(scan_size));

    uint32_t groupX = (static_cast<uint32_t>(num_lines) + (workgroupSizeX - 1)) / workgroupSizeX;

    PushConstants pushConstants = { const_cast<void*>(pcs.data()), pcs.size() };
    device->shader_manager->dispatchShader(
        shader_id,
        specialization,
        pushConstants,
        /* read = */ {xt},
        /* write = */ {val_t, idx_t},
        groupX, 1, 1
    );

    values.copy_(val_t.transpose(d, -1));
    indices.copy_(idx_t.transpose(d, -1));
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
