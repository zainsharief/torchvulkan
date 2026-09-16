#include <torch/extension.h>
#include <ATen/WrapDimUtils.h>
#include <c10/core/MemoryFormat.h>
#include <limits>
#include <tuple>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "shaders/shader_registry.h"
#include "api/ops/helpers.h"
#include "api/ops/internal.h"

namespace {

enum class ReduceOp {
    SUM = 0,
    AMAX = 1,
    AMIN = 2,
    PROD = 3
};

at::Tensor dispatch_reduce_shader(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    ReduceOp operation,
    double empty_value,
    bool has_subgroup)
{
    int64_t ndim = self.dim();
    if (ndim == 0) return self.clone();

    dim = c10::maybe_wrap_dim(dim, ndim);
    int64_t reduce_ndim = std::max<int64_t>(ndim, 1);

    TORCH_CHECK(reduce_ndim <= MAX_DIMS, "torchvulkan [ERROR]: Coalesced dimensions (", reduce_ndim, ") exceed maximum supported (", MAX_DIMS, ").");

    std::vector<int64_t> self_sizes(self.sizes().begin(), self.sizes().end());
    std::vector<int64_t> self_strides(self.strides().begin(), self.strides().end());

    std::vector<int64_t> out_sizes = self_sizes;
    int64_t reduce_size = out_sizes[dim];
    out_sizes[dim] = 1;

    at::Tensor out = at::empty(out_sizes, self.options());
    uint64_t numel_out = out.numel();
    if (numel_out == 0) return keepdim ? out : out.squeeze(dim);
    if (reduce_size == 0) {
        out.fill_(empty_value);
        return keepdim ? out : out.squeeze(dim);
    }

    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    uint32_t op = static_cast<uint32_t>(operation);
    int32_t ndim32 = static_cast<int32_t>(reduce_ndim);

    at::ScalarType stype = self.scalar_type();
    bool dtype_needs_extended = c10::elementSize(stype) != 4;
    bool dtype_supported_subgroup =
        at::isFloatingType(stype) && c10::elementSize(stype) <= 4 &&
        (!dtype_needs_extended || device->support_subgroup_extended_types);

    bool use_subgroup =
        has_subgroup &&
        device->support_subgroup_arithmetic &&
        device->subgroup_size > 0 &&
        dtype_supported_subgroup &&
        reduce_size >= static_cast<int64_t>(device->subgroup_size) &&
        numel_out <= static_cast<uint64_t>(device->properties.limits.maxComputeWorkGroupCount[0]);

    uint32_t workgroupSizeX = use_subgroup
        ? device->subgroup_size
        : get_dtype_workgroup_size(self.scalar_type(), 1);
    torchvulkan::ShaderID shader_id = use_subgroup
        ? torchvulkan::get_shader_id_reduce_subgroup(self.scalar_type())
        : torchvulkan::get_shader_id_reduce(self.scalar_type());

    SpecializationBuilder spd{};
    spd.push(op)
       .push(ndim32)
       .push(workgroupSizeX);
    SpecializationArgs specialization = spd.build();

    IntDivider sizes;
    uint32_t strides_in[MAX_DIMS] = {0};
    for (int64_t i = 0; i < reduce_ndim; i++) {
        int64_t actual_dim = reduce_ndim - 1 - i;
        sizes.set(i, static_cast<uint32_t>(out_sizes[actual_dim]));
        strides_in[i] = static_cast<uint32_t>(self_strides[actual_dim]);
    }

    MetadataBuilder metadataBuilder{};
    metadataBuilder.push(sizes, ndim32)
                   .push_array(strides_in, ndim32);
    Metadata metadata = metadataBuilder.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(self))
       .push(get_tensor_address(out))
       .push(device->shader_manager->registerMetadata(metadata))
       .push(numel_out)
       .push(static_cast<uint32_t>(self_strides[dim]))
       .push(static_cast<uint32_t>(reduce_size));

    uint32_t groupX = use_subgroup
        ? static_cast<uint32_t>(numel_out)
        : static_cast<uint32_t>((numel_out + (workgroupSizeX - 1)) / workgroupSizeX);

    PushConstants pushConstants = { const_cast<void*>(pcs.data()), pcs.size() };
    device->shader_manager->dispatchShader(
        shader_id,
        specialization,
        pushConstants,
        /* read = */ {self},
        /* write = */ {out},
        groupX, 1, 1
    );

    return keepdim ? out : out.squeeze(dim);
}

at::Tensor cpu_reduce_fallback(
    const at::Tensor& cpu_self,
    at::OptionalIntArrayRef dims,
    bool keepdim,
    ReduceOp operation)
{
    at::IntArrayRef dim_ref = dims.has_value() ? *dims : at::IntArrayRef{};
    switch (operation) {
        case ReduceOp::SUM:  return at::sum(cpu_self, dims, keepdim);
        case ReduceOp::AMAX: return at::amax(cpu_self, dim_ref, keepdim);
        case ReduceOp::AMIN: return at::amin(cpu_self, dim_ref, keepdim);
        case ReduceOp::PROD: {
            if (!dims.has_value() || dims->size() == 0) return at::prod(cpu_self);
            std::vector<int64_t> dl;
            for (int64_t d : *dims) dl.push_back(c10::maybe_wrap_dim(d, cpu_self.dim()));
            std::sort(dl.begin(), dl.end(), std::greater<int64_t>());
            at::Tensor r = cpu_self;
            for (int64_t d : dl) r = at::prod(r, d, keepdim);
            return r;
        }
    }
    return at::sum(cpu_self, dims, keepdim);
}

at::Tensor reduce_dims_vulkan(
    const at::Tensor& self,
    at::OptionalIntArrayRef dims,
    bool keepdim,
    ReduceOp operation,
    double empty_value,
    bool has_subgroup)
{
    if (!is_dtype_supported(self.scalar_type())) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Vulkan device does not support ", self.scalar_type(), ". Falling back to CPU.");
        return cpu_reduce_fallback(self.cpu(), dims, keepdim, operation).to(self.device());
    }

    if (self.dim() > MAX_DIMS) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Tensor rank (", self.dim(), ") exceed maximum supported (", MAX_DIMS, "). Falling back to CPU.");
        return cpu_reduce_fallback(self.cpu(), dims, keepdim, operation).to(self.device());
    }

    std::vector<int64_t> dim_list;
    if (dims.has_value() && dims->size() > 0) {
        for (int64_t d : *dims) dim_list.push_back(c10::maybe_wrap_dim(d, self.dim()));
    } else {
        for (int64_t d = 0; d < self.dim(); d++) dim_list.push_back(d);
    }

    if (dim_list.empty()) return self.clone();

    // reduce highest dims first so earlier indices stay valid when keepdim=false
    std::sort(dim_list.begin(), dim_list.end(), std::greater<int64_t>());

    at::Tensor result = self;
    for (int64_t d : dim_list) {
        result = dispatch_reduce_shader(result, d, keepdim, operation, empty_value, has_subgroup);
    }
    return result;
}

at::Tensor sum_dim_vulkan(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype)
{
    at::Tensor self_typed = self.to(reduce_compute_dtype(self, dtype));
    return reduce_dims_vulkan(self_typed, dim, keepdim, ReduceOp::SUM, 0.0, true);
}

} // namespace

at::Tensor torchvulkan::sum_vulkan(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype)
{
    return sum_dim_vulkan(self, at::OptionalIntArrayRef(), false, dtype);
}

namespace {

at::Tensor amax_vulkan(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim)
{
    return reduce_dims_vulkan(self, dim, keepdim, ReduceOp::AMAX, -std::numeric_limits<double>::infinity(), true);
}

at::Tensor amin_vulkan(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim)
{
    return reduce_dims_vulkan(self, dim, keepdim, ReduceOp::AMIN, std::numeric_limits<double>::infinity(), true);
}

at::Tensor prod_dim_vulkan(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype)
{
    at::Tensor self_typed = self.to(reduce_compute_dtype(self, dtype));
    return reduce_dims_vulkan(self_typed, at::IntArrayRef{dim}, keepdim, ReduceOp::PROD, 1.0, false);
}

at::Tensor prod_vulkan(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype)
{
    at::Tensor self_typed = self.to(reduce_compute_dtype(self, dtype));
    return reduce_dims_vulkan(self_typed, at::OptionalIntArrayRef(), false, ReduceOp::PROD, 1.0, false);
}

at::Tensor mean_dim_vulkan(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype)
{
    if (self.dim() == 0) return dtype.has_value() ? self.to(*dtype) : self.clone();

    at::Tensor summed = sum_dim_vulkan(self, dim, keepdim, dtype);

    int64_t count = 1;
    if (dim.has_value() && dim->size() > 0) {
        for (int64_t d : *dim) count *= self.size(c10::maybe_wrap_dim(d, self.dim()));
    } else {
        count = self.numel();
    }

    return torchvulkan::divide_scalar_vulkan(summed, (double)count);
}

at::Tensor mean_vulkan(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype)
{
    return mean_dim_vulkan(self, at::OptionalIntArrayRef(), false, dtype);
}

enum class ArgOp 
{ 
    ARGMAX = 0, 
    ARGMIN = 1 
};

at::Tensor argreduce(
    const at::Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim,
    ArgOp op)
{
    at::Tensor x = self;
    bool flatten = !dim.has_value();
    int64_t d = 0;
    if (flatten) {
        x = x.reshape({-1});
        keepdim = false;
    } else {
        d = c10::maybe_wrap_dim(*dim, x.dim());
    }

    if (!is_dtype_supported(x.scalar_type()) || x.dim() == 0 || x.dim() > MAX_DIMS || x.numel() == 0) {
        at::Tensor cpu = op == ArgOp::ARGMAX ? at::argmax(self.cpu(), dim, keepdim) : at::argmin(self.cpu(), dim, keepdim);
        return cpu.to(self.device());
    }

    std::vector<int64_t> perm;
    for (int64_t i = 0; i < x.dim(); i++) if (i != d) perm.push_back(i);
    perm.push_back(d);
    at::Tensor xt = x.permute(perm).contiguous();

    int64_t reduce_size = xt.size(-1);
    int64_t num_lines = reduce_size > 0 ? xt.numel() / reduce_size : 0;
    at::Tensor idx = at::empty({num_lines}, self.options().dtype(at::kLong));

    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    uint32_t workgroupSizeX = get_dtype_workgroup_size(x.scalar_type(), 1);
    torchvulkan::ShaderID shader_id = torchvulkan::get_shader_id_arg_reduce(x.scalar_type());
    uint32_t opv = static_cast<uint32_t>(op);

    SpecializationBuilder spd{};
    spd.push(opv)
       .push(workgroupSizeX);
    SpecializationArgs specialization = spd.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(xt))
       .push(get_tensor_address(idx))
       .push(static_cast<uint64_t>(num_lines))
       .push(static_cast<uint32_t>(reduce_size));

    uint32_t groupX = (static_cast<uint32_t>(num_lines) + (workgroupSizeX - 1)) / workgroupSizeX;

    PushConstants pushConstants = { const_cast<void*>(pcs.data()), pcs.size() };
    device->shader_manager->dispatchShader(
        shader_id,
        specialization,
        pushConstants,
        /* read = */ {xt},
        /* write = */ {idx},
        groupX, 1, 1
    );

    std::vector<int64_t> oshape(xt.sizes().begin(), xt.sizes().end() - 1);
    at::Tensor result = idx.reshape(oshape);
    if (keepdim) result = result.unsqueeze(d);
    return result;
}

at::Tensor& argmax_out_vulkan(
    const at::Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim,
    at::Tensor& out)
{
    out.copy_(argreduce(self, dim, keepdim, ArgOp::ARGMAX));
    return out;
}

at::Tensor& argmin_out_vulkan(
    const at::Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim,
    at::Tensor& out)
{
    out.copy_(argreduce(self, dim, keepdim, ArgOp::ARGMIN));
    return out;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("sum.dim_IntList", &sum_dim_vulkan);
    m.impl("sum", &torchvulkan::sum_vulkan);
    m.impl("amax", &amax_vulkan);
    m.impl("amin", &amin_vulkan);
    m.impl("prod", &prod_vulkan);
    m.impl("prod.dim_int", &prod_dim_vulkan);
    m.impl("mean.dim", &mean_dim_vulkan);
    m.impl("mean", &mean_vulkan);
    m.impl("argmax.out", &argmax_out_vulkan);
    m.impl("argmin.out", &argmin_out_vulkan);
}
