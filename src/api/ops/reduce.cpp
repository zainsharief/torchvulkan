#include <torch/extension.h>
#include <ATen/WrapDimUtils.h>
#include <limits>
#include "api/ops/reduce.h"
#include "api/ops/binary.h"

at::Tensor torchvulkan::dispatch_reduce_shader(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    ReduceOp operation)
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
        out.fill_(operation == ReduceOp::SUM ? 0.0 : -std::numeric_limits<double>::infinity());
        return keepdim ? out : out.squeeze(dim);
    }

    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    uint32_t workgroupSizeX = get_dtype_workgroup_size(self.scalar_type(), 1);
    torchvulkan::ShaderID shader_id = torchvulkan::get_shader_id_reduce(self.scalar_type());
    uint32_t op = static_cast<uint32_t>(operation);
    int32_t ndim32 = static_cast<int32_t>(reduce_ndim);

    SpecializationBuilder spd{};
    spd.push(op)
       .push(ndim32)
       .push(workgroupSizeX);
    uint32_t key = (workgroupSizeX << 8) | (ndim32 << 4) | op;
    SpecializationArgs specialization = {spd.data(), spd.offsets(), spd.sizes(), spd.numConstants(), key};

    IntDivider sizes;
    uint32_t strides_in[MAX_DIMS] = {0};
    for (int64_t i = 0; i < reduce_ndim; i++) {
        int64_t actual_dim = reduce_ndim - 1 - i;
        sizes.set(i, static_cast<uint32_t>(out_sizes[actual_dim]));
        strides_in[i] = static_cast<uint32_t>(self_strides[actual_dim]);
    }

    PushConstantBuilder pcs{};
    pcs.push(sizes)
       .push_array(strides_in)
       .push(static_cast<uint32_t>(self_strides[dim]))
       .push(static_cast<uint32_t>(reduce_size))
       .push(numel_out);

    uint32_t groupX = (numel_out + (workgroupSizeX - 1)) / workgroupSizeX;

    VulkanShader shader(shader_id, specialization, device);
    shader.dispatch(
        &pcs,
        pcs.size(),
        {self, out},
        groupX, 1, 1
    );

    return keepdim ? out : out.squeeze(dim);
}

at::Tensor torchvulkan::reduce_dims_vulkan(
    const at::Tensor& self,
    at::OptionalIntArrayRef dims,
    bool keepdim,
    ReduceOp operation)
{
    if (!is_dtype_supported(self.scalar_type())) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Vulkan device does not support ", self.scalar_type(), ". Falling back to CPU.");
        at::Tensor cpu_self = self.cpu();
        at::Tensor cpu_result = operation == ReduceOp::SUM
            ? at::sum(cpu_self, dims, keepdim)
            : at::amax(cpu_self, dims.has_value() ? *dims : at::IntArrayRef{}, keepdim);
        return cpu_result.to(self.device());
    }

    if (self.dim() > MAX_DIMS) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Tensor rank (", self.dim(), ") exceed maximum supported (", MAX_DIMS, "). Falling back to CPU.");
        at::Tensor cpu_self = self.cpu();
        at::Tensor cpu_result = operation == ReduceOp::SUM
            ? at::sum(cpu_self, dims, keepdim)
            : at::amax(cpu_self, dims.has_value() ? *dims : at::IntArrayRef{}, keepdim);
        return cpu_result.to(self.device());
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
        result = dispatch_reduce_shader(result, d, keepdim, operation);
    }
    return result;
}

at::Tensor torchvulkan::sum_dim_vulkan(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype)
{
    c10::ScalarType compute_dtype = dtype.value_or(self.scalar_type());
    if (!dtype.has_value() && at::isIntegralType(self.scalar_type(), /*includeBool=*/true) && self.scalar_type() != at::kLong) {
        compute_dtype = at::kLong;
    }

    at::Tensor self_typed = self.to(compute_dtype);
    return reduce_dims_vulkan(self_typed, dim, keepdim, ReduceOp::SUM);
}

at::Tensor torchvulkan::sum_vulkan(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype)
{
    return sum_dim_vulkan(self, at::OptionalIntArrayRef(), false, dtype);
}

at::Tensor torchvulkan::amax_vulkan(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim)
{
    return reduce_dims_vulkan(self, dim, keepdim, ReduceOp::AMAX);
}

at::Tensor torchvulkan::mean_dim_vulkan(
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

    return divide_scalar_vulkan(summed, (double)count);
}

at::Tensor torchvulkan::mean_vulkan(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype)
{
    return mean_dim_vulkan(self, at::OptionalIntArrayRef(), false, dtype);
}
