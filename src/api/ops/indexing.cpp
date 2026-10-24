#include <torch/extension.h>
#include <ATen/ExpandUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/IndexingUtils.h>
#include <tuple>
#include <vector>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "api/ops/helpers.h"
#include "api/fallback.h"

namespace {

bool dispatch_index_gather_shader(
    const at::Tensor& out,
    const at::Tensor& src,
    const std::vector<at::Tensor>& indices,
    c10::IntArrayRef index_sizes,
    c10::IntArrayRef index_strides,
    bool wrap_negative)
{
    static const uint32_t INDEX_MAX_TENSORS = 4;
    
    const size_t num_indices = indices.size();
    if (num_indices == 0 || num_indices > INDEX_MAX_TENSORS) return false;

    at::TensorIteratorConfig config;
    config.check_all_same_dtype(false)
          .set_check_mem_overlap(false)
          .resize_outputs(false)
          .add_output(out)
          .add_const_input(src);
    for (const at::Tensor& index : indices) config.add_const_input(index);
    at::TensorIterator iter = config.build();

    uint64_t numel = iter.numel();
    if (numel == 0) return true;
    int32_t out_dims = static_cast<int32_t>(iter.ndim());
    if (out_dims > MAX_DIMS || !fits_32bit_indexing(iter)) return false;

    for (size_t k = 0; k < num_indices; k++) {
        if (index_sizes[k] > std::numeric_limits<int32_t>::max() || index_strides[k] > std::numeric_limits<uint32_t>::max()) return false;
    }

    DeviceContext* device = get_device_context(out);
    uint32_t workgroupSizeX = get_dtype_workgroup_size(out.scalar_type(), 1, device);

    SpecializationBuilder spd{};
    spd.push(out_dims)
       .push(static_cast<uint32_t>(num_indices))
       .push(workgroupSizeX)
       .push(static_cast<uint32_t>(wrap_negative));

    MetadataBuilder metadataBuilder = get_iter_metadata(iter, {0, 1});
    for (size_t k = 0; k < INDEX_MAX_TENSORS; k++)
    {
        uint32_t strides[MAX_DIMS] = {0};
        if (k < num_indices) {
            int operand = static_cast<int>(2 + k);
            for (int32_t i = 0; i < out_dims; i++) strides[i] = static_cast<uint32_t>(iter.strides(operand)[i] / iter.element_size(operand));
        }
        metadataBuilder.push_array(strides, static_cast<uint32_t>(out_dims));
    }

    uint32_t bounds[2 * INDEX_MAX_TENSORS] = {0};
    for (size_t k = 0; k < num_indices; k++) {
        bounds[k] = static_cast<uint32_t>(index_sizes[k]);
        bounds[num_indices + k] = static_cast<uint32_t>(index_strides[k]);
    }

    metadataBuilder.push_array(bounds, static_cast<uint32_t>(2 * num_indices));
    Metadata metadata = metadataBuilder.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(out))
       .push(get_tensor_address(src));
    for (size_t k = 0; k < INDEX_MAX_TENSORS; k++) pcs.push(k < num_indices ? get_tensor_address(indices[k]) : uint64_t(0));
    pcs.push_metadata(&metadata)
       .push(numel);

    std::vector<at::Tensor> reads = {src};
    reads.insert(reads.end(), indices.begin(), indices.end());

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::INDEX_GATHER, get_dtype_width_variant(out.scalar_type())},
        spd.build(),
        pcs.build(),
        /* read = */ reads,
        /* write = */ {out},
        get_group_count(numel, workgroupSizeX), 1, 1
    );
    return true;
}

at::Tensor restride_for_dim(const at::Tensor& self, int64_t dim, c10::IntArrayRef sizes)
{
    std::vector<int64_t> strides(self.strides().begin(), self.strides().end());
    strides[dim] = 0;
    return self.as_strided(sizes, strides, self.storage_offset());
}

at::Tensor& gather_out_vulkan(const at::Tensor& self, int64_t dim, const at::Tensor& index, bool /* sparse_grad */, at::Tensor& out)
{
    TORCH_CHECK_INDEX(at::isIntegralType(index.scalar_type(), /*includeBool=*/false), "gather(): Expected dtype int64 for index");
    if (out.sizes() != index.sizes()) out.resize_(index.sizes());
    if (index.numel() == 0) return out;

    int64_t ndim = std::max<int64_t>(self.dim(), 1);
    dim = c10::maybe_wrap_dim(dim, ndim);
    TORCH_CHECK(std::max<int64_t>(index.dim(), 1) == ndim, "Index tensor must have the same number of dimensions as input tensor");

    at::Tensor src = self.dim() == 0 ? self.reshape({1}) : self;
    at::Tensor idx = (index.dim() == 0 ? index.reshape({1}) : index).to(at::kLong);
    for (int64_t d = 0; d < ndim; d++) {
        if (d == dim) continue;
        TORCH_CHECK(idx.size(d) <= src.size(d), "Size does not match at dimension ", d, " expected index ", idx.sizes(),
            " to be no larger than self ", src.sizes(), " apart from dimension ", dim);
    }

    at::Tensor result = out.dim() == 0 ? out.view({1}) : out;
    if (src.size(dim) == 0 || !is_dtype_copyable(self, self.scalar_type()) ||
        !dispatch_index_gather_shader(result, restride_for_dim(src, dim, idx.sizes()), {idx}, {src.size(dim)}, {src.stride(dim)}, false))
    {
        TORCHVULKAN_CPU_FALLBACK("This gather can't run on the Vulkan device. Falling back to CPU.");
        out.copy_(at::gather(self.cpu(), dim, index.cpu()));
    }
    return out;
}

at::Tensor index_select_vulkan(const at::Tensor& self, int64_t dim, const at::Tensor& index)
{
    TORCH_CHECK_INDEX(index.dim() <= 1, "index_select(): Index is supposed to be a vector");
    TORCH_CHECK_INDEX(index.scalar_type() == at::kLong || index.scalar_type() == at::kInt, "index_select(): Expected dtype int32 or int64 for index");

    if (self.dim() == 0) {
        TORCHVULKAN_CPU_FALLBACK("index_select on a 0-dim tensor runs on the CPU.");
        return at::index_select(self.cpu(), dim, index.cpu()).to(self.device());
    }

    dim = c10::maybe_wrap_dim(dim, self.dim());
    int64_t count = index.numel();
    std::vector<int64_t> out_sizes(self.sizes().begin(), self.sizes().end());
    out_sizes[dim] = count;
    at::Tensor out = at::empty(out_sizes, self.options());
    if (out.numel() == 0) return out;

    std::vector<int64_t> index_view(self.dim(), 1);
    index_view[dim] = count;
    at::Tensor idx = index.to(at::kLong).reshape(index_view).expand(out_sizes);

    if (self.size(dim) == 0 || !is_dtype_copyable(self, self.scalar_type()) ||
        !dispatch_index_gather_shader(out, restride_for_dim(self, dim, out_sizes), {idx}, {self.size(dim)}, {self.stride(dim)}, false))
    {
        TORCHVULKAN_CPU_FALLBACK("This index_select can't run on the Vulkan device. Falling back to CPU.");
        return at::index_select(self.cpu(), dim, index.cpu()).to(self.device());
    }
    return out;
}

at::Tensor& index_select_out_vulkan(const at::Tensor& self, int64_t dim, const at::Tensor& index, at::Tensor& out)
{
    return fill_out(out, index_select_vulkan(self, dim, index));
}

struct AdvancedIndexPlan {
    at::Tensor src;
    std::vector<at::Tensor> indices;
    std::vector<int64_t> index_sizes;
    std::vector<int64_t> index_strides;
};

AdvancedIndexPlan plan_advanced_index(const at::Tensor& self_in, const c10::List<std::optional<at::Tensor>>& orig)
{
    at::native::checkIndexTensorTypes(orig, /*allow_int=*/true);
    at::Tensor self = self_in;
    std::vector<at::Tensor> indices = at::native::expandTensors(self, orig, /*ensure_same_device=*/true);
    try {
        indices = at::expand_outplace(indices);
    } catch (std::exception&) {
        TORCH_CHECK_INDEX(false, "shape mismatch: indexing tensors could not be broadcast together");
    }
    while (indices.size() < static_cast<size_t>(self.dim())) indices.emplace_back();
    if (!at::native::hasContiguousSubspace(indices)) std::tie(self, indices) = at::native::transposeToFront(self, indices);

    AdvancedIndexPlan plan;
    int64_t dims_before = 0, dims_after = 0, dims_indexed = 0;
    c10::IntArrayRef replacement_shape;
    for (size_t d = 0; d < indices.size(); d++)
    {
        if (!indices[d].defined()) {
            if (dims_indexed == 0) dims_before++;
            else dims_after++;
            continue;
        }
        dims_indexed++;
        replacement_shape = indices[d].sizes();
        plan.index_sizes.push_back(self.size(d));
        plan.index_strides.push_back(self.stride(d));
    }

    bool indexes_empty_dim = std::find(plan.index_sizes.begin(), plan.index_sizes.end(), 0) != plan.index_sizes.end();
    bool replacement_empty = std::find(replacement_shape.begin(), replacement_shape.end(), 0) != replacement_shape.end();
    TORCH_CHECK_INDEX(!indexes_empty_dim || replacement_empty, "index is out of bounds for dimension with size 0");

    std::vector<int64_t> shape(self.sizes().begin(), self.sizes().end());
    std::vector<int64_t> strides(self.strides().begin(), self.strides().end());
    shape.erase(shape.begin() + dims_before, shape.begin() + dims_before + dims_indexed);
    strides.erase(strides.begin() + dims_before, strides.begin() + dims_before + dims_indexed);
    shape.insert(shape.begin() + dims_before, replacement_shape.begin(), replacement_shape.end());
    strides.insert(strides.begin() + dims_before, replacement_shape.size(), 0);
    plan.src = self.as_strided(shape, strides, self.storage_offset());

    for (const at::Tensor& index : indices)
    {
        if (!index.defined()) continue;
        std::vector<int64_t> index_shape(dims_before, 1);
        index_shape.insert(index_shape.end(), index.sizes().begin(), index.sizes().end());
        index_shape.insert(index_shape.end(), dims_after, 1);
        plan.indices.push_back(index.to(at::kLong).reshape(index_shape));
    }
    return plan;
}

at::Tensor& index_out_vulkan(const at::Tensor& self, const c10::List<std::optional<at::Tensor>>& indices, at::Tensor& out)
{
    AdvancedIndexPlan plan = plan_advanced_index(self, indices);
    if (out.sizes() != plan.src.sizes()) out.resize_(plan.src.sizes());
    if (out.numel() == 0) return out;

    if (!is_dtype_copyable(self, self.scalar_type()) ||
        !dispatch_index_gather_shader(out, plan.src, plan.indices, plan.index_sizes, plan.index_strides, /*wrap_negative=*/true))
    {
        TORCHVULKAN_CPU_FALLBACK("This index can't run on the Vulkan device. Falling back to CPU.");
        c10::List<std::optional<at::Tensor>> cpu_indices;
        for (const std::optional<at::Tensor>& index : indices) {
            cpu_indices.push_back(index.has_value() && index->defined() ? std::optional<at::Tensor>(index->cpu()) : std::nullopt);
        }
        out.copy_(at::index(self.cpu(), cpu_indices));
    }
    return out;
}

at::Tensor embedding_vulkan(
    const at::Tensor& weight,
    const at::Tensor& indices,
    c10::SymInt /* padding_idx */,
    bool /* scale_grad_by_freq */,
    bool sparse)
{
    TORCH_CHECK(!sparse, "torchvulkan [NOT IMPLEMENTED]: sparse embedding gradients are not supported.");
    TORCH_CHECK(weight.dim() == 2, "'weight' must be 2-D");

    std::vector<int64_t> out_shape(indices.sizes().begin(), indices.sizes().end());
    out_shape.push_back(weight.size(1));
    return index_select_vulkan(weight, 0, indices.reshape({-1})).view(out_shape);
}

at::Tensor embedding_dense_backward_vulkan(
    const at::Tensor& grad_output,
    const at::Tensor& indices,
    c10::SymInt num_weights,
    c10::SymInt padding_idx,
    bool scale_grad_by_freq)
{
    int64_t num_weights_ = num_weights.guard_int(__FILE__, __LINE__);
    int64_t padding_idx_ = padding_idx.guard_int(__FILE__, __LINE__);
    int64_t dim = grad_output.size(-1);

    at::Tensor flat_idx = indices.reshape({-1}).to(at::kLong);
    int64_t n = flat_idx.numel();
    at::Tensor grad_weight = at::empty({num_weights_, dim}, grad_output.options());
    if (grad_weight.numel() == 0) return grad_weight;
    if (n == 0) return grad_weight.zero_();

    at::Tensor flat_grad = grad_output.reshape({n, dim}).contiguous();
    DeviceContext* device = get_device_context(grad_output);
    if (!at::isFloatingType(grad_output.scalar_type()) || !is_dtype_supported(device, grad_output.scalar_type()) ||
        !fits_32bit_indexing({flat_grad, grad_weight}))
    {
        TORCHVULKAN_CPU_FALLBACK("This embedding backward can't run on the Vulkan device. Falling back to CPU.");
        return at::embedding_dense_backward(grad_output.cpu(), indices.cpu(), num_weights_, padding_idx_, scale_grad_by_freq).to(grad_output.device());
    }

    // sorting groups each row's occurrences, so each output sums its own run without atomics
    auto [sorted, permutation] = at::sort(flat_idx, /*stable=*/true, /*dim=*/0, /*descending=*/false);
    sorted = sorted.contiguous();
    permutation = permutation.contiguous();

    uint64_t numel_out = grad_weight.numel();
    uint32_t workgroupSizeX = get_dtype_workgroup_size(grad_output.scalar_type(), 1, device);

    SpecializationBuilder spd{};
    spd.push(workgroupSizeX);

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(flat_grad))
       .push(get_tensor_address(sorted))
       .push(get_tensor_address(permutation))
       .push(get_tensor_address(grad_weight))
       .push(padding_idx_)
       .push(numel_out)
       .push(static_cast<uint32_t>(n))
       .push(static_cast<uint32_t>(dim))
       .push(static_cast<uint32_t>(scale_grad_by_freq));

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::EMBEDDING_BACKWARD, grad_output.scalar_type()},
        spd.build(),
        pcs.build(),
        /* read = */ {flat_grad, sorted, permutation},
        /* write = */ {grad_weight},
        get_group_count(numel_out, workgroupSizeX), 1, 1
    );
    return grad_weight;
}

std::tuple<at::Tensor&, at::Tensor&> extremum_dim_out(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    bool is_max,
    at::Tensor& values,
    at::Tensor& indices)
{
    // argmax has no bool kernel; uint8 orders false before true the same way
    at::Tensor key = self.scalar_type() == at::kBool ? self.to(at::kByte) : self;
    at::Tensor idx = is_max ? at::argmax(key, dim, /*keepdim=*/true) : at::argmin(key, dim, /*keepdim=*/true);
    at::Tensor vals = at::gather(self, self.dim() == 0 ? 0 : dim, idx);
    if (!keepdim && self.dim() > 0) {
        int64_t d = c10::maybe_wrap_dim(dim, self.dim());
        idx = idx.squeeze(d);
        vals = vals.squeeze(d);
    }
    fill_out(values, vals);
    fill_out(indices, idx);
    return std::forward_as_tuple(values, indices);
}

std::tuple<at::Tensor&, at::Tensor&> max_dim_out_vulkan(const at::Tensor& self, int64_t dim, bool keepdim, at::Tensor& max, at::Tensor& max_indices)
{
    return extremum_dim_out(self, dim, keepdim, true, max, max_indices);
}

std::tuple<at::Tensor&, at::Tensor&> min_dim_out_vulkan(const at::Tensor& self, int64_t dim, bool keepdim, at::Tensor& min, at::Tensor& min_indices)
{
    return extremum_dim_out(self, dim, keepdim, false, min, min_indices);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("gather.out", &gather_out_vulkan);
    m.impl("index_select", &index_select_vulkan);
    m.impl("index_select.out", &index_select_out_vulkan);
    m.impl("index.Tensor_out", &index_out_vulkan);
    m.impl("embedding", &embedding_vulkan);
    m.impl("embedding_dense_backward", &embedding_dense_backward_vulkan);
    m.impl("max.dim_max", &max_dim_out_vulkan);
    m.impl("min.dim_min", &min_dim_out_vulkan);
}
