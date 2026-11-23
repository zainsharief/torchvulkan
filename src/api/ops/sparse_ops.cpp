#include <torch/extension.h>
#include <ATen/ops/_to_sparse_native.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/_to_sparse_csr_native.h>
#include <ATen/ops/_to_sparse_csc_native.h>
#include <ATen/ops/_to_sparse_bsr_native.h>
#include <ATen/ops/_to_sparse_bsc_native.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/DispatchStub.h>
#include <limits>
#include <algorithm>
#include <vector>
#include "api/ops/helpers.h"

namespace {

// the row-major position of each nonzero's sparse index
at::Tensor linear_index(const at::Tensor& indices, at::IntArrayRef sparse_sizes)
{
    at::Tensor linear = at::zeros({indices.size(1)}, indices.options());
    for (size_t d = 0; d < sparse_sizes.size(); d++) {
        linear = at::add(at::mul(linear, sparse_sizes[d]), indices.select(0, static_cast<int64_t>(d)));
    }
    return linear;
}

at::Tensor unlinear_index(const at::Tensor& linear, at::IntArrayRef sparse_sizes)
{
    std::vector<at::Tensor> rows(sparse_sizes.size());
    at::Tensor rest = linear;
    for (int64_t d = static_cast<int64_t>(sparse_sizes.size()) - 1; d >= 0; d--) {
        rows[d] = at::remainder(rest, sparse_sizes[d]);
        rest = at::div(rest, sparse_sizes[d], "floor");
    }
    return rows.empty() ? at::zeros({0, linear.size(0)}, linear.options()) : at::stack(rows);
}

// sort the indices and sum the values of repeated ones (CPU and CUDA each have their own kernel)
at::Tensor coalesce_sparse_vulkan(const at::Tensor& self)
{
    int64_t sparse_dim = self.sparse_dim(), dense_dim = self.dense_dim();
    at::Tensor indices = self._indices(), values = self._values();
    int64_t nnz = indices.size(1);
    if (nnz < 2) {
        return at::_sparse_coo_tensor_with_dims_and_tensors(sparse_dim, dense_dim, self.sizes(), indices.clone(), values.clone(),
                                                            self.options(), /*is_coalesced=*/true);
    }
    at::IntArrayRef sparse_sizes = self.sizes().slice(0, sparse_dim);
    auto [sorted, order] = at::sort(linear_index(indices, sparse_sizes), /*stable=*/true, /*dim=*/0, /*descending=*/false);
    auto [unique, inverse, counts] = at::unique_consecutive(sorted, /*return_inverse=*/true);
    std::vector<int64_t> value_shape = {unique.size(0)};
    value_shape.insert(value_shape.end(), values.sizes().begin() + 1, values.sizes().end());
    at::Tensor summed = at::zeros(value_shape, values.options()).index_add_(0, inverse, at::index_select(values, 0, order));
    return at::_sparse_coo_tensor_with_dims_and_tensors(sparse_dim, dense_dim, self.sizes(), unlinear_index(unique, sparse_sizes), summed,
                                                        self.options(), /*is_coalesced=*/true);
}

int64_t sparse_numel(at::IntArrayRef sparse_sizes)
{
    int64_t n = 1;
    for (int64_t s : sparse_sizes) n *= s;
    return n;
}

// out (a sparse tensor) becomes the given indices and values
at::Tensor& set_sparse(at::Tensor& out, at::IntArrayRef sizes, int64_t sparse_dim, const at::Tensor& indices, const at::Tensor& values, bool coalesced)
{
    at::sparse::get_sparse_impl(out)->resize_(sparse_dim, static_cast<int64_t>(sizes.size()) - sparse_dim, sizes);
    at::sparse::alias_into_sparse(out, indices, values.to(out.scalar_type()));
    out._coalesced_(coalesced);
    return out;
}

// a dense tensor plus alpha times a sparse one: the values added at their row-major positions
at::Tensor dense_plus_sparse(const at::Tensor& dense, const at::Tensor& sparse, const at::Scalar& alpha, at::ScalarType dtype)
{
    int64_t sparse_dim = sparse.sparse_dim();
    at::IntArrayRef sparse_sizes = sparse.sizes().slice(0, sparse_dim);
    std::vector<int64_t> flat_shape = {sparse_numel(sparse_sizes)};
    flat_shape.insert(flat_shape.end(), sparse.sizes().begin() + sparse_dim, sparse.sizes().end());
    at::Tensor result = dense.to(dtype).expand(sparse.sizes()).contiguous().clone().view(flat_shape);
    if (sparse._nnz() > 0) result.index_add_(0, linear_index(sparse._indices(), sparse_sizes), sparse._values().to(dtype), alpha);
    return result.view(sparse.sizes());
}

at::Tensor& add_out_sparse_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha, at::Tensor& out)
{
    at::ScalarType common = at::result_type(self, other);
    TORCH_CHECK(at::canCast(common, out.scalar_type()), "Can't convert result type ", common, " to output ", out.scalar_type(), " in add operation");
    if (!out.is_sparse()) {
        TORCH_CHECK(!self.is_sparse(), "add(sparse, dense) is not supported. Use add(dense, sparse) instead.");
        at::Tensor result = dense_plus_sparse(self, other, alpha, out.scalar_type());
        at::native::resize_output(out, result.sizes());
        return out.copy_(result);
    }
    TORCH_CHECK(self.is_sparse() && other.is_sparse(), "add(sparse, dense) is not supported. Use add(dense, sparse) instead.");
    TORCH_CHECK(self.sizes().equals(other.sizes()), "add: expected 'self' and 'other' to have same size, but self has size ", self.sizes(),
                " while other has size ", other.sizes(), " (FYI: dense-sparse addition does not currently support broadcasting)");
    at::ScalarType dtype = out.scalar_type();
    at::Tensor indices = at::cat({self._indices(), other._indices()}, 1);
    at::Tensor values = at::cat({self._values().to(dtype), at::mul(other._values().to(dtype), alpha)}, 0);
    bool coalesced = self.is_coalesced() && other.is_coalesced();
    at::Tensor sum = at::_sparse_coo_tensor_with_dims_and_tensors(self.sparse_dim(), self.dense_dim(), self.sizes(), indices, values, out.options());
    if (coalesced) sum = sum.coalesce();
    return set_sparse(out, self.sizes(), self.sparse_dim(), sum._indices(), sum._values(), coalesced);
}

// sparse * sparse keeps the indices both have; sparse * dense scales each value by the dense element at its index
at::Tensor& mul_out_sparse_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out)
{
    at::ScalarType common = at::result_type(self, other);
    TORCH_CHECK(at::canCast(common, out.scalar_type()), "Can't convert result type ", common, " to output ", out.scalar_type());
    // a 0-dim sparse operand is a scalar: as the CPU's mul_out_sparse (other checked first), the other operand's pattern
    // is kept, with no entry dropped for being zero
    if (self.is_sparse() && other.is_sparse()) {
        // both 0-dim: the pattern of the one with more stored elements, as the CPU's results show
        bool scalar_other = other.dim() == 0 && (self.dim() > 0 || other._nnz() <= self._nnz());
        if (scalar_other) return mul_out_sparse_vulkan(self, other.to_dense(), out);
        if (self.dim() == 0) return mul_out_sparse_vulkan(self.to_dense(), other, out);
    }
    at::ScalarType dtype = out.scalar_type();
    if (self.is_sparse() && other.is_sparse() && self.dim() > 0 && other.dim() > 0) {
        TORCH_CHECK(self.sparse_dim() == other.sparse_dim() && self.dim() == other.dim() &&
                    self.sizes().slice(0, self.sparse_dim()).equals(other.sizes().slice(0, other.sparse_dim())),
                    "sparse_binary_op_intersection_cpu(): expects sparse inputs with equal dimensionality, number of sparse dimensions, "
                    "and shape of sparse dimensions");
    }
    // broadcasting shapes (or a dense out, as in dense.mul_(sparse)): through the dense product, which keeps the (larger)
    // sparse operand's dense dims and makes the rest sparse, as the CPU does
    bool broadcasting = !self.sizes().equals(other.sizes()) && !(!self.is_sparse() && self.dim() == 0) && !(!other.is_sparse() && other.dim() == 0);
    if (broadcasting || !out.is_sparse()) {
        at::Tensor product = at::mul(self.is_sparse() ? self.to_dense() : self, other.is_sparse() ? other.to_dense() : other).to(dtype);
        if (!out.is_sparse()) {
            at::native::resize_output(out, product.sizes());
            return out.copy_(product);
        }
        bool self_leads = self.is_sparse() && (!other.is_sparse() || self.dim() >= other.dim());
        const at::Tensor& sparse = self_leads ? self : other;
        int64_t sparse_dim = product.dim() - sparse.dense_dim();
        if (sparse_dim == 0) {
            // no sparse dims: one element whose value is the whole product
            at::Tensor indices = at::zeros({0, 1}, product.options().dtype(at::kLong));
            return set_sparse(out, product.sizes(), 0, indices, product.unsqueeze(0), true);
        }
        at::Tensor result = product.to_sparse(sparse_dim);
        return set_sparse(out, result.sizes(), result.sparse_dim(), result._indices(), result._values(), true);
    }
    if (self.is_sparse() && other.is_sparse()) {
        TORCH_CHECK(self.sizes().equals(other.sizes()), "mul: expected 'self' and 'other' to have same sizes, but ", self.sizes(), " != ", other.sizes());
        at::Tensor a = self.coalesce(), b = other.coalesce();
        at::IntArrayRef sparse_sizes = self.sizes().slice(0, self.sparse_dim());
        at::Tensor la = linear_index(a._indices(), sparse_sizes), lb = linear_index(b._indices(), sparse_sizes);
        at::Tensor indices = a._indices().narrow(1, 0, 0), values = a._values().narrow(0, 0, 0).to(dtype);
        if (la.numel() > 0 && lb.numel() > 0) {
            at::Tensor position = at::clamp_max(at::searchsorted(lb, la), lb.numel() - 1);
            at::Tensor shared = at::nonzero(at::eq(at::index_select(lb, 0, position), la)).squeeze(1);
            indices = at::index_select(a._indices(), 1, shared);
            values = at::mul(at::index_select(a._values(), 0, shared).to(dtype), at::index_select(b._values(), 0, at::index_select(position, 0, shared)).to(dtype));
        }
        return set_sparse(out, self.sizes(), self.sparse_dim(), indices, values, true);
    }
    const at::Tensor& sparse = self.is_sparse() ? self : other;
    const at::Tensor& dense = self.is_sparse() ? other : self;
    at::Tensor values;
    if (dense.dim() == 0) {
        // a CPU scalar stays one: the binary op reads it as a value (moved over it would be a float64 tensor)
        values = at::mul(sparse._values().to(dtype), dense.is_cpu() ? dense : dense.to(sparse.device()));
    } else {
        TORCH_CHECK(dense.sizes().equals(sparse.sizes()), "mul: a sparse tensor times a dense one needs the same shape, but got ", sparse.sizes(), " and ", dense.sizes());
        int64_t sparse_dim = sparse.sparse_dim();
        at::IntArrayRef sparse_sizes = sparse.sizes().slice(0, sparse_dim);
        std::vector<int64_t> flat_shape = {sparse_numel(sparse_sizes)};
        flat_shape.insert(flat_shape.end(), sparse.sizes().begin() + sparse_dim, sparse.sizes().end());
        at::Tensor picked = at::index_select(dense.contiguous().view(flat_shape), 0, linear_index(sparse._indices(), sparse_sizes));
        values = at::mul(sparse._values().to(dtype), picked.to(dtype));
    }
    return set_sparse(out, sparse.sizes(), sparse.sparse_dim(), sparse._indices().clone(), values, sparse.is_coalesced());
}

// the dense product of a 2-D sparse matrix with a dense one: each nonzero scales a row of the dense matrix into its row
at::Tensor sparse_dense_mm(const at::Tensor& sparse, const at::Tensor& dense)
{
    TORCH_CHECK(sparse.sparse_dim() == 2 && sparse.dense_dim() == 0, "addmm: matrices expected, got ", sparse.sparse_dim(), "D tensor");
    TORCH_CHECK(dense.dim() == 2, "addmm: matrices expected, got ", dense.dim(), "D tensor");
    TORCH_CHECK(sparse.size(1) == dense.size(0), "mat1 and mat2 shapes cannot be multiplied (", sparse.size(0), "x", sparse.size(1), " and ",
                dense.size(0), "x", dense.size(1), ")");
    at::ScalarType dtype = at::result_type(sparse, dense);
    at::Tensor result = at::zeros({sparse.size(0), dense.size(1)}, dense.options().dtype(dtype));
    if (sparse._nnz() == 0) return result;
    at::Tensor rows = sparse._indices().select(0, 0), cols = sparse._indices().select(0, 1);
    at::Tensor contributions = at::mul(at::index_select(dense.to(dtype), 0, cols), sparse._values().to(dtype).unsqueeze(1));
    return result.index_add_(0, rows, contributions);
}

at::Tensor addmm_sparse_vulkan(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2, const at::Scalar& beta, const at::Scalar& alpha)
{
    for (const at::Tensor* t : {&mat1, &mat2}) {
        TORCH_CHECK(t->device() == self.device(), "Expected all tensors to be on the same device, but found at least two devices, ", self.device(),
            " and ", t->device(), "!");
    }
    TORCH_CHECK(mat1.is_sparse() && !mat2.is_sparse(), "addmm: only a sparse mat1 with a dense mat2 is supported on vulkan");
    at::Tensor product = at::mul(sparse_dense_mm(mat1, mat2), alpha);
    at::Tensor base = self.is_sparse() ? self.to_dense() : self;
    if (beta.toComplexDouble() == c10::complex<double>(0.0)) return product.expand({mat1.size(0), mat2.size(1)}).contiguous();
    return at::add(at::mul(base.expand({mat1.size(0), mat2.size(1)}), beta), product);
}

at::Tensor& addmm_out_sparse_vulkan(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2, const at::Scalar& beta, const at::Scalar& alpha, at::Tensor& out)
{
    at::Tensor result = addmm_sparse_vulkan(self, mat1, mat2, beta, alpha);
    at::native::resize_output(out, result.sizes());
    return out.copy_(result);
}

at::Tensor& addmm_sparse_vulkan_(at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2, const at::Scalar& beta, const at::Scalar& alpha)
{
    return self.copy_(addmm_sparse_vulkan(self, mat1, mat2, beta, alpha));
}

// self's values where mask has the same index (times the number of matches with accumulate_matches), zero elsewhere;
// torch's shared kernel goes through a CPU/CUDA-only DispatchStub
at::Tensor sparse_mask_projection_vulkan(const at::Tensor& self, const at::Tensor& mask, bool accumulate_matches)
{
    TORCH_CHECK(self.sizes().equals(mask.sizes()), "_sparse_mask_projection: self and mask must have the same shape");
    at::IntArrayRef sparse_sizes = self.sizes().slice(0, self.sparse_dim());
    at::Tensor values = self._values();
    at::Tensor matches = at::zeros({self._nnz()}, self._indices().options());
    if (self._nnz() > 0 && mask._nnz() > 0) {
        at::Tensor wanted = std::get<0>(at::sort(linear_index(mask._indices(), sparse_sizes)));
        at::Tensor have = linear_index(self._indices(), sparse_sizes);
        matches = at::sub(at::searchsorted(wanted, have, false, /*right=*/true), at::searchsorted(wanted, have, false, /*right=*/false));
    }
    at::Tensor scale = (accumulate_matches ? matches : at::gt(matches, 0)).to(values.scalar_type());
    std::vector<int64_t> view(values.dim(), 1);
    view[0] = values.size(0);
    return at::_sparse_coo_tensor_with_dims_and_tensors(self.sparse_dim(), self.dense_dim(), self.sizes(), self._indices().clone(),
                                                        at::mul(values, scale.view(view)), self.options(), self.is_coalesced());
}

at::Tensor& sparse_mask_projection_out_vulkan(const at::Tensor& self, const at::Tensor& mask, bool accumulate_matches, at::Tensor& out)
{
    at::Tensor result = sparse_mask_projection_vulkan(self, mask, accumulate_matches);
    return set_sparse(out, result.sizes(), result.sparse_dim(), result._indices(), result._values(), result.is_coalesced());
}

// sparse @ sparse: the first times the dense form of the second, back to sparse
at::Tensor sparse_sparse_matmul_vulkan(const at::Tensor& self, const at::Tensor& other)
{
    // the checks of the CPU's sparse_sparse_matmul
    TORCH_CHECK(self.dim() == 2 && other.dim() == 2, "sparse_sparse_matmul: matrices expected, got ", self.dim(), "-D and ", other.dim(), "-D");
    TORCH_CHECK(self.dense_dim() == 0, "sparse_sparse_matmul_cpu: scalar values expected, got ", self.dense_dim(), "D values");
    TORCH_CHECK(other.dense_dim() == 0, "sparse_sparse_matmul_cpu: scalar values expected, got ", other.dense_dim(), "D values");
    TORCH_CHECK(self.size(1) == other.size(0), "mat1 and mat2 shapes cannot be multiplied (", self.size(0), "x", self.size(1), " and ",
        other.size(0), "x", other.size(1), ")");
    TORCH_CHECK(self.scalar_type() == other.scalar_type(), "mat1 dtype ", self.scalar_type(), " does not match mat2 dtype ", other.scalar_type());
    return sparse_dense_mm(self, other.to_dense()).to_sparse();
}

// a batch of sparse [M, K] matrices times dense [K, N] ones
at::Tensor bmm_sparse_vulkan(const at::Tensor& self, const at::Tensor& mat2)
{
    TORCH_CHECK(self.sparse_dim() == 3 && self.dense_dim() == 0 && mat2.dim() == 3, "bmm_sparse: expected a 3-D sparse tensor and a 3-D dense tensor");
    int64_t batches = self.size(0), m = self.size(1), k = self.size(2), n = mat2.size(2);
    TORCH_CHECK(mat2.size(0) == batches && mat2.size(1) == k, "bmm_sparse: mat1 and mat2 shapes cannot be multiplied");
    at::ScalarType dtype = at::result_type(self, mat2);
    at::Tensor result = at::zeros({batches * m, n}, mat2.options().dtype(dtype));
    if (self._nnz() > 0) {
        at::Tensor b = self._indices().select(0, 0), i = self._indices().select(0, 1), j = self._indices().select(0, 2);
        at::Tensor rows = at::index_select(mat2.to(dtype).reshape({batches * k, n}), 0, at::add(at::mul(b, k), j));
        result.index_add_(0, at::add(at::mul(b, m), i), at::mul(rows, self._values().to(dtype).unsqueeze(1)));
    }
    return result.view({batches, m, n});
}

at::Tensor& bmm_out_sparse_vulkan(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out)
{
    at::Tensor result = bmm_sparse_vulkan(self, mat2);
    at::native::resize_output(out, result.sizes());
    return out.copy_(result);
}

// the slices of a sparse tensor at the given positions of dim
at::Tensor index_select_sparse_vulkan(const at::Tensor& self, int64_t dim, const at::Tensor& index)
{
    dim = c10::maybe_wrap_dim(dim, self.dim());
    TORCH_CHECK_INDEX(index.dim() <= 1, "index_select() argument index must be 1-D");
    at::Tensor idx = index.reshape({-1}).to(at::kLong);
    std::vector<int64_t> sizes = self.sizes().vec();
    sizes[dim] = idx.numel();
    int64_t sparse_dim = self.sparse_dim();
    if (dim >= sparse_dim) {
        at::Tensor values = at::index_select(self._values(), dim - sparse_dim + 1, idx);
        return at::_sparse_coo_tensor_with_dims_and_tensors(sparse_dim, self.dense_dim(), sizes, self._indices().clone(), values, self.options(), self.is_coalesced());
    }
    at::Tensor coordinate = self._indices().select(0, dim);
    at::Tensor pairs = at::nonzero(at::eq(coordinate.unsqueeze(1), idx.unsqueeze(0)));  // (nonzero, position in index)
    at::Tensor which = pairs.select(1, 0), position = pairs.select(1, 1);
    at::Tensor indices = at::index_select(self._indices(), 1, which).clone();
    indices.select(0, dim).copy_(position);
    return at::_sparse_coo_tensor_with_dims_and_tensors(sparse_dim, self.dense_dim(), sizes, indices, at::index_select(self._values(), 0, which), self.options());
}

// softmax over a sparse dim runs within each group of nonzeros that share every other sparse index
// (missing elements count as -inf); over a dense dim it is the dense softmax of the values
struct SoftmaxGroups {
    at::Tensor group;   // each nonzero's group, expanded over the dense dims
    int64_t count;
};

SoftmaxGroups softmax_groups(const at::Tensor& coalesced, int64_t dim)
{
    int64_t sparse_dim = coalesced.sparse_dim();
    std::vector<int64_t> keep, keep_sizes;
    for (int64_t d = 0; d < sparse_dim; d++) {
        if (d == dim) continue;
        keep.push_back(d);
        keep_sizes.push_back(coalesced.size(d));
    }
    at::Tensor indices = coalesced._indices();
    at::Tensor key = keep.empty() ? at::zeros({indices.size(1)}, indices.options())
                                  : linear_index(at::index_select(indices, 0, at::tensor(keep, indices.options())), keep_sizes);
    auto [unique, inverse, counts] = at::_unique2(key, /*sorted=*/true, /*return_inverse=*/true, /*return_counts=*/false);
    at::Tensor values = coalesced._values();
    std::vector<int64_t> view(values.dim(), 1);
    view[0] = values.size(0);
    return {inverse.view(view).expand(values.sizes()), unique.size(0)};
}

at::Tensor group_reduce(const at::Tensor& values, const SoftmaxGroups& g, c10::string_view reduce, double init)
{
    std::vector<int64_t> shape = values.sizes().vec();
    shape[0] = g.count;
    return at::full(shape, init, values.options()).scatter_reduce(0, g.group, values, reduce, /*include_self=*/true).gather(0, g.group);
}

at::Tensor sparse_softmax_impl(const at::Tensor& self, int64_t dim, bool log)
{
    dim = c10::maybe_wrap_dim(dim, self.dim());
    at::Tensor x = self.coalesce();
    at::Tensor values = x._values();
    at::Tensor out;
    if (values.numel() == 0) {
        out = values.clone();
    } else if (dim >= x.sparse_dim()) {
        out = log ? at::_log_softmax(values, dim - x.sparse_dim() + 1, false) : at::_softmax(values, dim - x.sparse_dim() + 1, false);
    } else {
        SoftmaxGroups g = softmax_groups(x, dim);
        at::Tensor shifted = at::sub(values, group_reduce(values, g, "amax", -std::numeric_limits<double>::infinity()));
        at::Tensor total = group_reduce(at::exp(shifted), g, "sum", 0.0);
        out = log ? at::sub(shifted, at::log(total)) : at::div(at::exp(shifted), total);
    }
    return at::_sparse_coo_tensor_with_dims_and_tensors(x.sparse_dim(), x.dense_dim(), x.sizes(), x._indices().clone(), out, x.options(), true);
}

at::Tensor sparse_softmax_vulkan(const at::Tensor& self, int64_t dim, bool /* half_to_float */)
{
    return sparse_softmax_impl(self, dim, false);
}

at::Tensor sparse_log_softmax_vulkan(const at::Tensor& self, int64_t dim, bool /* half_to_float */)
{
    return sparse_softmax_impl(self, dim, true);
}

// softmax: out * (g - sum(g * out)); log_softmax: g - exp(out) * sum(g), over the same groups
at::Tensor sparse_softmax_backward_impl(const at::Tensor& grad_output, const at::Tensor& output, int64_t dim, bool log)
{
    dim = c10::maybe_wrap_dim(dim, output.dim());
    at::Tensor out = output.coalesce(), grad = grad_output.coalesce();
    at::Tensor y = out._values(), g = grad._values();
    at::Tensor result;
    if (y.numel() == 0) {
        result = y.clone();
    } else if (dim >= out.sparse_dim()) {
        int64_t d = dim - out.sparse_dim() + 1;
        result = log ? at::sub(g, at::mul(at::exp(y), at::sum(g, d, true))) : at::mul(y, at::sub(g, at::sum(at::mul(g, y), d, true)));
    } else {
        SoftmaxGroups groups = softmax_groups(out, dim);
        result = log ? at::sub(g, at::mul(at::exp(y), group_reduce(g, groups, "sum", 0.0)))
                     : at::mul(y, at::sub(g, group_reduce(at::mul(g, y), groups, "sum", 0.0)));
    }
    return at::_sparse_coo_tensor_with_dims_and_tensors(out.sparse_dim(), out.dense_dim(), out.sizes(), out._indices().clone(), result, out.options(), true);
}

at::Tensor sparse_softmax_backward_vulkan(const at::Tensor& grad_output, const at::Tensor& output, int64_t dim, const at::Tensor& /* self */)
{
    return sparse_softmax_backward_impl(grad_output, output, dim, false);
}

at::Tensor sparse_log_softmax_backward_vulkan(const at::Tensor& grad_output, const at::Tensor& output, int64_t dim, const at::Tensor& /* self */)
{
    return sparse_softmax_backward_impl(grad_output, output, dim, true);
}

// compressed layouts (CSR and friends) compute through COO, which every compressed layout converts to and from

// out (a compressed tensor) takes result's members
at::Tensor& set_compressed(at::Tensor& out, const at::Tensor& result)
{
    auto [compressed, plain] = at::sparse_csr::getCompressedPlainIndices(result);
    at::sparse_csr::get_sparse_csr_impl(out)->set_member_tensors(compressed, plain, result.values().to(out.scalar_type()), result.sizes());
    return out;
}

at::Tensor to_layout(const at::Tensor& coo, at::Layout layout)
{
    switch (layout) {
        case at::kSparseCsr: return coo.to_sparse_csr();
        case at::kSparseCsc: return coo.to_sparse_csc();
        default: TORCH_CHECK(false, "torchvulkan [ERROR]: no conversion from COO to ", layout, " for this op.");
    }
}

at::Tensor& add_out_sparse_compressed_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha, at::Tensor& out)
{
    if (out.layout() == at::kStrided) {
        // as the CPU: a compressed self with a dense other is other + alpha * self
        bool swap = self.layout() != at::kStrided && other.layout() == at::kStrided;
        const at::Tensor& dense = swap ? other : self;
        const at::Tensor& sparse = swap ? self : other;
        at::Tensor result = dense_plus_sparse(dense.layout() == at::kStrided ? dense : dense.to_dense(), sparse.to_sparse(), alpha, out.scalar_type());
        at::native::resize_output(out, result.sizes());
        return out.copy_(result);
    }
    TORCH_CHECK(self.sizes() == other.sizes(), "torch.add: Expected input tensors to have the same shape, but got tensor `self` with shape ",
        self.sizes(), " and tensor `other` with shape ", other.sizes());
    at::Tensor sum = at::add(self.to_sparse(), other.to_sparse(), alpha).coalesce();
    at::Tensor result = to_layout(sum, out.layout());
    // the inputs' index dtype (COO computes in int64)
    const at::Tensor& like = self.layout() != at::kStrided ? self : other;
    at::ScalarType index_dtype = at::sparse_csr::getCompressedPlainIndices(like).first.scalar_type();
    auto [compressed, plain] = at::sparse_csr::getCompressedPlainIndices(result);
    at::sparse_csr::get_sparse_csr_impl(out)->set_member_tensors(compressed.to(index_dtype), plain.to(index_dtype),
                                                                 result.values().to(out.scalar_type()), result.sizes());
    return out;
}

at::Tensor& addmm_out_sparse_compressed_vulkan(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2, const at::Scalar& beta,
                                               const at::Scalar& alpha, at::Tensor& out)
{
    for (const at::Tensor* t : {&mat1, &mat2, static_cast<const at::Tensor*>(&out)}) {
        TORCH_CHECK(t->device() == self.device(), "Expected all tensors to be on the same device, but found at least two devices, ", self.device(),
            " and ", t->device(), "!");
    }
    // the CPU's dense messages, which test_sparse_csr compares against
    TORCH_CHECK(mat1.dim() == 2, "addmm: self must be a matrix, got ", mat1.dim(), "-D tensor");
    TORCH_CHECK(mat2.dim() == 2, "addmm: mat2 must be a matrix, got ", mat2.dim(), "-D tensor");
    TORCH_CHECK(mat1.size(1) == mat2.size(0), "addmm: mat1 and mat2 shapes cannot be multiplied (", mat1.size(0), "x", mat1.size(1), " and ",
        mat2.size(0), "x", mat2.size(1), ")");
    at::Tensor dense_self = self.layout() == at::kStrided ? self : self.to_dense();
    if (out.layout() != at::kStrided || (mat1.layout() != at::kStrided && mat2.layout() != at::kStrided)) {
        // both operands compressed, or a compressed out: the product through mat1 as COO times a dense mat2, then into
        // out's layout (its index dtype and block size kept)
        at::Tensor dense2 = mat2.layout() == at::kStrided ? mat2 : mat2.to_dense();
        at::Tensor product = mat1.layout() == at::kStrided ? at::mm(mat1, dense2) : sparse_dense_mm(mat1.to_sparse().coalesce(), dense2);
        at::Tensor result = at::mul(product, alpha);
        if (beta.toComplexDouble() != c10::complex<double>(0.0)) result = at::add(result, at::mul(dense_self.expand(result.sizes()), beta));
        if (out.layout() == at::kStrided) {
            at::native::resize_output(out, result.sizes());
            return out.copy_(result);
        }
        std::optional<at::IntArrayRef> blocksize;
        std::vector<int64_t> block;
        if (out.layout() == at::kSparseBsr || out.layout() == at::kSparseBsc) {
            block = {out.values().size(-2), out.values().size(-1)};
            blocksize = at::IntArrayRef(block);
        }
        at::Tensor sparse = result.to(out.scalar_type()).to_sparse(out.layout(), blocksize);
        auto [compressed, plain] = at::sparse_csr::getCompressedPlainIndices(sparse);
        auto [out_compressed, out_plain] = at::sparse_csr::getCompressedPlainIndices(out);
        at::sparse_csr::get_sparse_csr_impl(out)->set_member_tensors(compressed.to(out_compressed.scalar_type()), plain.to(out_plain.scalar_type()),
                                                                     sparse.values(), sparse.sizes());
        return out;
    }
    at::Tensor result;
    if (mat1.layout() != at::kStrided) {
        result = addmm_sparse_vulkan(dense_self, mat1.to_sparse(), mat2, beta, alpha);
    } else {
        // dense @ sparse as (sparse^T @ dense^T)^T
        at::Tensor product = sparse_dense_mm(mat2.to_sparse().t().coalesce(), mat1.t()).t();
        result = at::mul(product, alpha);
        if (beta.toComplexDouble() != c10::complex<double>(0.0)) result = at::add(result, at::mul(dense_self.expand(result.sizes()), beta));
    }
    at::native::resize_output(out, result.sizes());
    return out.copy_(result);
}

at::Tensor& addmv_out_sparse_compressed_vulkan(const at::Tensor& self, const at::Tensor& mat, const at::Tensor& vec, const at::Scalar& beta,
                                               const at::Scalar& alpha, at::Tensor& out)
{
    at::Tensor product = at::mul(sparse_dense_mm(mat.to_sparse(), vec.unsqueeze(1)).squeeze(1), alpha);
    at::Tensor result = beta.toComplexDouble() == c10::complex<double>(0.0) ? product : at::add(at::mul(self.expand(product.sizes()), beta), product);
    at::native::resize_output(out, result.sizes());
    return out.copy_(result);
}

// beta * self + alpha * (mat1 @ mat2) at self's entries: the dense product sampled by sparse_mask
at::Tensor sparse_sampled_addmm_vulkan(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2, const at::Scalar& beta,
                                       const at::Scalar& alpha)
{
    TORCH_CHECK(at::isFloatingType(self.scalar_type()) || at::isComplexType(self.scalar_type()),
        "sparse_sampled_addmm: Expected self to be a floating-point or complex tensor, but got ", self.scalar_type());
    for (auto [name, t] : {std::pair<const char*, const at::Tensor&>{"mat1", mat1}, {"mat2", mat2}}) {
        TORCH_CHECK(t.layout() == at::kStrided, "sampled_addmm: Expected ", name, " to have strided layout, but got ", t.layout());
        TORCH_CHECK(t.dim() >= 2, "sampled_addmm: Expected ", name, " to be a matrix, got ", t.dim(), "-D tensor");
    }
    TORCH_CHECK(mat1.scalar_type() == mat2.scalar_type(), "sampled_addmm: Expected mat1 and mat2 to have the same dtype, but got ",
        mat1.scalar_type(), " and ", mat2.scalar_type());
    TORCH_CHECK(self.scalar_type() == mat1.scalar_type(), "sampled_addmm: Expected self and mat1 to have the same dtype, but got ",
        self.scalar_type(), " and ", mat1.scalar_type());
    TORCH_CHECK(mat1.sizes().slice(0, mat1.dim() - 2).equals(mat2.sizes().slice(0, mat2.dim() - 2)),
        "sampled_addmm: Expected mat1 and mat2 to have the same batch size, but got ", mat1.sizes().slice(0, mat1.dim() - 2), " and ",
        mat2.sizes().slice(0, mat2.dim() - 2));
    TORCH_CHECK(mat1.size(-1) == mat2.size(-2), "sampled_addmm: mat1 and mat2 shapes cannot be multiplied (", mat1.size(-2), "x", mat1.size(-1),
        " and ", mat2.size(-2), "x", mat2.size(-1), ")");
    TORCH_CHECK(self.size(-2) == mat1.size(-2), "sampled_addmm: self.shape[-2] must match mat1.shape[-2]");
    TORCH_CHECK(self.size(-1) == mat2.size(-1), "sampled_addmm: self.shape[-1] must match mat2.shape[-1]");
    // an unbatched self broadcasts over mat1's batch: its pattern repeated per batch
    at::Tensor c = self;
    std::vector<int64_t> batch(mat1.sizes().begin(), mat1.sizes().end() - 2);
    if (self.dim() == 2 && !batch.empty()) {
        auto with_batch = [&](const at::Tensor& t) {
            std::vector<int64_t> shape = batch;
            shape.push_back(t.size(-1));
            std::vector<int64_t> expanded(batch.size(), 1);
            expanded.push_back(t.size(-1));
            return t.reshape(expanded).expand(shape).contiguous();
        };
        std::vector<int64_t> size = batch;
        size.push_back(self.size(-2));
        size.push_back(self.size(-1));
        c = at::sparse_csr_tensor(with_batch(self.crow_indices()), with_batch(self.col_indices()), with_batch(self.values()), size, self.options());
    }
    at::Tensor sampled = at::matmul(mat1, mat2).sparse_mask(c).values();
    at::Tensor values = at::mul(sampled, alpha);
    if (beta.toComplexDouble() != c10::complex<double>(0.0)) values = at::add(values, at::mul(c.values(), beta));
    return at::sparse_csr_tensor(c.crow_indices(), c.col_indices(), values, c.sizes(), self.options());
}

// a batched compressed mat1 times dense mat2, one batch at a time
at::Tensor bmm_compressed_dense(const at::Tensor& mat1, const at::Tensor& mat2)
{
    TORCH_CHECK(mat1.dim() == 3 && mat2.dim() == 3, "bmm: expected 3-D tensors, got ", mat1.dim(), "-D and ", mat2.dim(), "-D");
    TORCH_CHECK(mat1.size(0) == mat2.size(0), "bmm: batch sizes differ: ", mat1.size(0), " and ", mat2.size(0));
    TORCH_CHECK(mat1.size(2) == mat2.size(1), "bmm: mat1 and mat2 shapes cannot be multiplied (", mat1.size(1), "x", mat1.size(2), " and ",
        mat2.size(1), "x", mat2.size(2), ")");
    at::Tensor dense2 = mat2.layout() == at::kStrided ? mat2 : mat2.to_dense();
    // each batch as its own 2-D compressed tensor, from the batched members (neither select nor a COO conversion
    // takes a batched compressed tensor)
    std::vector<at::Tensor> products;
    for (int64_t b = 0; b < mat1.size(0); b++) {
        if (mat1.layout() == at::kStrided) {
            products.push_back(at::mm(mat1.select(0, b), dense2.select(0, b)));
            continue;
        }
        auto [compressed, plain] = at::sparse_csr::getCompressedPlainIndices(mat1);
        // 1-D members under a batched shape (torch's bmm tests build these) share one matrix across the batch
        bool shared = compressed.dim() == 1;
        at::Tensor a = at::_sparse_compressed_tensor_unsafe(shared ? compressed : compressed.select(0, b), shared ? plain : plain.select(0, b),
                                                            shared ? mat1.values() : mat1.values().select(0, b), {mat1.size(1), mat1.size(2)},
                                                            mat1.options());
        products.push_back(sparse_dense_mm(a.to_sparse().coalesce(), dense2.select(0, b)));
    }
    return products.empty() ? at::zeros({0, mat1.size(1), mat2.size(2)}, dense2.options()) : at::stack(products);
}

// a compressed A through a dense copy; as the CPU's sparse kernel, the coefficient isn't cloned and an A without stored
// elements gives NaN
std::tuple<at::Tensor&, at::Tensor&> triangular_solve_out_sparse_compressed_vulkan(const at::Tensor& self, const at::Tensor& A, bool upper,
                                                                                  bool transpose, bool unitriangular, at::Tensor& X, at::Tensor& M)
{
    at::Tensor solution;
    if (A._nnz() == 0) solution = at::full(self.sizes(), std::numeric_limits<double>::quiet_NaN(), self.options());
    else solution = std::get<0>(at::triangular_solve(self, A.to_dense(), upper, transpose, unitriangular));
    at::native::resize_output(X, solution.sizes());
    X.copy_(solution);
    at::native::resize_output(M, {0});
    return std::forward_as_tuple(X, M);
}

at::Tensor& bmm_out_sparse_compressed_vulkan(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out)
{
    at::Tensor result = bmm_compressed_dense(self, mat2);
    at::native::resize_output(out, result.sizes());
    return out.copy_(result);
}

at::Tensor& baddbmm_out_sparse_compressed_vulkan(const at::Tensor& self, const at::Tensor& batch1, const at::Tensor& batch2, const at::Scalar& beta,
                                                 const at::Scalar& alpha, at::Tensor& out)
{
    at::Tensor result = at::mul(bmm_compressed_dense(batch1, batch2), alpha);
    if (beta.toComplexDouble() != c10::complex<double>(0.0)) {
        at::Tensor dense_self = self.layout() == at::kStrided ? self : self.to_dense();
        result = at::add(result, at::mul(dense_self.expand(result.sizes()), beta));
    }
    at::native::resize_output(out, result.sizes());
    return out.copy_(result);
}

at::Tensor& sparse_sampled_addmm_out_vulkan(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2, const at::Scalar& beta,
                                            const at::Scalar& alpha, at::Tensor& out)
{
    at::Tensor result = sparse_sampled_addmm_vulkan(self, mat1, mat2, beta, alpha);
    out.resize_as_sparse_(result);
    return out.copy_(result);
}

// torch's empty leaves the compressed indices uninitialized (its own comment: the result "may fail" the invariants), so with
// invariant checks on, a reused allocation fails them; zero them for CSR/CSC (blocked layouts keep torch's)
at::Tensor empty_sparse_compressed_vulkan(at::IntArrayRef size, std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout,
                                          std::optional<at::Device> device, std::optional<bool> pin_memory,
                                          std::optional<at::MemoryFormat> memory_format)
{
    at::Layout l = layout.value_or(at::kSparseCsr);
    if (l != at::kSparseCsr && l != at::kSparseCsc) {
        return at::native::empty_sparse_compressed(size, dtype, layout, device, pin_memory, memory_format);
    }
    TORCH_CHECK(size.size() >= 2, "torch.empty: Only batched sparse compressed (non-block) tensors are supported, but got size ", size);
    std::vector<int64_t> compressed_size(size.begin(), size.end() - 2), plain_size(size.begin(), size.end() - 2);
    compressed_size.push_back(size[size.size() - (l == at::kSparseCsr ? 2 : 1)] + 1);
    plain_size.push_back(0);
    at::TensorOptions options = at::TensorOptions().dtype(at::kLong).device(device);
    return at::_sparse_compressed_tensor_unsafe(at::zeros(compressed_size, options), at::empty(plain_size, options),
        at::empty(plain_size, options.dtype(dtype.value_or(c10::typeMetaToScalarType(at::get_default_dtype())))), size,
        dtype, l, device, pin_memory);
}

// torch's kernel runs a CPU/CUDA stub: repeat each entry along every broadcast sparse dimension, expand the values
// along broadcast dense ones
at::Tensor sparse_broadcast_to_vulkan(const at::Tensor& self, at::IntArrayRef size)
{
    int64_t extra = static_cast<int64_t>(size.size()) - self.dim();
    TORCH_CHECK(extra >= 0, "sparse_broadcast_to: the requested size has fewer dimensions than the input");
    int64_t sparse_dims = extra + self.sparse_dim();
    std::vector<int64_t> current(extra, 1);
    current.insert(current.end(), self.sizes().begin(), self.sizes().end());
    for (size_t i = 0; i < size.size(); i++) {
        TORCH_CHECK(current[i] == size[i] || current[i] == 1, "The expanded size of the tensor (", size[i], ") must match the existing size (",
            current[i], ") at non-singleton dimension ", i);
    }
    at::Tensor indices = self._indices(), values = self._values();
    if (extra > 0) indices = at::cat({at::zeros({extra, indices.size(1)}, indices.options()), indices}, 0);
    bool repeated = false;
    for (int64_t i = 0; i < sparse_dims; i++) {
        if (current[i] == size[i]) continue;
        int64_t n = size[i], nnz = indices.size(1);
        indices = indices.unsqueeze(2).expand({sparse_dims, nnz, n}).clone();
        indices.select(0, i).copy_(at::arange(n, indices.options()).expand({nnz, n}));
        indices = indices.reshape({sparse_dims, nnz * n});
        std::vector<int64_t> spread = values.sizes().vec();
        spread.insert(spread.begin() + 1, n);
        values = values.unsqueeze(1).expand(spread).reshape([&] { std::vector<int64_t> v = spread; v[0] *= n; v.erase(v.begin() + 1); return v; }());
        repeated = true;
    }
    std::vector<int64_t> value_size = {indices.size(1)};
    value_size.insert(value_size.end(), size.begin() + sparse_dims, size.end());
    values = values.expand(value_size).contiguous();
    at::Tensor result = at::_sparse_coo_tensor_unsafe(indices, values, size, self.options());
    return result._coalesced_(!repeated && self.is_coalesced());
}

// torch's CSR sum/prod: entries only where the reduced slice stores something, and prod multiplies just the stored values
at::Tensor sparse_csr_reduce(const at::Tensor& self, at::IntArrayRef dims, bool keepdim, std::optional<at::ScalarType> dtype, bool prod)
{
    TORCH_CHECK(keepdim, "reduction operations on CSR tensors with keepdim=False is unsupported");
    TORCH_CHECK(self.dim() == 2, "reduction operations on CSR tensors support only 2-D tensors");
    at::ScalarType out_dtype = dtype.value_or(at::isIntegralType(self.scalar_type(), /*includeBool=*/true) ? at::kLong : self.scalar_type());
    at::Tensor coo = self.to_sparse();
    at::Tensor indices = coo._indices().clone();
    at::Tensor values = coo._values().to(out_dtype);
    std::vector<int64_t> shape = self.sizes().vec();
    for (int64_t d : dims) {
        d = c10::maybe_wrap_dim(d, 2);
        indices.select(0, d).zero_();
        shape[d] = 1;
    }
    at::Tensor keys = at::add(at::mul(indices.select(0, 0), shape[1]), indices.select(0, 1));
    auto [unique_keys, inverse, counts] = at::_unique2(keys, /*sorted=*/true, /*return_inverse=*/true, /*return_counts=*/false);
    at::Tensor reduced = (prod ? at::ones({unique_keys.size(0)}, values.options()) : at::zeros({unique_keys.size(0)}, values.options()))
        .scatter_reduce_(0, inverse, values, prod ? "prod" : "sum", /*include_self=*/false);
    at::Tensor out_indices = at::stack({at::div(unique_keys, shape[1], "floor"), at::remainder(unique_keys, shape[1])});
    return at::_sparse_coo_tensor_unsafe(out_indices, reduced, shape, values.options().layout(at::kSparse))._coalesced_(true).to_sparse_csr();
}

at::Tensor sparse_csr_sum_vulkan(const at::Tensor& self, at::IntArrayRef dims, bool keepdim, std::optional<at::ScalarType> dtype)
{
    return sparse_csr_reduce(self, dims, keepdim, dtype, /*prod=*/false);
}

at::Tensor sparse_csr_prod_vulkan(const at::Tensor& self, at::IntArrayRef dims, bool keepdim, std::optional<at::ScalarType> dtype)
{
    return sparse_csr_reduce(self, dims, keepdim, dtype, /*prod=*/true);
}

// sorted COO row indices to CSR's compressed ones: how many entries come before each row
at::Tensor& convert_indices_from_coo_to_csr_out_vulkan(const at::Tensor& self, int64_t size, bool out_int32, at::Tensor& out)
{
    at::Tensor boundaries = at::arange(size + 1, self.options().dtype(self.scalar_type()));
    at::Tensor crow = at::searchsorted(self.contiguous(), boundaries, out_int32, /*right=*/false);
    at::native::resize_output(out, {size + 1});
    return out.copy_(crow);
}

// CSR's compressed indices to COO rows (each entry's row is the last boundary at or before it), with batch indices first
at::Tensor& convert_indices_from_csr_to_coo_out_vulkan(const at::Tensor& crow_indices, const at::Tensor& col_indices, bool out_int32, bool transpose,
                                                      at::Tensor& out)
{
    at::ScalarType dtype = out_int32 ? at::kInt : at::kLong;
    int64_t batch_ndim = crow_indices.dim() - 1;
    int64_t nnz = col_indices.size(-1);
    int64_t rows = crow_indices.size(-1) - 1;
    at::Tensor crow = crow_indices.reshape({-1, rows + 1}).to(at::kLong).contiguous();
    at::Tensor positions = at::arange(nnz, crow.options()).expand({crow.size(0), nnz}).contiguous();
    at::Tensor row = at::sub(at::searchsorted(crow, positions, /*out_int32=*/false, /*right=*/true), 1).reshape({-1});
    at::Tensor col = col_indices.reshape({-1}).to(at::kLong);
    std::vector<at::Tensor> parts;
    if (batch_ndim > 0) {
        at::Tensor batch = at::repeat_interleave(at::arange(crow.size(0), crow.options()), nnz);
        std::vector<int64_t> batch_sizes(crow_indices.sizes().begin(), crow_indices.sizes().end() - 1);
        for (int64_t d = batch_ndim - 1; d >= 0; d--) {
            parts.insert(parts.begin(), at::remainder(batch, batch_sizes[d]));
            batch = at::div(batch, batch_sizes[d], "floor");
        }
    }
    parts.push_back(transpose ? col : row);
    parts.push_back(transpose ? row : col);
    at::Tensor result = at::stack(parts).to(dtype);
    at::native::resize_output(out, result.sizes());
    return out.copy_(result);
}

// the compressed-index invariants torch checks, in its order and words
void validate_compressed_sparse_indices_vulkan(bool is_crow, const at::Tensor& compressed, const at::Tensor& plain, int64_t cdim, int64_t dim, int64_t nnz)
{
    const char* c = is_crow ? "crow_indices" : "ccol_indices";
    const char* p = is_crow ? "col_indices" : "row_indices";
    const char* cn = is_crow ? "nrows" : "ncols";
    const char* pn = is_crow ? "ncols" : "nrows";
    // the device check a dispatched kernel makes first (torch validates indices before comparing their devices)
    TORCH_CHECK(compressed.device() == plain.device(), "Expected all tensors to be on the same device, but found at least two devices, ",
        compressed.device(), " and ", plain.device(), "!");
    TORCH_CHECK(compressed.size(-1) == cdim + 1, c, " have wrong shape: ", c, ".shape[-1] = ", compressed.size(-1), " is not equal to ", cn, " + 1 = ", cdim + 1);
    TORCH_CHECK(plain.size(-1) == nnz, p, " have wrong shape: ", p, ".shape[-1] = ", plain.size(-1), " is not equal to nnz = ", nnz);
    at::Tensor cidx = compressed.to(at::kLong), idx = plain.to(at::kLong);
    auto holds = [](const at::Tensor& condition) { return condition.numel() == 0 || condition.all().item<bool>(); };
    TORCH_CHECK(holds(at::logical_and(at::ge(idx, 0), at::lt(idx, dim))), "`0 <= ", p, " < ", pn, "` is not satisfied.");
    TORCH_CHECK(holds(at::eq(cidx.select(-1, 0), 0)), "`", c, "[..., 0] == 0` is not satisfied.");
    TORCH_CHECK(holds(at::eq(cidx.select(-1, -1), nnz)), "`", c, "[..., -1] == nnz` is not satisfied.");
    at::Tensor steps = at::sub(cidx.narrow(-1, 1, cdim), cidx.narrow(-1, 0, cdim));
    TORCH_CHECK(holds(at::logical_and(at::ge(steps, 0), at::le(steps, dim))),
                "`0 <= ", c, "[..., 1:] - ", c, "[..., :-1] <= ", pn, "` is not satisfied.");
    if (nnz > 1) {
        // neighbours in the same slice must increase
        at::Tensor positions = at::arange(nnz, idx.options()).expand(idx.sizes()).contiguous();
        at::Tensor slice = at::searchsorted(cidx.contiguous(), positions, /*out_int32=*/false, /*right=*/true);
        at::Tensor same = at::eq(slice.narrow(-1, 0, nnz - 1), slice.narrow(-1, 1, nnz - 1));
        at::Tensor increasing = at::lt(idx.narrow(-1, 0, nnz - 1), idx.narrow(-1, 1, nnz - 1));
        TORCH_CHECK(holds(at::logical_or(at::logical_not(same), increasing)), "`", p, "[..., ", c, "[..., i - 1]:", c, "[..., i]] for all i = 1, ..., ", cn,
                    " are sorted and distinct along the last dimension values` is not satisfied.");
    }
}

} // namespace

// at::sparse::flatten_indices' stub, as declared in ATen/native/sparse/SparseStubs.h (which the wheel doesn't ship); the
// CSR <-> CSC conversions hash indices through it
namespace at::native {
using flatten_indices_fn = Tensor (*)(const Tensor& indices, IntArrayRef size);
DECLARE_DISPATCH(flatten_indices_fn, flatten_indices_stub)

namespace {
Tensor flatten_indices_vulkan(const Tensor& indices, IntArrayRef size)
{
    return linear_index(indices.to(kLong), size);
}
} // namespace

REGISTER_PRIVATEUSE1_DISPATCH(flatten_indices_stub, &flatten_indices_vulkan)
} // namespace at::native

TORCH_LIBRARY_IMPL(aten, SparseCsrPrivateUse1, m) {
    m.impl("add.out", &add_out_sparse_compressed_vulkan);
    m.impl("addmm.out", &addmm_out_sparse_compressed_vulkan);
    m.impl("addmv.out", &addmv_out_sparse_compressed_vulkan);
    m.impl("bmm.out", &bmm_out_sparse_compressed_vulkan);
    m.impl("triangular_solve.X", &triangular_solve_out_sparse_compressed_vulkan);
    m.impl("baddbmm.out", &baddbmm_out_sparse_compressed_vulkan);
    m.impl("sparse_sampled_addmm", &sparse_sampled_addmm_vulkan);
    m.impl("empty.memory_format", &empty_sparse_compressed_vulkan);
    m.impl("_sparse_csr_sum.dim_dtype", &sparse_csr_sum_vulkan);
    m.impl("_sparse_csr_prod.dim_dtype", &sparse_csr_prod_vulkan);
    m.impl("sparse_sampled_addmm.out", &sparse_sampled_addmm_out_vulkan);
}

// dense to sparse conversions, shared by CPU and CUDA
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("_to_sparse.sparse_dim", static_cast<at::Tensor (*)(const at::Tensor&, int64_t)>(&at::native::dense_to_sparse));
    m.impl("_to_sparse", static_cast<at::Tensor (*)(const at::Tensor&, std::optional<at::Layout>, at::OptionalIntArrayRef, std::optional<int64_t>)>(
        &at::native::dense_to_sparse));
    m.impl("_to_sparse_csr", TORCH_FN(at::native::dense_to_sparse_csr));
    m.impl("_to_sparse_csc", TORCH_FN(at::native::dense_to_sparse_csc));
    m.impl("_to_sparse_bsr", TORCH_FN(at::native::dense_to_sparse_bsr));
    m.impl("_to_sparse_bsc", TORCH_FN(at::native::dense_to_sparse_bsc));
    m.impl("_validate_compressed_sparse_indices", &validate_compressed_sparse_indices_vulkan);
    m.impl("_convert_indices_from_coo_to_csr.out", &convert_indices_from_coo_to_csr_out_vulkan);
    m.impl("_convert_indices_from_csr_to_coo.out", &convert_indices_from_csr_to_coo_out_vulkan);
}

TORCH_LIBRARY_IMPL(aten, SparsePrivateUse1, m) {
    m.impl("_coalesce", &coalesce_sparse_vulkan);
    m.impl("add.out", &add_out_sparse_vulkan);
    m.impl("mul.out", &mul_out_sparse_vulkan);
    m.impl("addmm", &addmm_sparse_vulkan);
    m.impl("addmm.out", &addmm_out_sparse_vulkan);
    m.impl("addmm_", &addmm_sparse_vulkan_);
    m.impl("bmm", &bmm_sparse_vulkan);
    m.impl("bmm.out", &bmm_out_sparse_vulkan);
    m.impl("index_select", &index_select_sparse_vulkan);
    m.impl("_sparse_sparse_matmul", &sparse_sparse_matmul_vulkan);
    m.impl("_sparse_mask_projection", &sparse_mask_projection_vulkan);
    m.impl("_sparse_broadcast_to", &sparse_broadcast_to_vulkan);
    m.impl("_sparse_mask_projection.out", &sparse_mask_projection_out_vulkan);
    m.impl("_sparse_softmax", &sparse_softmax_vulkan);
    m.impl("_sparse_log_softmax", &sparse_log_softmax_vulkan);
    m.impl("_sparse_softmax_backward_data", &sparse_softmax_backward_vulkan);
    m.impl("_sparse_log_softmax_backward_data", &sparse_log_softmax_backward_vulkan);
}
