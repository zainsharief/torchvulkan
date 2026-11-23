#include <torch/extension.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/native/nested/NestedTensorMath.h>
#include <ATen/native/nested/NestedTensorUtils.h>
#include <ATen/ops/_nested_compute_contiguous_strides_offsets_native.h>
#include <ATen/ops/_nested_view_from_buffer_native.h>
#include <ATen/ops/_nested_from_padded_native.h>
#include <ATen/ops/_nested_tensor_from_mask_left_aligned_native.h>
#include <ATen/ops/_nested_tensor_from_mask_native.h>
#include <ATen/ops/_nested_from_padded_and_nested_example_native.h>
#include <ATen/ops/_nested_select_backward_native.h>
#include <ATen/ops/_nested_tensor_size_native.h>
#include <ATen/ops/_nested_tensor_storage_offsets_native.h>
#include <ATen/ops/_nested_tensor_strides_native.h>
#include <ATen/ops/_safe_softmax_native.h>
#include <ATen/ops/_softmax_backward_data_native.h>
#include <ATen/ops/_softmax_native.h>
#include <ATen/ops/_test_autograd_multiple_dispatch_native.h>
#include <ATen/ops/_to_copy_native.h>
#include <ATen/ops/_transformer_encoder_layer_fwd_native.h>
#include <ATen/ops/abs_native.h>
#include <ATen/ops/add_native.h>
#include <ATen/ops/alias_native.h>
#include <ATen/ops/all_native.h>
#include <ATen/ops/cat_native.h>
#include <ATen/ops/chunk_native.h>
#include <ATen/ops/clone_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/cos_native.h>
#include <ATen/ops/detach_native.h>
#include <ATen/ops/div_native.h>
#include <ATen/ops/embedding_native.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/eq_native.h>
#include <ATen/ops/fill_native.h>
#include <ATen/ops/ge_native.h>
#include <ATen/ops/gelu_backward_native.h>
#include <ATen/ops/gelu_native.h>
#include <ATen/ops/gt_native.h>
#include <ATen/ops/is_same_size_native.h>
#include <ATen/ops/isinf_native.h>
#include <ATen/ops/isnan_native.h>
#include <ATen/ops/isneginf_native.h>
#include <ATen/ops/isposinf_native.h>
#include <ATen/ops/linear_backward_native.h>
#include <ATen/ops/linear_native.h>
#include <ATen/ops/logical_not_native.h>
#include <ATen/ops/masked_fill_native.h>
#include <ATen/ops/matmul_backward_native.h>
#include <ATen/ops/matmul_native.h>
#include <ATen/ops/mul_native.h>
#include <ATen/ops/narrow_native.h>
#include <ATen/ops/native_dropout_backward_native.h>
#include <ATen/ops/native_dropout_native.h>
#include <ATen/ops/native_layer_norm_backward_native.h>
#include <ATen/ops/native_layer_norm_native.h>
#include <ATen/ops/neg_native.h>
#include <ATen/ops/normal_native.h>
#include <ATen/ops/ones_like_native.h>
#include <ATen/ops/relu_native.h>
#include <ATen/ops/select_native.h>
#include <ATen/ops/sgn_native.h>
#include <ATen/ops/silu_backward_native.h>
#include <ATen/ops/silu_native.h>
#include <ATen/ops/sin_native.h>
#include <ATen/ops/split_with_sizes_native.h>
#include <ATen/ops/sqrt_native.h>
#include <ATen/ops/squeeze_native.h>
#include <ATen/ops/sub_native.h>
#include <ATen/ops/tanh_native.h>
#include <ATen/ops/threshold_backward_native.h>
#include <ATen/ops/to_padded_tensor_native.h>
#include <ATen/ops/transpose_native.h>
#include <ATen/ops/unbind_native.h>
#include <ATen/ops/unsqueeze_native.h>
#include <ATen/ops/value_selecting_reduction_backward_native.h>
#include <ATen/ops/values_native.h>
#include <ATen/ops/view_native.h>
#include <ATen/ops/where_native.h>
#include <ATen/ops/zero_native.h>


namespace {

// each row of a jagged buffer: its batch and its position in that batch (one jagged dim, as the CPU kernels support)
std::tuple<at::Tensor, at::Tensor> jagged_rows(const at::Tensor& offsets, int64_t rows)
{
    at::Tensor o = offsets.to(at::kLong).contiguous();
    at::Tensor row = at::arange(rows, o.options());
    at::Tensor batch = at::sub(at::searchsorted(o, row, /*out_int32=*/false, /*right=*/true), 1);
    return std::make_tuple(batch, at::sub(row, at::index_select(o, 0, batch)));
}

at::Tensor jagged_to_padded_dense_forward_vulkan(const at::Tensor& values, at::TensorList offsets, c10::SymIntArrayRef max_lengths, double padding_value)
{
    TORCH_CHECK(offsets.size() == 1, "_jagged_to_padded_dense_forward(): only a single jagged dim is supported for now");
    int64_t batches = offsets[0].size(0) - 1, length = max_lengths[0].guard_int(__FILE__, __LINE__);
    std::vector<int64_t> padded_shape = {batches * length + 1};
    padded_shape.insert(padded_shape.end(), values.sizes().begin() + 1, values.sizes().end());
    // amax/amin pad with -+inf: an integer dtype takes its lowest/highest value instead
    at::Scalar fill = padding_value;
    if (at::isIntegralType(values.scalar_type(), /*includeBool=*/false)) {
        double lo = 0.0, hi = 0.0;
        switch (values.scalar_type()) {
            case at::kByte: lo = 0; hi = 255; break;
            case at::kChar: lo = -128; hi = 127; break;
            case at::kShort: lo = -32768; hi = 32767; break;
            case at::kInt: lo = std::numeric_limits<int32_t>::lowest(); hi = std::numeric_limits<int32_t>::max(); break;
            default: lo = static_cast<double>(std::numeric_limits<int64_t>::lowest()); hi = static_cast<double>(std::numeric_limits<int64_t>::max()); break;
        }
        if (std::isnan(padding_value)) fill = at::Scalar(int64_t(0));
        else if (padding_value <= lo) fill = values.scalar_type() == at::kLong ? at::Scalar(std::numeric_limits<int64_t>::lowest()) : at::Scalar(static_cast<int64_t>(lo));
        else if (padding_value >= hi) fill = values.scalar_type() == at::kLong ? at::Scalar(std::numeric_limits<int64_t>::max()) : at::Scalar(static_cast<int64_t>(hi));
        else fill = at::Scalar(static_cast<int64_t>(padding_value));
    }
    at::Tensor padded = at::full(padded_shape, fill, values.options());
    if (values.size(0) > 0) {
        auto [batch, position] = jagged_rows(offsets[0], values.size(0));
        // rows past max_length land in the extra last row, which is dropped
        at::Tensor target = at::where(at::lt(position, length), at::add(at::mul(batch, length), position), at::full_like(batch, batches * length));
        padded.index_copy_(0, target, values);
    }
    std::vector<int64_t> shape = {batches, length};
    shape.insert(shape.end(), values.sizes().begin() + 1, values.sizes().end());
    return padded.narrow(0, 0, batches * length).reshape(shape);
}

at::Tensor padded_dense_to_jagged_forward_vulkan(const at::Tensor& dense, at::TensorList offsets, c10::optional<c10::SymInt> total_L)
{
    TORCH_CHECK(offsets.size() == 1, "_padded_dense_to_jagged_forward(): only a single jagged dim is supported for now");
    int64_t batches = dense.size(0), length = dense.size(1);
    int64_t rows = total_L.has_value() ? total_L->guard_int(__FILE__, __LINE__) : offsets[0][-1].item<int64_t>();
    std::vector<int64_t> flat_shape = {batches * length};
    flat_shape.insert(flat_shape.end(), dense.sizes().begin() + 2, dense.sizes().end());
    std::vector<int64_t> shape = {rows};
    shape.insert(shape.end(), dense.sizes().begin() + 2, dense.sizes().end());
    if (rows == 0) return at::empty(shape, dense.options());
    at::Tensor o = offsets[0].to(at::kLong);
    int64_t longest = batches > 0 ? at::sub(o.narrow(0, 1, batches), o.narrow(0, 0, batches)).max().item<int64_t>() : 0;
    TORCH_CHECK(longest <= length, "_padded_dense_to_jagged_forward(): found batch item of length ", longest,
                " when max length specified by padded input is ", length);
    auto [batch, position] = jagged_rows(offsets[0], rows);
    at::Tensor inside = at::lt(position, length);
    at::Tensor values = at::index_select(dense.reshape(flat_shape), 0, at::where(inside, at::add(at::mul(batch, length), position), at::zeros_like(batch)));
    std::vector<int64_t> mask_shape(values.dim(), 1);
    mask_shape[0] = rows;
    return at::where(inside.view(mask_shape), values, at::zeros_like(values));
}

// one product per component (torch's nested bmm kernel for the CPU reads host memory)
at::Tensor bmm_nested_vulkan(const at::Tensor& self, const at::Tensor& mat2)
{
    TORCH_CHECK(self.dim() == 3 && mat2.dim() == 3, "batch1 must be a 3D tensor, batch2 must be a 3D tensor");
    std::vector<at::Tensor> a = self.unbind(0), b = mat2.unbind(0);
    TORCH_CHECK(a.size() == b.size(), "Expected size for first two dimensions of batch2 tensor to be: [", a.size(), ", ...] but got: [", b.size(), ", ...].");
    std::vector<at::Tensor> products;
    for (size_t i = 0; i < a.size(); i++) products.push_back(at::mm(a[i], b[i]));
    return at::_nested_tensor_from_tensor_list(products);
}

// the components share the normalized trailing dims, so normalizing the contiguous buffer as [N, M] rows is the
// nested layer norm; torch's own nested kernels go through a CPU/CUDA-only DispatchStub
int64_t normalized_numel(c10::SymIntArrayRef normalized_shape)
{
    int64_t m = 1;
    for (const c10::SymInt& s : normalized_shape) m *= s.guard_int(__FILE__, __LINE__);
    return m;
}

c10::optional<at::Tensor> flat_param(const c10::optional<at::Tensor>& t, int64_t m)
{
    return t.has_value() && t->defined() ? c10::optional<at::Tensor>(t->reshape({m})) : c10::nullopt;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> nested_layer_norm_vulkan(
    const at::Tensor& input,
    c10::SymIntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    double eps)
{
    at::native::_check_nested_layer_norm_inputs(*at::native::get_nested_tensor_impl(input), C10_AS_INTARRAYREF_SLOW(normalized_shape), weight.value_or(at::Tensor()), bias.value_or(at::Tensor()));
    TORCH_CHECK(at::native::nested_tensor_impl_is_contiguous(at::native::get_nested_tensor_impl(input)), "NestedTensor must be contiguous to get buffer.");
    int64_t m = normalized_numel(normalized_shape);
    at::Tensor buffer = at::native::get_buffer(input);
    int64_t rows = m == 0 ? 0 : buffer.numel() / m;
    auto [out, mean, rstd] = at::native_layer_norm(buffer.view({rows, m}), {m}, flat_param(weight, m), flat_param(bias, m), eps);
    at::Tensor sizes = at::native::get_nested_tensor_impl(input)->get_nested_sizes().clone();
    return std::make_tuple(at::native::wrap_buffer(out.reshape({-1}), sizes), mean, rstd);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> nested_layer_norm_backward_vulkan(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    c10::SymIntArrayRef normalized_shape,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    std::array<bool, 3> output_mask)
{
    int64_t m = normalized_numel(normalized_shape);
    at::Tensor x = at::native::get_buffer(input);
    int64_t rows = m == 0 ? 0 : x.numel() / m;
    at::Tensor g = at::native::get_buffer(grad_out.contiguous()).view({rows, m});
    auto [grad_input, grad_weight, grad_bias] = at::native_layer_norm_backward(g, x.view({rows, m}), {m}, mean, rstd, flat_param(weight, m), flat_param(bias, m), output_mask);
    at::Tensor sizes = at::native::get_nested_tensor_impl(input)->get_nested_sizes().clone();
    if (grad_input.defined()) grad_input = at::native::wrap_buffer(grad_input.reshape({-1}), sizes);
    if (grad_weight.defined() && weight.has_value() && weight->defined()) grad_weight = grad_weight.reshape(weight->sizes());
    if (grad_bias.defined() && bias.has_value() && bias->defined()) grad_bias = grad_bias.reshape(bias->sizes());
    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, NestedTensorPrivateUse1, m) {
    m.impl("bmm", &bmm_nested_vulkan);
}

// the dense-key helpers that build nested tensors, shared by CPU and CUDA (or written generically)
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("_jagged_to_padded_dense_forward", &jagged_to_padded_dense_forward_vulkan);
    m.impl("_padded_dense_to_jagged_forward", &padded_dense_to_jagged_forward_vulkan);
    m.impl("_nested_tensor_from_mask", TORCH_FN(at::native::NestedTensor_nested_tensor_from_mask));
    m.impl("_nested_tensor_from_mask_left_aligned", TORCH_FN(at::native::NestedTensor_nested_tensor_from_mask_left_aligned));
    m.impl("_nested_from_padded", TORCH_FN(at::native::nested_from_padded_generic));
    m.impl("_nested_view_from_buffer", TORCH_FN(at::native::_nested_view_from_buffer));
    m.impl("_nested_compute_contiguous_strides_offsets", TORCH_FN(at::native::_nested_compute_contiguous_strides_offsets));
}

// Nested tensors: torch's nested kernels that CPU and CUDA share are device-generic (they work on the nested
// tensor's buffer and metadata through other ops), so they serve the vulkan nested key as they are. The CPU-only ones
// (bmm, sum, softmax-with-shape, transform_bias_rescale_qkv, multi-head attention) read host memory and are left out.
TORCH_LIBRARY_IMPL(aten, NestedTensorPrivateUse1, m) {
    m.impl("native_dropout", TORCH_FN(at::native::native_dropout_nested));
    m.impl("native_dropout_backward", TORCH_FN(at::native::native_dropout_backward));
    m.impl("abs", TORCH_FN(at::native::NestedTensor_abs));
    m.impl("abs_", TORCH_FN(at::native::NestedTensor_abs_));
    m.impl("sgn", TORCH_FN(at::native::NestedTensor_sgn));
    m.impl("sgn_", TORCH_FN(at::native::NestedTensor_sgn_));
    m.impl("add.Tensor", TORCH_FN(at::native::NestedTensor_add_Tensor));
    m.impl("add_.Tensor", TORCH_FN(at::native::NestedTensor_add__Tensor));
    m.impl("all.dim", TORCH_FN(at::native::NestedTensor_all));
    m.impl("logical_not", TORCH_FN(at::native::NestedTensor_logical_not));
    m.impl("logical_not_", TORCH_FN(at::native::NestedTensor_logical_not_));
    m.impl("cat", TORCH_FN(at::native::cat_nested));
    m.impl("chunk", TORCH_FN(at::native::chunk_nested_tensor));
    m.impl("copy_", TORCH_FN(at::native::copy_nested_));
    m.impl("cos", TORCH_FN(at::native::NestedTensor_cos));
    m.impl("div.Tensor", TORCH_FN(at::native::NestedTensor_div_Tensor));
    m.impl("div.Scalar", TORCH_FN(at::native::NestedTensor_div_Scalar));
    m.impl("embedding", TORCH_FN(at::native::NestedTensor_embedding));
    m.impl("empty_like", TORCH_FN(at::native::empty_like_nested));
    m.impl("fill_.Scalar", static_cast<at::Tensor & (*)(at::Tensor &, const at::Scalar &)>(&at::native::fill_nested_));
    m.impl("fill_.Tensor", static_cast<at::Tensor & (*)(at::Tensor &, const at::Tensor &)>(&at::native::fill_nested_));
    m.impl("isnan", TORCH_FN(at::native::NestedTensor_isnan));
    m.impl("is_same_size", TORCH_FN(at::native::nested_is_same_size));
    m.impl("native_layer_norm", &nested_layer_norm_vulkan);
    m.impl("native_layer_norm_backward", &nested_layer_norm_backward_vulkan);
    m.impl("linear", TORCH_FN(at::native::nested_linear));
    m.impl("linear_backward", TORCH_FN(at::native::nested_linear_backward));
    m.impl("matmul", TORCH_FN(at::native::matmul_nested));
    m.impl("matmul_backward", TORCH_FN(at::native::matmul_backward_nested));
    m.impl("matmul.out", TORCH_FN(at::native::matmul_out_nested));
    m.impl("value_selecting_reduction_backward", TORCH_FN(at::native::value_selecting_reduction_backward_nested_symint));
    m.impl("mul.Tensor", TORCH_FN(at::native::NestedTensor_mul_Tensor));
    m.impl("mul_.Tensor", TORCH_FN(at::native::NestedTensor_mul__Tensor));
    m.impl("mul.Scalar", TORCH_FN(at::native::NestedTensor_mul_Scalar));
    m.impl("mul_.Scalar", TORCH_FN(at::native::NestedTensor_mul__Scalar));
    m.impl("narrow", TORCH_FN(at::native::narrow_nested_symint));
    m.impl("ones_like", TORCH_FN(at::native::ones_like));
    m.impl("neg", TORCH_FN(at::native::NestedTensor_neg));
    m.impl("neg_", TORCH_FN(at::native::NestedTensor_neg_));
    m.impl("relu", TORCH_FN(at::native::NestedTensor_relu));
    m.impl("relu_", TORCH_FN(at::native::NestedTensor_relu_));
    m.impl("gelu_", TORCH_FN(at::native::NestedTensor_gelu_));
    m.impl("gelu", TORCH_FN(at::native::NestedTensor_gelu));
    m.impl("gelu_backward", TORCH_FN(at::native::gelu_backwards_nested));
    m.impl("select.int", TORCH_FN(at::native::select_nested));
    m.impl("_nested_select_backward", TORCH_FN(at::native::_nested_select_backward_symint));
    m.impl("silu", TORCH_FN(at::native::NestedTensor_silu));
    m.impl("silu_", TORCH_FN(at::native::NestedTensor_silu_));
    m.impl("silu_backward", TORCH_FN(at::native::silu_backward_nested));
    m.impl("sin", TORCH_FN(at::native::NestedTensor_sin));
    m.impl("detach", TORCH_FN(at::native::detach));
    m.impl("_softmax", TORCH_FN(at::native::softmax_nested));
    m.impl("_softmax_backward_data", TORCH_FN(at::native::nested_softmax_backward));
    m.impl("split_with_sizes", TORCH_FN(at::native::split_with_sizes_nested));
    m.impl("squeeze", TORCH_FN(at::native::squeeze_nested));
    m.impl("squeeze.dim", static_cast<at::Tensor (*)(const at::Tensor &, int64_t)>(&at::native::squeeze_dim_nested));
    m.impl("squeeze.dims", static_cast<at::Tensor (*)(const at::Tensor &, at::IntArrayRef)>(&at::native::squeeze_dim_nested));
    m.impl("sqrt", TORCH_FN(at::native::NestedTensor_sqrt));
    m.impl("tanh", TORCH_FN(at::native::NestedTensor_tanh));
    m.impl("tanh_", TORCH_FN(at::native::NestedTensor_tanh_));
    m.impl("threshold_backward", TORCH_FN(at::native::threshold_backwards_nested));
    m.impl("transpose.int", TORCH_FN(at::native::transpose_nested));
    m.impl("_nested_tensor_size", TORCH_FN(at::native::_nested_tensor_size));
    m.impl("_nested_tensor_strides", TORCH_FN(at::native::_nested_tensor_strides));
    m.impl("_nested_tensor_storage_offsets", TORCH_FN(at::native::_nested_tensor_storage_offsets));
    m.impl("_nested_from_padded_and_nested_example", TORCH_FN(at::native::NestedTensor_from_padded_and_nested_example));
    m.impl("unsqueeze", TORCH_FN(at::native::unsqueeze_nested));
    m.impl("where.self", TORCH_FN(at::native::NestedTensor_where));
    m.impl("where.self_out", TORCH_FN(at::native::NestedTensor_where_out));
    m.impl("clone", TORCH_FN(at::native::clone_nested));
    m.impl("zero_", TORCH_FN(at::native::zero_nested_));
    m.impl("sub.Tensor", TORCH_FN(at::native::NestedTensor_sub_Tensor));
    m.impl("values", TORCH_FN(at::native::values_nested));
    m.impl("unbind.int", TORCH_FN(at::native::NestedTensor_unbind));
    m.impl("_to_copy", TORCH_FN(at::native::_to_copy_nested));
    m.impl("masked_fill.Scalar", TORCH_FN(at::native::NestedTensor_masked_fill));
    m.impl("view", TORCH_FN(at::native::view_nested));
    m.impl("eq.Scalar", TORCH_FN(at::native::eq_scalar_nested));
    m.impl("eq.Tensor", TORCH_FN(at::native::eq_tensor_nested));
    m.impl("ge.Scalar", TORCH_FN(at::native::ge_scalar_nested));
    m.impl("gt.Scalar", TORCH_FN(at::native::gt_scalar_nested));
    m.impl("normal_", TORCH_FN(at::native::normal_nested_));
    m.impl("alias", TORCH_FN(at::native::alias_nested));
    m.impl("isinf", TORCH_FN(at::native::NestedTensor_isinf));
    m.impl("isposinf", TORCH_FN(at::native::NestedTensor_isposinf));
    m.impl("isneginf", TORCH_FN(at::native::NestedTensor_isneginf));
    m.impl("_test_autograd_multiple_dispatch.fullcoverage", TORCH_FN(at::native::_test_autograd_multiple_dispatch_fullcoverage));
    m.impl("_test_autograd_multiple_dispatch.ntonly", TORCH_FN(at::native::_test_autograd_multiple_dispatch_ntonly));
    m.impl("_safe_softmax", TORCH_FN(at::native::_safe_softmax));
    m.impl("_transformer_encoder_layer_fwd", TORCH_FN(at::native::transformer_encoder_layer_forward));
    m.impl("to_padded_tensor", TORCH_FN(at::native::NestedTensor_to_padded_tensor_generic));
}
