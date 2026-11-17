#include <torch/extension.h>
#include <ATen/WrapDimUtils.h>
#include <algorithm>
#include <vector>
#include "vulkan/vulkan_context.h"
#include "api/ops/helpers.h"

namespace {

at::Tensor long_tensor(const std::vector<int64_t>& values, const at::Tensor& like)
{
    return at::tensor(values, at::TensorOptions().dtype(at::kLong)).to(like.device());
}

at::Tensor flip_vulkan(const at::Tensor& self, at::IntArrayRef dims)
{
    std::vector<int64_t> wrapped;
    for (int64_t d : dims) {
        int64_t w = c10::maybe_wrap_dim(d, self.dim(), /*wrap_scalar=*/true);
        TORCH_CHECK(std::find(wrapped.begin(), wrapped.end(), w) == wrapped.end(), "dims has duplicates, original flip dims size=", dims.size());
        if (self.dim() == 0) continue;
        wrapped.push_back(w);
    }
    at::Tensor result = self;
    for (int64_t d : wrapped) {
        int64_t n = self.size(d);
        if (n <= 1) continue;
        std::vector<int64_t> reversed(n);
        for (int64_t i = 0; i < n; i++) reversed[i] = n - 1 - i;
        result = at::index_select(result, d, long_tensor(reversed, self));
    }
    return result.is_same(self) ? self.clone() : result;
}

at::Tensor triangle(const at::Tensor& self, int64_t diagonal, bool lower)
{
    TORCH_CHECK(self.dim() >= 2, lower ? "tril" : "triu", ": input tensor must have at least 2 dimensions");
    int64_t rows = self.size(-2), cols = self.size(-1);
    at::Tensor r = at::arange(rows, self.options().dtype(at::kLong)).view({rows, 1});
    at::Tensor c = at::arange(cols, self.options().dtype(at::kLong)).view({1, cols});
    at::Tensor offset = at::sub(c, r);
    at::Tensor keep = lower ? at::le(offset, diagonal) : at::ge(offset, diagonal);
    return at::where(keep, self, at::zeros({}, self.options()));
}

at::Tensor& tril_out_vulkan(const at::Tensor& self, int64_t diagonal, at::Tensor& out)
{
    return fill_out(out, triangle(self, diagonal, true));
}

at::Tensor& triu_out_vulkan(const at::Tensor& self, int64_t diagonal, at::Tensor& out)
{
    return fill_out(out, triangle(self, diagonal, false));
}

std::vector<int64_t> pad_source(int64_t input, int64_t before, int64_t after, bool reflect)
{
    int64_t output = input + before + after;
    std::vector<int64_t> source(std::max<int64_t>(output, 0));
    for (int64_t o = 0; o < output; o++) {
        int64_t i = o - before;
        if (reflect) {
            if (i < 0) i = -i;
            if (i >= input) i = 2 * (input - 1) - i;
        } else {
            i = std::clamp<int64_t>(i, 0, input - 1);
        }
        source[o] = i;
    }
    return source;
}

void check_pad(const at::Tensor& self, at::IntArrayRef padding, int64_t dims, bool reflect, const char* name)
{
    TORCH_CHECK(static_cast<int64_t>(padding.size()) == 2 * dims, name, ": padding size is expected to be ", 2 * dims, ", but got: ", padding.size());
    // only the batch dimension (of a batched input) may be empty
    bool nonzero_rest = true;
    for (int64_t d = self.dim() == dims + 2 ? 1 : 0; d < self.dim(); d++) nonzero_rest = nonzero_rest && self.size(d) != 0;
    TORCH_CHECK((self.dim() == dims + 1 || self.dim() == dims + 2) && nonzero_rest, "Expected ", dims + 1, "D or ", dims + 2,
        "D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for input, but got: ", self.sizes()
    );
    for (int64_t k = 0; k < dims; k++) {
        int64_t d = self.dim() - 1 - k;
        int64_t input = self.size(d);
        int64_t before = padding[2 * k], after = padding[2 * k + 1];
        if (reflect) {
            TORCH_CHECK(before < input && after < input, "Argument #4: Padding size should be less than the corresponding input dimension, but got: padding (",
                before, ", ", after, ") at dimension ", d, " of input ", self.sizes()
            );
        }
        TORCH_CHECK(input + before + after >= 1, name, ": input (", self.sizes(), ") is too small. Calculated output size ", input + before + after, " is too small");
    }
}

at::Tensor pad_forward(const at::Tensor& self, at::IntArrayRef padding, int64_t dims, bool reflect, const char* name)
{
    check_pad(self, padding, dims, reflect, name);
    at::Tensor result = self;
    for (int64_t k = 0; k < dims; k++) {
        int64_t d = self.dim() - 1 - k;
        result = at::index_select(result, d, long_tensor(pad_source(self.size(d), padding[2 * k], padding[2 * k + 1], reflect), self));
    }
    return result.contiguous(self.suggest_memory_format());  // channels_last in, channels_last out
}

at::Tensor pad_backward(const at::Tensor& grad_output, const at::Tensor& self, at::IntArrayRef padding, int64_t dims, bool reflect)
{
    int64_t channel = self.dim() - dims - 1;
    TORCH_CHECK(grad_output.dim() == self.dim() && grad_output.size(channel) == self.size(channel), "grad_output channel unexpected. Expected: ",
        self.size(channel), ", Got: ", grad_output.dim() == self.dim() ? grad_output.size(channel) : -1);
    at::Tensor grad = grad_output;
    for (int64_t k = dims - 1; k >= 0; k--) {
        int64_t d = self.dim() - 1 - k;
        std::vector<int64_t> shape = grad.sizes().vec();
        shape[d] = self.size(d);
        at::Tensor source = long_tensor(pad_source(self.size(d), padding[2 * k], padding[2 * k + 1], reflect), self);
        grad = at::zeros(shape, grad.options()).index_add_(d, source, grad);
    }
    return grad.contiguous(self.suggest_memory_format());
}

// functional variants, so the result keeps the input's memory format (torch's wrapper allocates a contiguous out)
at::Tensor reflection_pad1d_vulkan(const at::Tensor& self, at::IntArrayRef padding) { return pad_forward(self, padding, 1, true, "reflection_pad1d"); }
at::Tensor reflection_pad3d_vulkan(const at::Tensor& self, at::IntArrayRef padding) { return pad_forward(self, padding, 3, true, "reflection_pad3d"); }
at::Tensor replication_pad1d_vulkan(const at::Tensor& self, at::IntArrayRef padding) { return pad_forward(self, padding, 1, false, "replication_pad1d"); }
at::Tensor replication_pad2d_vulkan(const at::Tensor& self, at::IntArrayRef padding) { return pad_forward(self, padding, 2, false, "replication_pad2d"); }
at::Tensor replication_pad3d_vulkan(const at::Tensor& self, at::IntArrayRef padding) { return pad_forward(self, padding, 3, false, "replication_pad3d"); }

at::Tensor& reflection_pad1d_out_vulkan(const at::Tensor& self, at::IntArrayRef padding, at::Tensor& out)
{
    return fill_out(out, pad_forward(self, padding, 1, true, "reflection_pad1d"));
}

at::Tensor reflection_pad2d_vulkan(const at::Tensor& self, at::IntArrayRef padding)
{
    return pad_forward(self, padding, 2, true, "reflection_pad2d");
}

at::Tensor& reflection_pad2d_out_vulkan(const at::Tensor& self, at::IntArrayRef padding, at::Tensor& out)
{
    return fill_out(out, reflection_pad2d_vulkan(self, padding));
}

at::Tensor& reflection_pad3d_out_vulkan(const at::Tensor& self, at::IntArrayRef padding, at::Tensor& out)
{
    return fill_out(out, pad_forward(self, padding, 3, true, "reflection_pad3d"));
}

at::Tensor& replication_pad1d_out_vulkan(const at::Tensor& self, at::IntArrayRef padding, at::Tensor& out)
{
    return fill_out(out, pad_forward(self, padding, 1, false, "replication_pad1d"));
}

at::Tensor& replication_pad2d_out_vulkan(const at::Tensor& self, at::IntArrayRef padding, at::Tensor& out)
{
    return fill_out(out, pad_forward(self, padding, 2, false, "replication_pad2d"));
}

at::Tensor& replication_pad3d_out_vulkan(const at::Tensor& self, at::IntArrayRef padding, at::Tensor& out)
{
    return fill_out(out, pad_forward(self, padding, 3, false, "replication_pad3d"));
}

at::Tensor reflection_pad1d_backward_vulkan(const at::Tensor& grad_output, const at::Tensor& self, at::IntArrayRef padding) { return pad_backward(grad_output, self, padding, 1, true); }
at::Tensor reflection_pad3d_backward_vulkan(const at::Tensor& grad_output, const at::Tensor& self, at::IntArrayRef padding) { return pad_backward(grad_output, self, padding, 3, true); }
at::Tensor replication_pad1d_backward_vulkan(const at::Tensor& grad_output, const at::Tensor& self, at::IntArrayRef padding) { return pad_backward(grad_output, self, padding, 1, false); }

at::Tensor& reflection_pad1d_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& self, at::IntArrayRef padding, at::Tensor& grad_input)
{
    return fill_out(grad_input, pad_backward(grad_output, self, padding, 1, true));
}

at::Tensor reflection_pad2d_backward_vulkan(const at::Tensor& grad_output, const at::Tensor& self, at::IntArrayRef padding)
{
    return pad_backward(grad_output, self, padding, 2, true);
}

at::Tensor& reflection_pad2d_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& self, at::IntArrayRef padding, at::Tensor& grad_input)
{
    return fill_out(grad_input, reflection_pad2d_backward_vulkan(grad_output, self, padding));
}

at::Tensor& reflection_pad3d_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& self, at::IntArrayRef padding, at::Tensor& grad_input)
{
    return fill_out(grad_input, pad_backward(grad_output, self, padding, 3, true));
}

at::Tensor& replication_pad1d_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& self, at::IntArrayRef padding, at::Tensor& grad_input)
{
    return fill_out(grad_input, pad_backward(grad_output, self, padding, 1, false));
}

at::Tensor replication_pad2d_backward_vulkan(const at::Tensor& grad_output, const at::Tensor& self, at::IntArrayRef padding)
{
    return pad_backward(grad_output, self, padding, 2, false);
}

at::Tensor& replication_pad2d_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& self, at::IntArrayRef padding, at::Tensor& grad_input)
{
    return fill_out(grad_input, replication_pad2d_backward_vulkan(grad_output, self, padding));
}

at::Tensor replication_pad3d_backward_vulkan(const at::Tensor& grad_output, const at::Tensor& self, at::IntArrayRef padding)
{
    return pad_backward(grad_output, self, padding, 3, false);
}

at::Tensor& replication_pad3d_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& self, at::IntArrayRef padding, at::Tensor& grad_input)
{
    return fill_out(grad_input, replication_pad3d_backward_vulkan(grad_output, self, padding));
}

at::Tensor channel_shuffle_vulkan(const at::Tensor& self, c10::SymInt groups_)
{
    int64_t groups = groups_.guard_int(__FILE__, __LINE__);
    TORCH_CHECK(self.dim() > 2, "channel_shuffle expects input to have at least 3 dimensions, but got input with sizes ", self.sizes());
    TORCH_CHECK(groups > 0, "Number of groups to divide channels in must be positive. Value of groups:", groups);
    int64_t channels = self.size(1);
    TORCH_CHECK(channels % groups == 0, "Number of channels must be divisible by groups. Got ", channels, " channels and ", groups, " groups.");
    std::vector<int64_t> grouped = self.sizes().vec();
    grouped[1] = channels / groups;
    grouped.insert(grouped.begin() + 1, groups);
    return self.reshape(grouped).transpose(1, 2).contiguous().reshape(self.sizes());
}

at::Tensor unfold_backward_vulkan(const at::Tensor& grad_in, c10::SymIntArrayRef input_sizes_, int64_t dim, int64_t size, int64_t step)
{
    std::vector<int64_t> input_sizes;
    TORCH_CHECK_VALUE(step > 0, "step is ", step, " but must be > 0");
    TORCH_CHECK(size >= 0, "size is ", size, " but must be >= 0");
    for (const c10::SymInt& s : input_sizes_) input_sizes.push_back(s.guard_int(__FILE__, __LINE__));
    int64_t d = c10::maybe_wrap_dim(dim, static_cast<int64_t>(input_sizes.size()), /*wrap_scalar=*/true);
    at::Tensor grad_input = at::zeros(input_sizes, grad_in.options());
    if (grad_in.numel() == 0) return grad_input;
    if (input_sizes.empty()) return grad_input.add_(grad_in.reshape({}));

    int64_t windows = grad_in.size(d);
    at::Tensor stacked = grad_in.movedim(-1, d + 1);
    std::vector<int64_t> merged = stacked.sizes().vec();
    merged[d] = windows * size;
    merged.erase(merged.begin() + d + 1);
    stacked = stacked.reshape(merged);
    // the CPU broadcasts grad over the other dimensions (istft passes a batch of 1)
    std::vector<int64_t> target = input_sizes;
    target[d] = windows * size;
    stacked = stacked.expand(target);

    at::Tensor starts = at::mul(at::arange(windows, grad_in.options().dtype(at::kLong)), step).view({windows, 1});
    at::Tensor positions = at::add(starts, at::arange(size, grad_in.options().dtype(at::kLong)).view({1, size})).reshape({-1});
    return grad_input.index_add_(d, positions, stacked);
}

at::Tensor& embedding_renorm_vulkan_(at::Tensor& self, const at::Tensor& indices, double max_norm, double norm_type)
{
    TORCH_CHECK(self.dim() == 2, "embedding_renorm_(): expected self to be 2-D");
    if (indices.numel() == 0) return self;
    at::Tensor rows = std::get<0>(at::_unique(indices.reshape({-1}).to(at::kLong), /*sorted=*/true, /*return_inverse=*/false));
    at::Tensor selected = at::index_select(self, 0, rows);
    at::Tensor norms = at::linalg_vector_norm(selected, norm_type, {1}, /*keepdim=*/true);
    at::Tensor scale = at::where(at::gt(norms, max_norm), at::div(at::full_like(norms, max_norm), at::add(norms, 1e-7)), at::ones_like(norms));
    self.index_copy_(0, rows, at::mul(selected, scale));
    return self;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("flip", &flip_vulkan);
    m.impl("tril.out", &tril_out_vulkan);
    m.impl("triu.out", &triu_out_vulkan);
    m.impl("reflection_pad1d.out", &reflection_pad1d_out_vulkan);
    m.impl("reflection_pad1d", &reflection_pad1d_vulkan);
    m.impl("reflection_pad3d", &reflection_pad3d_vulkan);
    m.impl("replication_pad1d", &replication_pad1d_vulkan);
    m.impl("replication_pad2d", &replication_pad2d_vulkan);
    m.impl("replication_pad3d", &replication_pad3d_vulkan);
    m.impl("reflection_pad2d", &reflection_pad2d_vulkan);
    m.impl("reflection_pad2d.out", &reflection_pad2d_out_vulkan);
    m.impl("reflection_pad3d.out", &reflection_pad3d_out_vulkan);
    m.impl("replication_pad1d.out", &replication_pad1d_out_vulkan);
    m.impl("replication_pad2d.out", &replication_pad2d_out_vulkan);
    m.impl("replication_pad3d.out", &replication_pad3d_out_vulkan);
    m.impl("reflection_pad1d_backward", &reflection_pad1d_backward_vulkan);
    m.impl("reflection_pad1d_backward.grad_input", &reflection_pad1d_backward_out_vulkan);
    m.impl("reflection_pad3d_backward", &reflection_pad3d_backward_vulkan);
    m.impl("replication_pad1d_backward", &replication_pad1d_backward_vulkan);
    m.impl("reflection_pad2d_backward", &reflection_pad2d_backward_vulkan);
    m.impl("reflection_pad2d_backward.grad_input", &reflection_pad2d_backward_out_vulkan);
    m.impl("reflection_pad3d_backward.grad_input", &reflection_pad3d_backward_out_vulkan);
    m.impl("replication_pad1d_backward.grad_input", &replication_pad1d_backward_out_vulkan);
    m.impl("replication_pad2d_backward", &replication_pad2d_backward_vulkan);
    m.impl("replication_pad2d_backward.grad_input", &replication_pad2d_backward_out_vulkan);
    m.impl("replication_pad3d_backward", &replication_pad3d_backward_vulkan);
    m.impl("replication_pad3d_backward.grad_input", &replication_pad3d_backward_out_vulkan);
    m.impl("channel_shuffle", &channel_shuffle_vulkan);
    m.impl("unfold_backward", &unfold_backward_vulkan);
    m.impl("embedding_renorm_", &embedding_renorm_vulkan_);
}
