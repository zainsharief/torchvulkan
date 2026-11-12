#include <torch/extension.h>
#include <c10/util/accumulate.h>
#include <algorithm>
#include <array>
#include <tuple>
#include <vector>
#include "vulkan/vulkan_context.h"
#include "api/ops/helpers.h"

namespace {

struct ConvGeometry {
    std::vector<int64_t> kernel;
    std::vector<int64_t> stride;
    std::vector<int64_t> padding;
    std::vector<int64_t> dilation;
};

int64_t window_count(int64_t size, int64_t kernel, int64_t stride, int64_t padding, int64_t dilation)
{
    return (size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
}

std::vector<int64_t> window_counts(c10::IntArrayRef spatial, const ConvGeometry& g)
{
    std::vector<int64_t> counts;
    for (size_t d = 0; d < spatial.size(); d++) counts.push_back(window_count(spatial[d], g.kernel[d], g.stride[d], g.padding[d], g.dilation[d]));
    return counts;
}

at::Tensor im2col_nd(const at::Tensor& input, const ConvGeometry& g, c10::optional<std::vector<int64_t>> window_override = c10::nullopt)
{
    int64_t dims = static_cast<int64_t>(g.kernel.size());
    std::vector<int64_t> pad;
    for (int64_t d = dims - 1; d >= 0; d--) {
        pad.push_back(g.padding[d]);
        pad.push_back(g.padding[d]);
    }
    bool unpadded = std::all_of(g.padding.begin(), g.padding.end(), [](int64_t p) { return p == 0; });
    at::Tensor x = unpadded ? input : at::constant_pad_nd(input, pad, 0);
    for (int64_t d = 0; d < dims; d++) {
        int64_t span = (g.kernel[d] - 1) * g.dilation[d] + 1;
        x = x.unfold(2 + d, span, g.stride[d]);
        if (g.dilation[d] > 1) x = x.slice(-1, 0, span, g.dilation[d]);
    }
    if (window_override.has_value()) {
        for (int64_t d = 0; d < dims; d++) x = x.narrow(2 + d, 0, (*window_override)[d]);
    }

    std::vector<int64_t> order = {0, 1};
    for (int64_t d = 0; d < dims; d++) order.push_back(2 + dims + d);
    for (int64_t d = 0; d < dims; d++) order.push_back(2 + d);
    at::Tensor cols = x.permute(order);
    int64_t windows = 1;
    for (int64_t d = 0; d < dims; d++) windows *= x.size(2 + d);
    return cols.reshape({input.size(0), input.size(1) * c10::multiply_integers(g.kernel), windows});
}

at::Tensor col2im_nd(
    const at::Tensor& cols,
    int64_t channels,
    c10::IntArrayRef output_size,
    const ConvGeometry& g,
    c10::optional<std::vector<int64_t>> window_override = c10::nullopt)
{
    int64_t dims = static_cast<int64_t>(g.kernel.size());
    int64_t n = cols.size(0);
    std::vector<int64_t> padded;
    for (int64_t d = 0; d < dims; d++) padded.push_back(output_size[d] + 2 * g.padding[d]);
    std::vector<int64_t> windows = window_override.has_value() ? *window_override : window_counts(output_size, g);
    int64_t kernel_volume = c10::multiply_integers(g.kernel);
    TORCH_CHECK(cols.size(1) == channels * kernel_volume, "Expected size of input's dimension 1 to be divisible by the product of kernel_size, but got input.size(1)=",
        cols.size(1), " and kernel_size=", g.kernel
    );
    TORCH_CHECK(cols.size(2) == c10::multiply_integers(windows), "Given output_size=", output_size, ", kernel_size=", g.kernel, ", dilation=", g.dilation, ", padding=", g.padding,
        ", stride=", g.stride, ", expected size of input's dimension 2 to match the calculated number of sliding blocks ", c10::multiply_integers(windows), ", but got input.size(2)=", cols.size(2), "."
    );

    auto padded_strides = c10::contiguous_strides(padded);
    at::TensorOptions long_options = cols.options().dtype(at::kLong);
    at::Tensor positions = at::zeros({}, long_options);
    for (int64_t d = 0; d < dims; d++) {
        std::vector<int64_t> k_view(2 * dims, 1), o_view(2 * dims, 1);
        k_view[d] = g.kernel[d];
        o_view[dims + d] = windows[d];
        at::Tensor k = at::mul(at::arange(g.kernel[d], long_options), g.dilation[d] * padded_strides[d]).view(k_view);
        at::Tensor o = at::mul(at::arange(windows[d], long_options), g.stride[d] * padded_strides[d]).view(o_view);
        positions = at::add(positions, at::add(k, o));
    }
    positions = positions.reshape({-1});

    at::Tensor image = at::zeros({n, channels, c10::multiply_integers(padded)}, cols.options());
    image.index_add_(2, positions, cols.reshape({n, channels, kernel_volume * c10::multiply_integers(windows)}));
    std::vector<int64_t> shape = {n, channels};
    shape.insert(shape.end(), padded.begin(), padded.end());
    at::Tensor result = image.view(shape);
    for (int64_t d = 0; d < dims; d++) result = result.narrow(2 + d, g.padding[d], output_size[d]);
    return result.contiguous();
}

std::vector<int64_t> expand_param(at::IntArrayRef values, int64_t dims)
{
    std::vector<int64_t> out(dims);
    for (int64_t i = 0; i < dims; i++) out[i] = values.size() == 1 ? values[0] : values[i];
    return out;
}

ConvGeometry make_geometry(c10::IntArrayRef kernel, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation)
{
    int64_t dims = static_cast<int64_t>(kernel.size());
    return ConvGeometry{kernel.vec(), expand_param(stride, dims), expand_param(padding, dims), expand_param(dilation, dims)};
}

at::Tensor im2col_vulkan(const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride)
{
    TORCH_CHECK(kernel_size.size() == 2 && dilation.size() == 2 && padding.size() == 2 && stride.size() == 2, "im2col: expected 2 values each for kernel_size, dilation, padding and stride");
    TORCH_CHECK(self.dim() == 3 || self.dim() == 4, "Expected 3D or 4D (batch mode) tensor for input with possibly 0 batch size and non-zero dimensions, but got: ", self.sizes());
    bool batched = self.dim() == 4;
    at::Tensor input = batched ? self : self.unsqueeze(0);
    ConvGeometry g = make_geometry(kernel_size, stride, padding, dilation);
    for (int64_t d = 0; d < 2; d++) {
        TORCH_CHECK(g.kernel[d] > 0 && g.stride[d] > 0 && g.dilation[d] > 0 && g.padding[d] >= 0, "im2col: kernel size, stride and dilation should be positive and padding non-negative");
        TORCH_CHECK(window_count(input.size(2 + d), g.kernel[d], g.stride[d], g.padding[d], g.dilation[d]) >= 1,
            "Given input with spatial size (", input.size(2), ", ", input.size(3), "), kernel_size=", kernel_size, ", calculated shape of the array of sliding blocks is non-positive."
        );
    }
    at::Tensor cols = im2col_nd(input, g);
    return batched ? cols : cols.squeeze(0);
}

at::Tensor& im2col_out_vulkan(const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor& out)
{
    return fill_out(out, im2col_vulkan(self, kernel_size, dilation, padding, stride));
}

at::Tensor col2im_vulkan(
    const at::Tensor& self,
    c10::SymIntArrayRef output_size_,
    at::IntArrayRef kernel_size,
    at::IntArrayRef dilation,
    at::IntArrayRef padding,
    at::IntArrayRef stride)
{
    std::vector<int64_t> output_size;
    for (const c10::SymInt& s : output_size_) output_size.push_back(s.guard_int(__FILE__, __LINE__));
    TORCH_CHECK(output_size.size() == 2 && kernel_size.size() == 2, "col2im: expected 2 values each for output_size and kernel_size");
    TORCH_CHECK(self.dim() == 2 || self.dim() == 3, "Expected 2D or 3D (batch mode) tensor for input with possibly 0 batch size and non-zero dimensions, but got: ", self.sizes());
    bool batched = self.dim() == 3;
    at::Tensor cols = batched ? self : self.unsqueeze(0);
    ConvGeometry g = make_geometry(kernel_size, stride, padding, dilation);
    int64_t kernel_volume = c10::multiply_integers(g.kernel);
    TORCH_CHECK(cols.size(1) % kernel_volume == 0, "Expected size of input's dimension 1 to be divisible by the product of kernel_size, but got input.size(1)=",
        cols.size(1), " and kernel_size=", kernel_size
    );
    at::Tensor image = col2im_nd(cols, cols.size(1) / kernel_volume, output_size, g);
    return batched ? image : image.squeeze(0);
}

at::Tensor& col2im_out_vulkan(
    const at::Tensor& self,
    c10::SymIntArrayRef output_size,
    at::IntArrayRef kernel_size,
    at::IntArrayRef dilation,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::Tensor& out)
{
    return fill_out(out, col2im_vulkan(self, output_size, kernel_size, dilation, padding, stride));
}

std::vector<int64_t> spatial_kernel(const at::Tensor& weight)
{
    return std::vector<int64_t>(weight.sizes().begin() + 2, weight.sizes().end());
}

at::Tensor add_bias(const at::Tensor& output, const c10::optional<at::Tensor>& bias)
{
    if (!bias.has_value() || !bias->defined()) return output;
    std::vector<int64_t> view(output.dim(), 1);
    view[1] = bias->numel();
    return at::add(output, bias->view(view));
}

at::Tensor conv_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const ConvGeometry& g,
    int64_t groups,
    c10::optional<std::vector<int64_t>> window_override = c10::nullopt)
{
    int64_t n = input.size(0);
    int64_t out_channels = weight.size(0);
    std::vector<int64_t> windows = window_override.has_value() ? *window_override : window_counts(input.sizes().slice(2), g);
    at::Tensor cols = im2col_nd(input, g, window_override).view({n, groups, -1, c10::multiply_integers(windows)});
    at::Tensor w = weight.reshape({groups, out_channels / groups, -1});
    at::Tensor out = at::matmul(w.unsqueeze(0), cols);
    std::vector<int64_t> shape = {n, out_channels};
    shape.insert(shape.end(), windows.begin(), windows.end());
    return out.reshape(shape);
}

at::Tensor conv_transpose_forward(const at::Tensor& input, const at::Tensor& weight, const ConvGeometry& g, int64_t groups, c10::IntArrayRef output_size)
{
    int64_t n = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t out_channels = weight.size(1) * groups;
    int64_t positions = 1;
    for (int64_t d = 2; d < input.dim(); d++) positions *= input.size(d);
    at::Tensor x = input.reshape({n, groups, in_channels / groups, positions});
    at::Tensor w = weight.reshape({groups, in_channels / groups, -1}).transpose(1, 2);
    at::Tensor cols = at::matmul(w.unsqueeze(0), x).reshape({n, out_channels * c10::multiply_integers(g.kernel), positions});
    std::vector<int64_t> windows(input.sizes().begin() + 2, input.sizes().end());
    return col2im_nd(cols, out_channels, output_size, g, windows);
}

std::vector<int64_t> transposed_output_size(const at::Tensor& input, const ConvGeometry& g, const std::vector<int64_t>& output_padding)
{
    std::vector<int64_t> sizes;
    for (size_t d = 0; d < g.kernel.size(); d++) {
        sizes.push_back((input.size(2 + d) - 1) * g.stride[d] - 2 * g.padding[d] + g.dilation[d] * (g.kernel[d] - 1) + output_padding[d] + 1);
    }
    return sizes;
}

enum class ConvMode : uint32_t {
    FORWARD = 0,
    GRAD_INPUT = 1,
    GRAD_WEIGHT = 2
};

struct ConvProblem {
    ConvGeometry g;
    int64_t n;
    int64_t groups;
    int64_t cx;
    int64_t cy;
    std::vector<int64_t> in;
    std::vector<int64_t> out;
};

const uint32_t CONV_TILE_N = 128;
const uint32_t CONV_TILE_K = 16;
const uint32_t CONV_WG = 16;
const int64_t CONV_DIRECT_ROWS = 8;
const int64_t CONV_DIRECT_COLUMNS = 8;

int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

uint32_t conv_tile_m(DeviceContext* device, at::ScalarType dtype)
{
    size_t accum = c10::elementSize(reduce_accumulate_dtype(dtype));
    auto fits = [&](uint32_t m) {
        return accum * (m * (CONV_TILE_K + 1) + CONV_TILE_K * CONV_TILE_N) <= device->properties.limits.maxComputeSharedMemorySize;
    };
    uint32_t tile_m = c10::elementSize(dtype) < 8 ? 128 : 64;
    if (tile_m == 128 && !fits(128)) tile_m = 64;
    return fits(tile_m) ? tile_m : 0;
}

struct ConvLaunch {
    bool direct;
    uint32_t tile_m;
};

ConvLaunch conv_launch(DeviceContext* device, at::ScalarType dtype, int64_t rows)
{
    uint32_t tile_m = rows <= 64 ? 64 : conv_tile_m(device, dtype);
    return ConvLaunch{rows <= CONV_DIRECT_ROWS, tile_m};
}

std::pair<int64_t, int64_t> conv_grid(const ConvLaunch& launch, int64_t rows, int64_t columns)
{
    if (launch.direct) return {ceil_div(columns, CONV_WG * CONV_WG * CONV_DIRECT_COLUMNS), ceil_div(rows, CONV_DIRECT_ROWS)};
    return {ceil_div(columns, CONV_TILE_N), ceil_div(rows, launch.tile_m)};
}

bool implicit_gemm_supported(const ConvProblem& p, std::initializer_list<at::Tensor> operands)
{
    const at::Tensor& first = *operands.begin();
    at::ScalarType dtype = first.scalar_type();
    bool floating = dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kDouble;
    if (!floating || !is_dtype_supported(first, dtype) || p.g.kernel.empty() || p.g.kernel.size() > 3) return false;
    DeviceContext* device = get_device_context(first);
    const auto& limits = device->properties.limits;
    if (CONV_WG * CONV_WG > limits.maxComputeWorkGroupInvocations || conv_tile_m(device, dtype) == 0) return false;
    if (p.groups > limits.maxComputeWorkGroupCount[2]) return false;
    int64_t max_columns = static_cast<int64_t>(limits.maxComputeWorkGroupCount[0]) * CONV_TILE_N;
    if (c10::multiply_integers(p.in) > max_columns || c10::multiply_integers(p.out) > max_columns || p.cx * c10::multiply_integers(p.g.kernel) > max_columns) return false;
    for (const at::Tensor& t : operands) {
        if (t.defined() && t.numel() == 0) return false;
    }
    return fits_32bit_indexing(operands);
}

MetadataBuilder conv_metadata(const ConvProblem& p)
{
    int64_t dims = static_cast<int64_t>(p.g.kernel.size());
    std::vector<uint32_t> params;
    auto padded = [&](const std::vector<int64_t>& values, int64_t fill) {
        std::vector<uint32_t> r(3 - dims, static_cast<uint32_t>(fill));
        for (int64_t v : values) r.push_back(static_cast<uint32_t>(v));
        return r;
    };
    auto push_divider = [&](uint32_t value) {
        IntDivider divider;
        divider.set(0, value);
        params.push_back(divider.divisor[0]);
        params.push_back(divider.multiplier[0]);
        params.push_back(divider.shift_val[0]);
    };
    for (const std::vector<int64_t>* size : {&p.in, &p.out, &p.g.kernel}) {
        std::vector<uint32_t> s = padded(*size, 1);
        params.insert(params.end(), s.begin(), s.end());
        push_divider(s[1]);
        push_divider(s[2]);
    }
    std::vector<uint32_t> stride = padded(p.g.stride, 1);
    std::vector<uint32_t> padding = padded(p.g.padding, 0);
    std::vector<uint32_t> dilation = padded(p.g.dilation, 1);
    params.insert(params.end(), stride.begin(), stride.end());
    params.insert(params.end(), padding.begin(), padding.end());
    params.insert(params.end(), dilation.begin(), dilation.end());
    params.push_back(static_cast<uint32_t>(p.groups));
    params.push_back(static_cast<uint32_t>(p.cx));
    params.push_back(static_cast<uint32_t>(p.cy));
    for (uint32_t s : stride) push_divider(s);
    push_divider(static_cast<uint32_t>(c10::multiply_integers(p.in)));
    push_divider(static_cast<uint32_t>(c10::multiply_integers(p.out)));
    push_divider(static_cast<uint32_t>(c10::multiply_integers(p.g.kernel)));
    push_divider(static_cast<uint32_t>(p.groups));
    TORCH_INTERNAL_ASSERT(params.size() == 60);

    MetadataBuilder metadata{};
    metadata.push_array(params.data(), static_cast<uint32_t>(params.size()));
    return metadata;
}

void dispatch_conv_shader(
    ConvMode mode,
    const ConvProblem& p,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& y,
    const at::Tensor& bias,
    int64_t splits = 1,
    int64_t chunk = 0)
{
    DeviceContext* device = get_device_context(x);
    at::ScalarType dtype = x.scalar_type();
    int64_t taps = c10::multiply_integers(p.g.kernel);
    int64_t inputs = c10::multiply_integers(p.in);
    int64_t outputs = c10::multiply_integers(p.out);
    int64_t vec = get_dtype_vec_size(dtype);
    bool has_bias = bias.defined();
    const auto& limits = device->properties.limits;

    int64_t rows = mode == ConvMode::GRAD_INPUT ? p.cx : p.cy;
    int64_t positions = mode == ConvMode::FORWARD ? outputs : inputs;
    int64_t depth = mode == ConvMode::FORWARD ? p.cx * taps : mode == ConvMode::GRAD_INPUT ? p.cy * taps : chunk;
    ConvLaunch launch = conv_launch(device, dtype, rows);
    int64_t columns_per_workgroup = launch.direct ? CONV_WG * CONV_WG * CONV_DIRECT_COLUMNS : CONV_TILE_N;
    int64_t items_per_dispatch = mode == ConvMode::GRAD_WEIGHT
        ? p.n
        : std::max<int64_t>(1, static_cast<int64_t>(limits.maxComputeWorkGroupCount[0]) * columns_per_workgroup / positions);

    std::vector<at::Tensor> reads, writes;
    switch (mode) {
        case ConvMode::FORWARD: reads = {x, weight}; writes = {y}; break;
        case ConvMode::GRAD_INPUT: reads = {y, weight}; writes = {x}; break;
        case ConvMode::GRAD_WEIGHT: reads = {x, y}; writes = {weight}; break;
    }
    if (has_bias) reads.push_back(bias);

    MetadataBuilder metadataBuilder = conv_metadata(p);
    Metadata metadata = metadataBuilder.build();

    for (int64_t item0 = 0; item0 < p.n; item0 += items_per_dispatch) {
        int64_t items = std::min(items_per_dispatch, p.n - item0);
        int64_t columns = mode == ConvMode::GRAD_WEIGHT ? p.cx * taps : items * positions;
        uint32_t aligned = (depth % vec == 0) && (columns % vec == 0);

        SpecializationBuilder spd{};
        spd.push(CONV_WG)
           .push(CONV_WG)
           .push(aligned)
           .push(static_cast<uint32_t>(has_bias))
           .push(uint32_t(0)) // ACTIVATION
           .push(static_cast<uint32_t>(mode))
           .push(static_cast<uint32_t>(launch.direct));

        uint64_t x_address = get_tensor_address(x) + item0 * p.groups * p.cx * inputs * x.element_size();
        uint64_t y_address = get_tensor_address(y) + item0 * p.groups * p.cy * outputs * y.element_size();

        PushConstantBuilder pcs{};
        pcs.push(x_address)
           .push(get_tensor_address(weight))
           .push(y_address)
           .push(has_bias ? get_tensor_address(bias) : uint64_t(0))
           .push_metadata(&metadata)
           .push(static_cast<uint32_t>(items))
           .push(static_cast<uint32_t>(splits))
           .push(static_cast<uint32_t>(chunk));

        auto [grid_x, grid_y] = conv_grid(launch, rows, columns);

        device->shader_manager->dispatchShader(
            ShaderKey{launch.tile_m == 128 ? torchvulkan::Kernel::CONV_128 : torchvulkan::Kernel::CONV_64, dtype},
            device->default_subgroup_size,
            spd.build(),
            pcs.build(),
            /* read = */ reads,
            /* write = */ writes,
            static_cast<uint32_t>(grid_x),
            static_cast<uint32_t>(grid_y),
            static_cast<uint32_t>(p.groups * splits)
        );
    }
}

at::Tensor implicit_forward(const ConvProblem& p, const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias)
{
    std::vector<int64_t> shape = {p.n, p.groups * p.cy};
    shape.insert(shape.end(), p.out.begin(), p.out.end());
    at::Tensor y = at::empty(shape, x.options());
    dispatch_conv_shader(ConvMode::FORWARD, p, x, weight, y, bias);
    return y;
}

at::Tensor implicit_grad_input(const ConvProblem& p, const at::Tensor& y, const at::Tensor& weight, const at::Tensor& bias)
{
    std::vector<int64_t> shape = {p.n, p.groups * p.cx};
    shape.insert(shape.end(), p.in.begin(), p.in.end());
    at::Tensor x = at::empty(shape, y.options());
    dispatch_conv_shader(ConvMode::GRAD_INPUT, p, x, weight, y, bias);
    return x;
}

at::Tensor implicit_grad_weight(const ConvProblem& p, const at::Tensor& x, const at::Tensor& y, c10::IntArrayRef weight_sizes)
{
    DeviceContext* device = get_device_context(x);
    const int64_t target_threads = 65536;
    const int64_t min_chunk = 256;
    int64_t taps = c10::multiply_integers(p.g.kernel);
    int64_t length = p.n * c10::multiply_integers(p.out);
    ConvLaunch launch = conv_launch(device, x.scalar_type(), p.cy);
    auto [grid_x, grid_y] = conv_grid(launch, p.cy, p.cx * taps);
    int64_t threads = launch.direct ? p.cx * taps * grid_y * p.groups : grid_x * grid_y * CONV_WG * CONV_WG * p.groups;
    int64_t max_splits = std::max<int64_t>(1, device->properties.limits.maxComputeWorkGroupCount[2] / p.groups);
    int64_t splits = std::clamp<int64_t>(ceil_div(target_threads, threads), 1, std::max<int64_t>(1, length / min_chunk));
    splits = std::min(splits, max_splits);
    int64_t chunk = ceil_div(ceil_div(length, splits), CONV_TILE_K) * CONV_TILE_K;
    splits = ceil_div(length, chunk);

    at::Tensor partial = at::empty({splits, p.groups * p.cy, p.cx * taps}, x.options().dtype(reduce_accumulate_dtype(x.scalar_type())));
    if (!fits_32bit_indexing({partial})) return at::Tensor();
    dispatch_conv_shader(ConvMode::GRAD_WEIGHT, p, x, partial, y, at::Tensor(), splits, chunk);
    at::Tensor summed = splits == 1 ? partial.squeeze(0) : at::sum(partial, 0);
    return summed.to(x.scalar_type()).view(weight_sizes);
}

ConvProblem forward_problem(
    const at::Tensor& x,
    const at::Tensor& weight,
    const ConvGeometry& g,
    int64_t groups,
    c10::optional<std::vector<int64_t>> window_override = c10::nullopt)
{
    std::vector<int64_t> in(x.sizes().begin() + 2, x.sizes().end());
    std::vector<int64_t> out = window_override.has_value() ? *window_override : window_counts(in, g);
    return ConvProblem{g, x.size(0), groups, x.size(1) / groups, weight.size(0) / groups, in, out};
}

ConvProblem transposed_problem(const at::Tensor& y, const at::Tensor& weight, const ConvGeometry& g, int64_t groups, std::vector<int64_t> in)
{
    std::vector<int64_t> out(y.sizes().begin() + 2, y.sizes().end());
    return ConvProblem{g, y.size(0), groups, weight.size(1), y.size(1) / groups, in, out};
}

at::Tensor convolution_vulkan(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    c10::SymIntArrayRef stride_,
    c10::SymIntArrayRef padding_,
    c10::SymIntArrayRef dilation_,
    bool transposed,
    c10::SymIntArrayRef output_padding_,
    c10::SymInt groups_)
{
    int64_t groups = groups_.guard_int(__FILE__, __LINE__);
    std::vector<int64_t> kernel = spatial_kernel(weight);
    int64_t dims = static_cast<int64_t>(kernel.size());
    ConvGeometry g = make_geometry(kernel, to_ints(stride_), to_ints(padding_), to_ints(dilation_));
    at::Tensor x = input.to(weight.scalar_type());
    std::vector<int64_t> output_size = transposed ? transposed_output_size(x, g, expand_param(to_ints(output_padding_), dims)) : std::vector<int64_t>();

    at::Tensor xc = x.contiguous();
    at::Tensor w = weight.contiguous();
    at::Tensor b = bias.has_value() && bias->defined() ? bias->to(x.scalar_type()).contiguous() : at::Tensor();
    ConvProblem p = transposed ? transposed_problem(xc, w, g, groups, output_size) : forward_problem(xc, w, g, groups);
    if (implicit_gemm_supported(p, {xc, w, b})) return transposed ? implicit_grad_input(p, xc, w, b) : implicit_forward(p, xc, w, b);

    at::Tensor output = transposed ? conv_transpose_forward(x, weight, g, groups, output_size) : conv_forward(x, weight, g, groups);
    return add_bias(output, bias);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward_vulkan(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    c10::SymIntArrayRef stride_,
    c10::SymIntArrayRef padding_,
    c10::SymIntArrayRef dilation_,
    bool transposed,
    c10::SymIntArrayRef output_padding_,
    c10::SymInt groups_,
    std::array<bool, 3> output_mask)
{
    int64_t groups = groups_.guard_int(__FILE__, __LINE__);
    std::vector<int64_t> kernel = spatial_kernel(weight);
    int64_t dims = static_cast<int64_t>(kernel.size());
    ConvGeometry g = make_geometry(kernel, to_ints(stride_), to_ints(padding_), to_ints(dilation_));
    int64_t n = input.size(0);

    at::Tensor grad_input, grad_weight, grad_bias;
    std::vector<int64_t> spatial(input.sizes().begin() + 2, input.sizes().end());

    at::Tensor go = grad_output.contiguous();
    at::Tensor in = input.contiguous();
    at::Tensor w = weight.contiguous();
    std::vector<int64_t> grad_spatial(go.sizes().begin() + 2, go.sizes().end());
    ConvProblem p = transposed ? transposed_problem(in, w, g, groups, grad_spatial) : forward_problem(in, w, g, groups, grad_spatial);
    const at::Tensor& x = transposed ? go : in;
    const at::Tensor& y = transposed ? in : go;
    if (implicit_gemm_supported(p, {x, w, y})) {
        if (output_mask[0]) grad_input = transposed ? implicit_forward(p, go, w, at::Tensor()) : implicit_grad_input(p, go, w, at::Tensor());
        if (output_mask[1]) grad_weight = implicit_grad_weight(p, x, y, weight.sizes());
    }

    if (output_mask[0] && !grad_input.defined()) {
        grad_input = transposed ? conv_forward(grad_output, weight, g, groups, spatial) : conv_transpose_forward(grad_output, weight, g, groups, spatial);
    }
    if (output_mask[1] && !grad_weight.defined()) {
        int64_t positions = 1;
        for (int64_t s : (transposed ? spatial : grad_spatial)) positions *= s;
        at::Tensor cols = (transposed ? im2col_nd(grad_output, g, spatial) : im2col_nd(input, g)).view({n, groups, -1, positions});
        const at::Tensor& rows = transposed ? input : grad_output;
        at::Tensor grouped = rows.reshape({n, groups, rows.size(1) / groups, positions});
        grad_weight = at::sum(at::matmul(grouped, cols.transpose(2, 3)), 0).reshape(weight.sizes());
    }
    if (output_mask[2]) {
        std::vector<int64_t> reduce = {0};
        for (int64_t d = 0; d < dims; d++) reduce.push_back(2 + d);
        grad_bias = at::sum(grad_output, reduce);
    }
    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("im2col", &im2col_vulkan);
    m.impl("im2col.out", &im2col_out_vulkan);
    m.impl("col2im", &col2im_vulkan);
    m.impl("col2im.out", &col2im_out_vulkan);
    m.impl("convolution_overrideable", &convolution_vulkan);
    m.impl("convolution_backward_overrideable", &convolution_backward_vulkan);
}
