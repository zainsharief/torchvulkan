#include <torch/extension.h>
#include <c10/util/safe_numerics.h>
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
    bool nonzero_rest = true;
    for (int64_t d = self.dim() == 4 ? 1 : 0; d < self.dim(); d++) nonzero_rest = nonzero_rest && self.size(d) != 0;
    TORCH_CHECK((self.dim() == 3 || self.dim() == 4) && nonzero_rest,
        "Expected 3D or 4D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for input, but got: ", self.sizes());
    bool batched = self.dim() == 4;
    at::Tensor input = batched ? self : self.unsqueeze(0);
    ConvGeometry g = make_geometry(kernel_size, stride, padding, dilation);
    for (int64_t d = 0; d < 2; d++) {
    }
    // im2col_shape_check's messages
    TORCH_CHECK(kernel_size[0] > 0 && kernel_size[1] > 0, "kernel size should be greater than zero, but got kernel_height: ", kernel_size[0],
        " kernel_width: ", kernel_size[1]);
    TORCH_CHECK(stride[0] > 0 && stride[1] > 0, "stride should be greater than zero, but got stride_height: ", stride[0], " stride_width: ", stride[1]);
    TORCH_CHECK(dilation[0] > 0 && dilation[1] > 0, "dilation should be greater than zero, but got dilation_height: ", dilation[0],
        " dilation_width: ", dilation[1]);
    TORCH_CHECK(padding[0] >= 0 && padding[1] >= 0, "padding should be non-negative, but got pad_height: ", padding[0], " pad_width: ", padding[1]);
    int64_t blocks_h = window_count(input.size(2), g.kernel[0], g.stride[0], g.padding[0], g.dilation[0]);
    int64_t blocks_w = window_count(input.size(3), g.kernel[1], g.stride[1], g.padding[1], g.dilation[1]);
    TORCH_CHECK(blocks_h >= 1 && blocks_w >= 1, "Given input with spatial size (", input.size(2), ", ", input.size(3), "), kernel_size=(", kernel_size[0], ", ",
        kernel_size[1], "), dilation=(", dilation[0], ", ", dilation[1], "), padding=(", padding[0], ", ", padding[1],
        "), calculated shape of the array of sliding blocks as (", blocks_h, ", ", blocks_w, "), but its components must be at least one.");
    at::Tensor cols = im2col_nd(input, g);
    if (cols.is_alias_of(self)) cols = cols.clone();  // a 1x1 kernel is only a reshape; the op returns a new tensor
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
    int64_t kernel_product = 0;
    TORCH_CHECK(!c10::mul_overflows(kernel_size[0], kernel_size[1], &kernel_product), "the product of kernel_width and kernel_height overflowed.");
    TORCH_CHECK(self.dim() == 2 || self.dim() == 3, "Expected 2D or 3D (batch mode) tensor for input with possibly 0 batch size and non-zero dimensions, but got: ", self.sizes());
    bool batched = self.dim() == 3;
    at::Tensor cols = batched ? self : self.unsqueeze(0);
    ConvGeometry g = make_geometry(kernel_size, stride, padding, dilation);
    int64_t kernel_volume = c10::multiply_integers(g.kernel);
    TORCH_CHECK(cols.size(1) % kernel_volume == 0, "Expected size of input's dimension 1 to be divisible by the product of kernel_size, but got input.size(1)=",
        cols.size(1), " and kernel_size=", kernel_size
    );
    at::Tensor image = col2im_nd(cols, cols.size(1) / kernel_volume, output_size, g);
    if (image.storage_offset() != 0 || !image.is_contiguous()) image = image.clone(at::MemoryFormat::Contiguous);  // a new tensor, not the unpadded view
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
const size_t CONV_PARAMS = 33; // PARAMS in helpers/conv.slang
// conv's and winograd's own spec constants start here, past the GEMM libraries': CONV_SPEC in helpers/conv.slang, FLIP in
// helpers/winograd.slang
const uint32_t OP_SPEC = 24;

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
    auto push_divider = [&](uint32_t value) {
        IntDivider divider;
        divider.set(0, value);
        params.push_back(divider.divisor[0]);
        params.push_back(divider.multiplier[0]);
        params.push_back(divider.shift_val[0]);
    };
    for (const std::vector<int64_t>* size : {&p.in, &p.out}) {
        std::vector<uint32_t> s(3 - dims, 1);
        for (int64_t v : *size) s.push_back(static_cast<uint32_t>(v));
        params.insert(params.end(), s.begin(), s.end());
        push_divider(s[1]);
        push_divider(s[2]);
    }
    params.push_back(static_cast<uint32_t>(p.groups));
    params.push_back(static_cast<uint32_t>(p.cx));
    params.push_back(static_cast<uint32_t>(p.cy));
    push_divider(static_cast<uint32_t>(c10::multiply_integers(p.in)));
    push_divider(static_cast<uint32_t>(c10::multiply_integers(p.out)));
    push_divider(static_cast<uint32_t>(p.groups));
    push_divider(static_cast<uint32_t>(p.cy));
    TORCH_INTERNAL_ASSERT(params.size() == CONV_PARAMS);

    MetadataBuilder metadata{};
    metadata.push_array(params.data(), static_cast<uint32_t>(params.size()));
    return metadata;
}

const int64_t CONV_MAX_PHASE_TAPS = 64; // MAX_PHASE_TAPS in helpers/conv.slang
const uint32_t CONV_GATHER_COLUMNS = 4; // GATHER_COLUMNS in conv_coop.slang

bool conv_padded(const ConvGeometry& g)
{
    return std::any_of(g.padding.begin(), g.padding.end(), [](int64_t v) { return v != 0; });
}

bool conv_unit_stride(const ConvGeometry& g)
{
    return std::all_of(g.stride.begin(), g.stride.end(), [](int64_t v) { return v == 1; });
}

struct SubpixelSplit {
    int64_t phases = 0;
    int64_t positions = 0;
    int64_t taps = 0;
};

SubpixelSplit subpixel_split(ConvMode mode, const ConvProblem& p, DeviceContext* device)
{
    if (mode != ConvMode::GRAD_INPUT || conv_unit_stride(p.g)) return {};
    SubpixelSplit split{1, 1, 1};
    for (size_t d = 0; d < p.g.kernel.size(); d++) {
        int64_t stride = p.g.stride[d];
        int64_t most = 0;
        for (int64_t phase = 0; phase < stride; phase++) {
            int64_t count = 0;
            for (int64_t k = 0; k < p.g.kernel[d]; k++) count += (phase + p.g.padding[d] - k * p.g.dilation[d]) % stride == 0;
            most = std::max(most, count);
        }
        split.phases *= stride;
        split.positions *= ceil_div(p.in[d], stride);
        split.taps *= most;
    }
    if (split.taps > CONV_MAX_PHASE_TAPS || p.groups * split.phases > device->properties.limits.maxComputeWorkGroupCount[2]) return {};
    return split;
}

void push_conv_geometry(SpecializationBuilder& spd, const ConvGeometry& g, const SubpixelSplit& subpixel)
{
    for (const std::vector<int64_t>* values : {&g.kernel, &g.stride, &g.padding, &g.dilation}) {
        uint32_t fill = values == &g.padding ? 0 : 1;
        for (size_t i = values->size(); i < 3; i++) spd.push(fill);
        for (int64_t v : *values) spd.push(static_cast<uint32_t>(v));
    }
    spd.push(static_cast<uint32_t>(conv_padded(g)))
       .push(static_cast<uint32_t>(conv_unit_stride(g)))
       .push(static_cast<uint32_t>(subpixel.phases > 0));
}

void conv_operands(
    ConvMode mode,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& y,
    const at::Tensor& bias,
    std::vector<at::Tensor>& reads,
    std::vector<at::Tensor>& writes)
{
    switch (mode) {
        case ConvMode::FORWARD: reads = {x, weight}; writes = {y}; break;
        case ConvMode::GRAD_INPUT: reads = {y, weight}; writes = {x}; break;
        case ConvMode::GRAD_WEIGHT: reads = {x, y}; writes = {weight}; break;
    }
    if (bias.defined()) reads.push_back(bias);
    for (const at::Tensor& t : writes) materialize_cow(t);
}

// conv items per SIMD dispatch: the grid's x dimension is limited
int64_t conv_simd_items(ConvMode mode, const ConvProblem& p, const at::Tensor& x)
{
    if (mode == ConvMode::GRAD_WEIGHT) return p.n;
    DeviceContext* device = get_device_context(x);
    SubpixelSplit subpixel = subpixel_split(mode, p, device);
    int64_t rows = mode == ConvMode::GRAD_INPUT ? p.cx : p.cy;
    int64_t positions = mode == ConvMode::FORWARD ? c10::multiply_integers(p.out) : subpixel.phases ? subpixel.positions : c10::multiply_integers(p.in);
    ConvLaunch launch = conv_launch(device, x.scalar_type(), rows);
    int64_t columns_per_workgroup = launch.direct ? CONV_WG * CONV_WG * CONV_DIRECT_COLUMNS : CONV_TILE_N;
    return std::max<int64_t>(1, static_cast<int64_t>(device->properties.limits.maxComputeWorkGroupCount[0]) * columns_per_workgroup / positions);
}

// the cooperative-matrix tuner times this too: emulated cooperative matrices can lose to it by far
TuneDispatch conv_simd_dispatch(
    ConvMode mode,
    const ConvProblem& p,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& y,
    const at::Tensor& bias,
    const std::vector<at::Tensor>& reads,
    const std::vector<at::Tensor>& writes,
    const Metadata& metadata,
    int64_t splits,
    int64_t chunk,
    int64_t item0,
    int64_t items)
{
    DeviceContext* device = get_device_context(x);
    at::ScalarType dtype = x.scalar_type();
    int64_t taps = c10::multiply_integers(p.g.kernel);
    int64_t inputs = c10::multiply_integers(p.in);
    int64_t outputs = c10::multiply_integers(p.out);
    int64_t vec = get_dtype_vec_size(dtype);
    bool has_bias = bias.defined();

    SubpixelSplit subpixel = subpixel_split(mode, p, device);
    int64_t rows = mode == ConvMode::GRAD_INPUT ? p.cx : p.cy;
    int64_t positions = mode == ConvMode::FORWARD ? outputs : subpixel.phases ? subpixel.positions : inputs;
    int64_t depth = mode == ConvMode::FORWARD ? p.cx * taps : mode == ConvMode::GRAD_INPUT ? p.cy * (subpixel.phases ? subpixel.taps : taps) : chunk;
    ConvLaunch launch = conv_launch(device, dtype, rows);
    int64_t columns = mode == ConvMode::GRAD_WEIGHT ? p.cx * taps : items * positions;
    uint32_t aligned = !subpixel.phases && (depth % vec == 0) && (columns % vec == 0);

    TuneDispatch d;
    d.key = ShaderKey{torchvulkan::Kernel::CONV_SIMD, dtype};
    d.subgroupSize = device->default_subgroup_size;
    d.spd.push(CONV_WG)
         .push(CONV_WG)
         .push(aligned)
         .push(static_cast<uint32_t>(has_bias))
         .push(uint32_t(0)) // ACTIVATION
         .push(launch.tile_m)
         .push(static_cast<uint32_t>(launch.direct))
         .at(OP_SPEC)
         .push(static_cast<uint32_t>(mode));
    push_conv_geometry(d.spd, p.g, subpixel);

    uint64_t x_address = get_tensor_address(x) + item0 * p.groups * p.cx * inputs * x.element_size();
    uint64_t y_address = get_tensor_address(y) + item0 * p.groups * p.cy * outputs * y.element_size();

    d.pcs.push(x_address)
         .push(get_tensor_address(weight))
         .push(y_address)
         .push(has_bias ? get_tensor_address(bias) : uint64_t(0))
         .push_metadata(&metadata)
         .push(static_cast<uint32_t>(items))
         .push(static_cast<uint32_t>(splits))
         .push(static_cast<uint32_t>(chunk));

    auto [grid_x, grid_y] = conv_grid(launch, rows, columns);
    d.read = reads;
    d.write = writes;
    d.groupX = static_cast<uint32_t>(grid_x);
    d.groupY = static_cast<uint32_t>(grid_y);
    d.groupZ = static_cast<uint32_t>(p.groups * (subpixel.phases ? subpixel.phases : splits));
    return d;
}

CoopTiling coop_tiling(const TuneConfig& config)
{
    return CoopTiling{config[0], config[1], config[2], config[3], config[4], config[5], config[6]};
}

bool dispatch_conv_coop_shader(
    ConvMode mode,
    const ConvProblem& p,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& y,
    const at::Tensor& bias,
    int64_t splits,
    int64_t chunk)
{
    DeviceContext* device = get_device_context(x);
    const auto& limits = device->properties.limits;
    at::ScalarType dtype = x.scalar_type();
    at::ScalarType accumulate = reduce_accumulate_dtype(dtype);
    std::vector<uint32_t> blocks = coop_block_sizes(device, dtype, accumulate);
    if (blocks.empty()) return false;

    SubpixelSplit subpixel = subpixel_split(mode, p, device);
    int64_t taps = c10::multiply_integers(p.g.kernel);
    int64_t inputs = c10::multiply_integers(p.in);
    int64_t outputs = c10::multiply_integers(p.out);
    bool has_bias = bias.defined();

    int64_t rows = mode == ConvMode::GRAD_INPUT ? p.cx : p.cy;
    int64_t positions = mode == ConvMode::FORWARD ? outputs : subpixel.phases ? subpixel.positions : inputs;
    int64_t depth = mode == ConvMode::FORWARD ? p.cx * taps : mode == ConvMode::GRAD_INPUT ? p.cy * (subpixel.phases ? subpixel.taps : taps) : chunk;
    int64_t gemm_columns = mode == ConvMode::GRAD_WEIGHT ? p.cx * taps : p.n * positions;
    int64_t planes = p.groups * (subpixel.phases ? subpixel.phases : splits);
    // grad weight gathers both operands along k, with B transposed
    bool k_major = mode == ConvMode::GRAD_WEIGHT;
    uint32_t gathered = k_major ? 1 : CONV_GATHER_COLUMNS;

    std::vector<at::Tensor> reads, writes;
    conv_operands(mode, x, weight, y, bias, reads, writes);

    MetadataBuilder metadataBuilder = conv_metadata(p);
    Metadata metadata = metadataBuilder.build();

    // the last param picks the cooperative-matrix kernel (1) or the SIMD one (0)
    const size_t kernel_param = 7;
    auto items_per_dispatch = [&](const TuneConfig& config) {
        if (config[kernel_param] == 0) return conv_simd_items(mode, p, x);
        if (mode == ConvMode::GRAD_WEIGHT) return p.n;
        return std::max<int64_t>(1, static_cast<int64_t>(limits.maxComputeWorkGroupCount[0]) * coop_tile_n(coop_tiling(config)) / positions);
    };

    auto make_dispatch = [&](const TuneConfig& config, int64_t item0, int64_t items) {
        if (config[kernel_param] == 0) return conv_simd_dispatch(mode, p, x, weight, y, bias, reads, writes, metadata, splits, chunk, item0, items);
        CoopTiling t = coop_tiling(config);
        uint32_t threads = coop_threads(t);
        uint32_t tile_m = coop_tile_m(t);
        uint32_t tile_n = coop_tile_n(t);
        int64_t columns = mode == ConvMode::GRAD_WEIGHT ? p.cx * taps : items * positions;

        TuneDispatch d;
        d.key = ShaderKey{torchvulkan::Kernel::CONV_COOP, dtype};
        d.subgroupSize = t.subgroup;
        d.spd.push(threads)
             .push(t.subgroup)
             .push(t.warps_m)
             .push(t.warps_n)
             .push(t.frags_m)
             .push(t.frags_n)
             .push(t.bk)
             .push(uint32_t(0)) // HAS_ALPHA
             .push(uint32_t(0)) // ACTIVATION
             .push(static_cast<uint32_t>(has_bias))
             .push(uint32_t(0)) // A_TRANSPOSED
             .push(static_cast<uint32_t>(k_major)) // B_TRANSPOSED
             .push(uint32_t(0)) // OUT_ALIGNED
             .push(static_cast<uint32_t>(ceil_div(tile_m * t.bk, threads))) // LOADS_A
             .push(uint32_t(0)) // M_ALIGNED
             .push(uint32_t(0)) // N_ALIGNED
             .push(uint32_t(0)) // A_VEC_ALIGNED
             .push(uint32_t(0)) // B_VEC_ALIGNED
             .push(t.block)
             .push(uint32_t(1)) // USE_SHARED_MEMORY
             .push(static_cast<uint32_t>(ceil_div(t.bk * tile_n / gathered, threads))) // LOADS_B
             .at(OP_SPEC)
             .push(static_cast<uint32_t>(mode));
        push_conv_geometry(d.spd, p.g, subpixel);

        uint64_t x_address = get_tensor_address(x) + item0 * p.groups * p.cx * inputs * x.element_size();
        uint64_t y_address = get_tensor_address(y) + item0 * p.groups * p.cy * outputs * y.element_size();

        d.pcs.push(x_address)
             .push(get_tensor_address(weight))
             .push(y_address)
             .push(has_bias ? get_tensor_address(bias) : uint64_t(0))
             .push_metadata(&metadata)
             .push(static_cast<uint32_t>(items))
             .push(static_cast<uint32_t>(splits))
             .push(static_cast<uint32_t>(chunk));

        d.read = reads;
        d.write = writes;
        d.groupX = static_cast<uint32_t>(ceil_div(columns, tile_n));
        d.groupY = static_cast<uint32_t>(ceil_div(rows, tile_m));
        d.groupZ = static_cast<uint32_t>(planes);
        return d;
    };

    uint64_t max_m = uint64_t(1) << log2_ceil(rows);
    uint64_t max_n = uint64_t(1) << log2_ceil(gemm_columns);
    TuneSpace space;
    space.params = {
        {"block", blocks},
        {"warps_m", {1, 2, 4, 8}},
        {"warps_n", {1, 2, 4, 8}},
        {"frags_m", {1, 2, 4}},
        {"frags_n", {1, 2, 4}},
        {"bk", {8, 16, 32, 64}},
        {"subgroup", subgroup_size_options(device)},
        {"kernel", {0, 1}},
    };
    space.kernel_param = kernel_param;
    space.valid = [&](const TuneConfig& config) {
        if (config[kernel_param] == 0) {
            // the SIMD kernel takes only the first option of the cooperative-matrix params
            for (size_t i = 0; i < kernel_param; i++) {
                if (config[i] != space.params[i].options.front()) return false;
            }
            return true;
        }
        CoopTiling t = coop_tiling(config);
        uint64_t subgroups = t.warps_m * t.warps_n;
        uint64_t threads = coop_threads(t);
        uint64_t tile_m = coop_tile_m(t);
        uint64_t tile_n = coop_tile_n(t);

        // must match Shared<T> and EpilogueShared in helpers/gemm_coop.slang, and params and the phase taps in helpers/conv.slang
        uint64_t stage_b = k_major ? tile_n * (t.bk + COOP_PAD) : t.bk * (tile_n + COOP_PAD);
        uint64_t staged = c10::elementSize(dtype) * 2 * (tile_m * (t.bk + COOP_PAD) + stage_b);
        uint64_t epilogue = c10::elementSize(accumulate) * subgroups * t.block * t.block;
        uint64_t conv_shared = CONV_PARAMS * sizeof(uint32_t) + CONV_MAX_PHASE_TAPS * 20 + 4;
        if (t.bk % t.block != 0 || staged + epilogue + conv_shared > limits.maxComputeSharedMemorySize) return false;

        if (subgroups > 8 || !fits_workgroup_size(device, threads, t.subgroup)) return false;
        if (k_major ? threads % t.bk != 0 : (tile_n % gathered != 0 || threads % (tile_n / gathered) != 0)) return false;
        if (ceil_div(tile_m * t.bk, threads) > 16 || ceil_div(t.bk * tile_n / gathered, threads) * gathered > 16) return false;
        if (tile_m > std::max<uint64_t>(2 * max_m, t.block) || tile_n > std::max<uint64_t>(2 * max_n, t.block)) return false;
        return ceil_div(max_m, tile_m) <= limits.maxComputeWorkGroupCount[1];
    };

    uint64_t problem = static_cast<uint64_t>(mode) | (log2_ceil(rows) << 8) | (log2_ceil(gemm_columns) << 16) | (log2_ceil(depth) << 24) | (log2_ceil(planes) << 32);
    problem |= (log2_ceil(taps) << 40) | (uint64_t(conv_padded(p.g)) << 46) | (uint64_t(conv_unit_stride(p.g)) << 47) | (uint64_t(subpixel.phases > 0) << 48);

    const TuneConfig& config = device->tuner->select(
        ShaderKey{torchvulkan::Kernel::CONV_COOP, dtype},
        problem,
        space,
        [&](const TuneConfig& c) { return make_dispatch(c, 0, std::min(p.n, items_per_dispatch(c))); }
    );
    if (config.empty()) return false;

    int64_t per_dispatch = items_per_dispatch(config);
    for (int64_t item0 = 0; item0 < p.n; item0 += per_dispatch) {
        dispatch_tuned_shader(device, make_dispatch(config, item0, std::min(per_dispatch, p.n - item0)));
    }
    return true;
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
    if (device->support_coopmat && dispatch_conv_coop_shader(mode, p, x, weight, y, bias, splits, chunk)) return;

    std::vector<at::Tensor> reads, writes;
    conv_operands(mode, x, weight, y, bias, reads, writes);

    MetadataBuilder metadataBuilder = conv_metadata(p);
    Metadata metadata = metadataBuilder.build();

    int64_t items_per_dispatch = conv_simd_items(mode, p, x);
    for (int64_t item0 = 0; item0 < p.n; item0 += items_per_dispatch) {
        int64_t items = std::min(items_per_dispatch, p.n - item0);
        dispatch_tuned_shader(device, conv_simd_dispatch(mode, p, x, weight, y, bias, reads, writes, metadata, splits, chunk, item0, items));
    }
}

enum class WinogradMode : uint32_t {
    FILTER = 0,
    INPUT = 1,
    OUTPUT = 2,
    OUTPUT_GRAD = 3,
    FILTER_GRAD = 4
};

const uint32_t WINOGRAD_WG = 64;

struct WinogradTiles {
    uint32_t height;
    uint32_t width;
    uint32_t tiles_h;
    uint32_t tiles_w;
    uint32_t tiles;
    int32_t pad_h;
    int32_t pad_w;
    uint32_t out_height = 0;
    uint32_t out_width = 0;
};

void push_winograd_constants(
    PushConstantBuilder& pcs,
    const at::Tensor& src,
    const at::Tensor& dst,
    const at::Tensor& bias,
    const at::Tensor& weight,
    uint32_t rows,
    uint32_t cols,
    const WinogradTiles& t)
{
    pcs.push(get_tensor_address(src))
       .push(get_tensor_address(dst))
       .push(bias.defined() ? get_tensor_address(bias) : uint64_t(0))
       .push(weight.defined() ? get_tensor_address(weight) : uint64_t(0))
       .push(rows)
       .push(cols)
       .push(t.height)
       .push(t.width)
       .push(t.tiles_h)
       .push(t.tiles_w)
       .push(t.tiles)
       .push(t.pad_h)
       .push(t.pad_w)
       .push(t.out_height)
       .push(t.out_width);
}

void dispatch_winograd_shader(
    WinogradMode mode,
    const at::Tensor& src,
    const at::Tensor& dst,
    const at::Tensor& bias,
    bool flip,
    uint32_t rows,
    uint32_t cols,
    const WinogradTiles& t)
{
    DeviceContext* device = get_device_context(src);

    SpecializationBuilder spd{};
    spd.push(static_cast<uint32_t>(mode))
       .push(WINOGRAD_WG)
       .push(static_cast<uint32_t>(bias.defined()))
       .at(OP_SPEC)
       .push(static_cast<uint32_t>(flip));

    PushConstantBuilder pcs{};
    push_winograd_constants(pcs, src, dst, bias, at::Tensor(), rows, cols, t);

    std::vector<at::Tensor> reads = {src};
    if (bias.defined()) reads.push_back(bias);
    uint32_t columns = mode == WinogradMode::FILTER || mode == WinogradMode::FILTER_GRAD ? cols : t.tiles;

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::WINOGRAD_TRANSFORM, src.scalar_type()},
        device->default_subgroup_size,
        spd.build(),
        pcs.build(),
        /* read = */ reads,
        /* write = */ {dst},
        static_cast<uint32_t>(ceil_div(columns, WINOGRAD_WG)), rows, 1
    );
}

// the cooperative-matrix tuner times this too: emulated cooperative matrices can lose to it by far
TuneDispatch winograd_simd_dispatch(
    const at::Tensor& src,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& dst,
    bool flip,
    uint32_t rows,
    uint32_t cols,
    const WinogradTiles& t,
    uint32_t threads,
    uint32_t row_threads,
    uint32_t thread_rows,
    uint32_t thread_cols,
    uint32_t bk)
{
    DeviceContext* device = get_device_context(src);

    TuneDispatch d;
    d.key = ShaderKey{torchvulkan::Kernel::WINOGRAD_SIMD, src.scalar_type()};
    d.subgroupSize = device->default_subgroup_size;
    d.spd.push(threads)
         .push(static_cast<uint32_t>(bias.defined()))
         .push(row_threads)
         .push(thread_rows)
         .push(thread_cols)
         .push(bk)
         .at(OP_SPEC)
         .push(static_cast<uint32_t>(flip));

    push_winograd_constants(d.pcs, src, dst, bias, weight, rows, cols, t);

    d.read = {src, weight};
    if (bias.defined()) d.read.push_back(bias);
    d.write = {dst};
    d.groupX = static_cast<uint32_t>(ceil_div(t.tiles, threads / row_threads * thread_cols));
    d.groupY = static_cast<uint32_t>(ceil_div(rows, row_threads * thread_rows));
    return d;
}

bool dispatch_winograd_coop_shader(
    const at::Tensor& src,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& dst,
    bool flip,
    uint32_t rows,
    uint32_t cols,
    const WinogradTiles& t)
{
    DeviceContext* device = get_device_context(src);
    const auto& limits = device->properties.limits;
    at::ScalarType dtype = src.scalar_type();
    at::ScalarType accumulate = reduce_accumulate_dtype(dtype);
    std::vector<uint32_t> blocks = coop_block_sizes(device, dtype, accumulate);
    if (blocks.empty()) return false;

    bool has_bias = bias.defined();
    std::vector<at::Tensor> reads = {src, weight};
    if (has_bias) reads.push_back(bias);

    // param 7 picks the cooperative-matrix kernel (1) or the SIMD one (0), which takes params 8 on
    const size_t kernel_param = 7;
    auto make_dispatch = [&](const TuneConfig& config) {
        if (config[kernel_param] == 0) return winograd_simd_dispatch(src, weight, bias, dst, flip, rows, cols, t, config[8], config[9], config[10], config[11], config[12]);
        CoopTiling c = coop_tiling(config);

        TuneDispatch d;
        d.key = ShaderKey{torchvulkan::Kernel::WINOGRAD_COOP, dtype};
        d.subgroupSize = c.subgroup;
        d.spd.push(coop_threads(c))
             .push(c.subgroup)
             .push(c.warps_m)
             .push(c.warps_n)
             .push(c.frags_m)
             .push(c.frags_n)
             .push(c.bk)
             .push(uint32_t(0)) // HAS_ALPHA
             .push(uint32_t(0)) // ACTIVATION
             .push(static_cast<uint32_t>(has_bias))
             .push(uint32_t(0)) // A_TRANSPOSED
             .push(uint32_t(0)) // B_TRANSPOSED
             .push(uint32_t(0)) // OUT_ALIGNED
             .push(uint32_t(1)) // LOADS_A
             .push(uint32_t(0)) // M_ALIGNED
             .push(uint32_t(0)) // N_ALIGNED
             .push(uint32_t(0)) // A_VEC_ALIGNED
             .push(uint32_t(0)) // B_VEC_ALIGNED
             .push(c.block)
             .push(uint32_t(1)) // USE_SHARED_MEMORY
             .push(uint32_t(1)) // LOADS_B
             .at(OP_SPEC)
             .push(static_cast<uint32_t>(flip));

        push_winograd_constants(d.pcs, src, dst, bias, weight, rows, cols, t);

        d.read = reads;
        d.write = {dst};
        d.groupX = static_cast<uint32_t>(ceil_div(t.tiles, coop_tile_n(c)));
        d.groupY = static_cast<uint32_t>(ceil_div(rows, coop_tile_m(c)));
        return d;
    };

    uint64_t max_rows = uint64_t(1) << log2_ceil(rows);
    uint64_t max_cols = uint64_t(1) << log2_ceil(t.tiles);
    TuneSpace space;
    space.params = {
        {"block", blocks},
        {"warps_m", {1, 2, 4}},
        {"warps_n", {1, 2, 4, 8}},
        {"frags_m", {1, 2}},
        {"frags_n", {1, 2, 4}},
        {"bk", {8, 16, 32}},
        {"subgroup", subgroup_size_options(device)},
        {"kernel", {0, 1}},
        {"simd_wg", {64, 128, 256}},
        {"simd_row_threads", {1, 2, 4, 8, 16, 32}},
        {"simd_thread_rows", {1, 2, 4}},
        {"simd_thread_cols", {1, 2, 4}},
        {"simd_bk", {4, 8, 16}},
    };
    space.kernel_param = kernel_param;
    space.valid = [&](const TuneConfig& config) {
        // each kernel takes only the first option of the other's params
        bool coop = config[kernel_param] == 1;
        for (size_t i = 0; i < space.params.size(); i++) {
            if (i != kernel_param && (i < kernel_param) != coop && config[i] != space.params[i].options.front()) return false;
        }
        if (!coop) {
            uint64_t threads = config[8], row_threads = config[9], thread_rows = config[10], thread_cols = config[11], bk = config[12];
            uint64_t tile_rows = row_threads * thread_rows;
            uint64_t tile_cols = threads / row_threads * thread_cols;
            // must match Shared in winograd_simd.slang
            uint64_t staged = c10::elementSize(dtype) * 16 * bk * (tile_rows + tile_cols);
            if (thread_rows * thread_cols > 4 || staged > limits.maxComputeSharedMemorySize) return false;
            if (!fits_workgroup_size(device, threads, device->default_subgroup_size)) return false;
            if (tile_rows > std::max<uint64_t>(2 * max_rows, row_threads) || tile_cols > 2 * max_cols) return false;
            return ceil_div(t.tiles, tile_cols) <= limits.maxComputeWorkGroupCount[0] && ceil_div(rows, tile_rows) <= limits.maxComputeWorkGroupCount[1];
        }
        CoopTiling c = coop_tiling(config);
        uint64_t tile_rows = coop_tile_m(c);
        uint64_t tile_cols = coop_tile_n(c);
        if (c.bk % c.block != 0 || (c.block * c.block) % c.subgroup != 0) return false;
        if (c.warps_m * c.warps_n > 8 || !fits_workgroup_size(device, coop_threads(c), c.subgroup)) return false;
        if (16 * c.frags_m * c.frags_n * c.block * c.block / c.subgroup > 64) return false;

        // must match FusedShared and FusedScratch in winograd_coop.slang
        uint64_t staged = c10::elementSize(dtype) * 16 * (tile_rows * (c.bk + COOP_PAD) + c.bk * (tile_cols + COOP_PAD));
        uint64_t scratch = c10::elementSize(accumulate) * tile_rows * tile_cols;
        if (staged + scratch > limits.maxComputeSharedMemorySize) return false;

        if (tile_rows > std::max<uint64_t>(2 * max_rows, c.block) || tile_cols > std::max<uint64_t>(2 * max_cols, c.block)) return false;
        return ceil_div(t.tiles, tile_cols) <= limits.maxComputeWorkGroupCount[0] && ceil_div(rows, tile_rows) <= limits.maxComputeWorkGroupCount[1];
    };

    const TuneConfig& config = device->tuner->select(
        ShaderKey{torchvulkan::Kernel::WINOGRAD_COOP, dtype},
        uint64_t(flip) | (log2_ceil(rows) << 8) | (log2_ceil(cols) << 16) | (log2_ceil(t.tiles) << 24) | (uint64_t(has_bias) << 40),
        space,
        make_dispatch
    );
    if (config.empty()) return false;

    dispatch_tuned_shader(device, make_dispatch(config));
    return true;
}

// Winograd F(2x2, 3x3)
bool winograd_applies(const ConvProblem& p, const at::Tensor& src, int64_t channels_in, int64_t channels_out, const std::vector<int64_t>& out)
{
    if (p.g.kernel != std::vector<int64_t>{3, 3} || p.groups != 1) return false;
    for (size_t d = 0; d < 2; d++) {
        if (p.g.stride[d] != 1 || p.g.dilation[d] != 1 || p.g.padding[d] > 2) return false;
    }
    at::ScalarType dtype = src.scalar_type();
    if (dtype != at::kHalf && dtype != at::kFloat) return false;
    if (channels_in < 16 || channels_out < 16) return false;

    int64_t tiles = p.n * ceil_div(out[0], 2) * ceil_div(out[1], 2);
    const auto& limits = get_device_context(src)->properties.limits;
    if (ceil_div(tiles, WINOGRAD_WG) > limits.maxComputeWorkGroupCount[0] || std::max(channels_in, channels_out) > limits.maxComputeWorkGroupCount[1]) return false;
    return 16 * std::max(channels_in, channels_out) * tiles <= std::numeric_limits<int32_t>::max();
}

// with flip, the input gradient: `src` is the output gradient
at::Tensor winograd_conv(const ConvProblem& p, const at::Tensor& src, const at::Tensor& weight, const at::Tensor& bias, bool flip)
{
    int64_t channels_in = flip ? p.cy : p.cx;
    int64_t channels_out = flip ? p.cx : p.cy;
    const std::vector<int64_t>& in = flip ? p.out : p.in;
    const std::vector<int64_t>& out = flip ? p.in : p.out;
    if (!winograd_applies(p, src, channels_in, channels_out, out)) return at::Tensor();
    int64_t tiles_h = ceil_div(out[0], 2);
    int64_t tiles_w = ceil_div(out[1], 2);
    int64_t tiles = p.n * tiles_h * tiles_w;

    at::TensorOptions options = src.options();
    at::Tensor y = at::empty({p.n, channels_out, out[0], out[1]}, options);

    // the input gradient pads by kernel - 1 - padding
    int32_t pad_h = static_cast<int32_t>(flip ? 2 - p.g.padding[0] : p.g.padding[0]);
    int32_t pad_w = static_cast<int32_t>(flip ? 2 - p.g.padding[1] : p.g.padding[1]);
    WinogradTiles input{uint32_t(in[0]), uint32_t(in[1]), uint32_t(tiles_h), uint32_t(tiles_w), uint32_t(tiles), pad_h, pad_w, uint32_t(out[0]), uint32_t(out[1])};
    WinogradTiles output{uint32_t(out[0]), uint32_t(out[1]), uint32_t(tiles_h), uint32_t(tiles_w), uint32_t(tiles), 0, 0};

    bool fused = std::max(channels_in, channels_out) <= 32 && get_device_context(src)->support_coopmat;
    if (fused && dispatch_winograd_coop_shader(src, weight, bias, y, flip, uint32_t(channels_out), uint32_t(channels_in), input)) return y;

    at::Tensor U = at::empty({16, channels_out, channels_in}, options);
    at::Tensor V = at::empty({16, channels_in, tiles}, options);
    dispatch_winograd_shader(WinogradMode::FILTER, weight, U, at::Tensor(), flip, uint32_t(channels_out), uint32_t(channels_in), input);
    dispatch_winograd_shader(WinogradMode::INPUT, src, V, at::Tensor(), false, uint32_t(channels_in), 0, input);
    at::Tensor M = at::bmm(U, V);
    dispatch_winograd_shader(WinogradMode::OUTPUT, M, y, bias, false, uint32_t(channels_out), 0, output);
    return y;
}

at::Tensor winograd_grad_weight(const ConvProblem& p, const at::Tensor& x, const at::Tensor& y)
{
    if (!winograd_applies(p, x, p.cx, p.cy, p.out)) return at::Tensor();
    int64_t tiles_h = ceil_div(p.out[0], 2);
    int64_t tiles_w = ceil_div(p.out[1], 2);
    int64_t tiles = p.n * tiles_h * tiles_w;

    at::TensorOptions options = x.options();
    at::Tensor V = at::empty({16, p.cx, tiles}, options);
    at::Tensor dM = at::empty({16, p.cy, tiles}, options);
    at::Tensor grad_weight = at::empty({p.cy, p.cx, 3, 3}, options);

    int32_t pad_h = static_cast<int32_t>(p.g.padding[0]);
    int32_t pad_w = static_cast<int32_t>(p.g.padding[1]);
    WinogradTiles input{uint32_t(p.in[0]), uint32_t(p.in[1]), uint32_t(tiles_h), uint32_t(tiles_w), uint32_t(tiles), pad_h, pad_w};
    WinogradTiles output{uint32_t(p.out[0]), uint32_t(p.out[1]), uint32_t(tiles_h), uint32_t(tiles_w), uint32_t(tiles), 0, 0};

    dispatch_winograd_shader(WinogradMode::INPUT, x, V, at::Tensor(), false, uint32_t(p.cx), 0, input);
    dispatch_winograd_shader(WinogradMode::OUTPUT_GRAD, y, dM, at::Tensor(), false, uint32_t(p.cy), 0, output);
    at::Tensor dU = at::bmm(dM, V.transpose(1, 2));
    dispatch_winograd_shader(WinogradMode::FILTER_GRAD, dU, grad_weight, at::Tensor(), false, uint32_t(p.cy), uint32_t(p.cx), output);
    return grad_weight;
}

at::Tensor implicit_forward(const ConvProblem& p, const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias)
{
    if (at::Tensor y = winograd_conv(p, x, weight, bias, false); y.defined()) return y;
    std::vector<int64_t> shape = {p.n, p.groups * p.cy};
    shape.insert(shape.end(), p.out.begin(), p.out.end());
    at::Tensor y = at::empty(shape, x.options());
    dispatch_conv_shader(ConvMode::FORWARD, p, x, weight, y, bias);
    return y;
}

at::Tensor implicit_grad_input(const ConvProblem& p, const at::Tensor& y, const at::Tensor& weight, const at::Tensor& bias)
{
    if (at::Tensor x = winograd_conv(p, y, weight, bias, true); x.defined()) return x;
    std::vector<int64_t> shape = {p.n, p.groups * p.cx};
    shape.insert(shape.end(), p.in.begin(), p.in.end());
    at::Tensor x = at::empty(shape, y.options());
    dispatch_conv_shader(ConvMode::GRAD_INPUT, p, x, weight, y, bias);
    return x;
}

at::Tensor implicit_grad_weight(const ConvProblem& p, const at::Tensor& x, const at::Tensor& y, c10::IntArrayRef weight_sizes)
{
    if (at::Tensor w = winograd_grad_weight(p, x, y); w.defined()) return w.view(weight_sizes);
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
    if (device->support_coopmat) {
        int64_t tiles = ceil_div(p.cy, 64) * ceil_div(p.cx * taps, 64) * p.groups;
        splits = std::clamp<int64_t>(ceil_div(128, tiles), 1, std::max<int64_t>(1, length / min_chunk));
    }
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

at::MemoryFormat conv_memory_format(const at::Tensor& input, const at::Tensor& weight)
{
    for (at::MemoryFormat format : {input.suggest_memory_format(), weight.suggest_memory_format()}) {
        if (format != at::MemoryFormat::Contiguous) return format;
    }
    return at::MemoryFormat::Contiguous;
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
    at::MemoryFormat format = conv_memory_format(input, weight);
    if (implicit_gemm_supported(p, {xc, w, b})) {
        return (transposed ? implicit_grad_input(p, xc, w, b) : implicit_forward(p, xc, w, b)).contiguous(format);
    }

    at::Tensor output = transposed ? conv_transpose_forward(x, weight, g, groups, output_size) : conv_forward(x, weight, g, groups);
    return add_bias(output, bias).contiguous(format);
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
    at::MemoryFormat format = conv_memory_format(input, weight);
    if (grad_input.defined()) grad_input = grad_input.contiguous(format);
    if (grad_weight.defined()) grad_weight = grad_weight.contiguous(format);
    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

// the direct slow/depthwise convolution ops, with the CPU's kernel_size check
at::Tensor direct_convolution(
    const at::Tensor& self,
    const at::Tensor& weight,
    c10::SymIntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias,
    c10::SymIntArrayRef stride,
    c10::SymIntArrayRef padding,
    c10::SymIntArrayRef dilation,
    bool transposed,
    c10::SymIntArrayRef output_padding,
    int64_t groups)
{
    std::vector<int64_t> kernel = spatial_kernel(weight);
    std::vector<int64_t> requested = to_ints(kernel_size);
    TORCH_CHECK(requested == kernel, "kernel_size ", c10::IntArrayRef(requested), " must match weight spatial dimensions ", c10::IntArrayRef(kernel));
    std::vector<c10::SymInt> ones(kernel.size(), 1);
    std::vector<c10::SymInt> zeros(kernel.size(), 0);
    return convolution_vulkan(self, weight, bias, stride, padding, dilation.empty() ? c10::SymIntArrayRef(ones) : dilation,
                              transposed, output_padding.empty() ? c10::SymIntArrayRef(zeros) : output_padding, groups);
}

at::Tensor slow_conv_transpose_vulkan(
    const at::Tensor& self,
    const at::Tensor& weight,
    c10::SymIntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias,
    c10::SymIntArrayRef stride,
    c10::SymIntArrayRef padding,
    c10::SymIntArrayRef output_padding,
    c10::SymIntArrayRef dilation)
{
    return direct_convolution(self, weight, kernel_size, bias, stride, padding, dilation, true, output_padding, 1);
}

at::Tensor& slow_conv_transpose_out_vulkan(
    const at::Tensor& self,
    const at::Tensor& weight,
    c10::SymIntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias,
    c10::SymIntArrayRef stride,
    c10::SymIntArrayRef padding,
    c10::SymIntArrayRef output_padding,
    c10::SymIntArrayRef dilation,
    at::Tensor& out)
{
    return fill_out(out, slow_conv_transpose_vulkan(self, weight, kernel_size, bias, stride, padding, output_padding, dilation));
}

at::Tensor slow_conv_dilated_vulkan(
    const at::Tensor& self,
    const at::Tensor& weight,
    c10::SymIntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias,
    c10::SymIntArrayRef stride,
    c10::SymIntArrayRef padding,
    c10::SymIntArrayRef dilation)
{
    return direct_convolution(self, weight, kernel_size, bias, stride, padding, dilation, false, {}, 1);
}

at::Tensor& slow_conv_dilated_out_vulkan(
    const at::Tensor& self,
    const at::Tensor& weight,
    c10::SymIntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias,
    c10::SymIntArrayRef stride,
    c10::SymIntArrayRef padding,
    c10::SymIntArrayRef dilation,
    at::Tensor& out)
{
    return fill_out(out, slow_conv_dilated_vulkan(self, weight, kernel_size, bias, stride, padding, dilation));
}

at::Tensor slow_conv_forward_vulkan(
    const at::Tensor& self,
    const at::Tensor& weight,
    c10::SymIntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias,
    c10::SymIntArrayRef stride,
    c10::SymIntArrayRef padding)
{
    return direct_convolution(self, weight, kernel_size, bias, stride, padding, {}, false, {}, 1);
}

at::Tensor& slow_conv_forward_out_vulkan(
    const at::Tensor& self,
    const at::Tensor& weight,
    c10::SymIntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias,
    c10::SymIntArrayRef stride,
    c10::SymIntArrayRef padding,
    at::Tensor& out)
{
    return fill_out(out, slow_conv_forward_vulkan(self, weight, kernel_size, bias, stride, padding));
}

at::Tensor conv_depthwise_vulkan(
    const at::Tensor& self,
    const at::Tensor& weight,
    c10::SymIntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias,
    c10::SymIntArrayRef stride,
    c10::SymIntArrayRef padding,
    c10::SymIntArrayRef dilation)
{
    return direct_convolution(self, weight, kernel_size, bias, stride, padding, dilation, false, {}, self.size(1));
}

at::Tensor& conv_depthwise_out_vulkan(
    const at::Tensor& self,
    const at::Tensor& weight,
    c10::SymIntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias,
    c10::SymIntArrayRef stride,
    c10::SymIntArrayRef padding,
    c10::SymIntArrayRef dilation,
    at::Tensor& out)
{
    return fill_out(out, conv_depthwise_vulkan(self, weight, kernel_size, bias, stride, padding, dilation));
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("im2col", &im2col_vulkan);
    m.impl("im2col.out", &im2col_out_vulkan);
    m.impl("col2im", &col2im_vulkan);
    m.impl("col2im.out", &col2im_out_vulkan);
    m.impl("convolution_overrideable", &convolution_vulkan);
    m.impl("convolution_backward_overrideable", &convolution_backward_vulkan);
    m.impl("slow_conv_transpose2d", &slow_conv_transpose_vulkan);
    m.impl("slow_conv_transpose2d.out", &slow_conv_transpose_out_vulkan);
    m.impl("slow_conv_transpose3d", &slow_conv_transpose_vulkan);
    m.impl("slow_conv_transpose3d.out", &slow_conv_transpose_out_vulkan);
    m.impl("slow_conv_dilated2d", &slow_conv_dilated_vulkan);
    m.impl("slow_conv_dilated2d.out", &slow_conv_dilated_out_vulkan);
    m.impl("slow_conv_dilated3d", &slow_conv_dilated_vulkan);
    m.impl("slow_conv_dilated3d.out", &slow_conv_dilated_out_vulkan);
    m.impl("_slow_conv2d_forward", &slow_conv_forward_vulkan);
    m.impl("_slow_conv2d_forward.output", &slow_conv_forward_out_vulkan);
    m.impl("slow_conv3d_forward", &slow_conv_forward_vulkan);
    m.impl("slow_conv3d_forward.output", &slow_conv_forward_out_vulkan);
    m.impl("_conv_depthwise2d", &conv_depthwise_vulkan);
    m.impl("_conv_depthwise2d.out", &conv_depthwise_out_vulkan);
    m.impl("conv_depthwise3d", &conv_depthwise_vulkan);
    m.impl("conv_depthwise3d.out", &conv_depthwise_out_vulkan);
}
