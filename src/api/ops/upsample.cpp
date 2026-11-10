#include <torch/extension.h>
#include <ATen/native/UpSample.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "api/ops/helpers.h"

namespace {

enum class ResampleMode {
    NEAREST = 0,
    NEAREST_EXACT = 1,
    LINEAR = 2,
    CUBIC = 3,
    AA_LINEAR = 4,
    AA_CUBIC = 5
};

using Scales = std::vector<c10::optional<double>>;

// in the opmath type, as the CPU
double axis_scale(int64_t input, int64_t output, bool align_corners, c10::optional<double> scale, bool double_math)
{
    if (double_math) return at::native::area_pixel_compute_scale<double>(input, output, align_corners, scale);
    return static_cast<double>(at::native::area_pixel_compute_scale<float>(input, output, align_corners, scale));
}

double aa_support(ResampleMode mode, double scale)
{
    double interp_size = mode == ResampleMode::AA_LINEAR ? 2.0 : 4.0;
    return scale >= 1.0 ? interp_size * 0.5 * scale : interp_size * 0.5;
}

void dispatch_resample_shader(
    const at::Tensor& input,
    const at::Tensor& output,
    int64_t in_length,
    int64_t out_length,
    int64_t inner,
    double scale,
    double support,
    ResampleMode mode,
    bool align_corners,
    bool backward)
{
    uint64_t count = output.numel();
    if (count == 0) return;
    DeviceContext* device = get_device_context(output);
    at::ScalarType dtype = output.scalar_type();
    uint32_t workgroupSizeX = get_dtype_workgroup_size(dtype, 1, device);

    SpecializationBuilder spd{};
    spd.push(static_cast<uint32_t>(mode))
       .push(static_cast<uint32_t>(backward))
       .push(static_cast<uint32_t>(align_corners))
       .push(workgroupSizeX);

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(input))
       .push(get_tensor_address(output))
       .push(count)
       .push(static_cast<uint32_t>(in_length))
       .push(static_cast<uint32_t>(out_length))
       .push(static_cast<uint32_t>(inner))
       .push(uint32_t(0))
       .push_scalar(scale, dtype)
       .push_scalar(support, dtype);

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::RESAMPLE, dtype},
        device->default_subgroup_size,
        spd.build(),
        pcs.build(),
        /* read = */ {input},
        /* write = */ {output},
        get_group_count(count, workgroupSizeX), 1, 1
    );
}

bool is_aa(ResampleMode mode) { return mode == ResampleMode::AA_LINEAR || mode == ResampleMode::AA_CUBIC; }

at::ScalarType compute_dtype(const at::Tensor& self)
{
    return (self.scalar_type() == at::kDouble && is_dtype_supported(self, at::kDouble)) ? at::kDouble : at::kFloat;
}

// quantized to int16 as ATen's _compute_index_ranges_int16_weights
struct FixedPointTaps {
    std::vector<int64_t> index_min;
    std::vector<int64_t> index_size;
    std::vector<int16_t> weights;
    int64_t taps = 0;
    int64_t precision = 0;
};

double cubic_convolution1(double x, double a) { return ((a + 2) * x - (a + 3)) * x * x + 1; }
double cubic_convolution2(double x, double a) { return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; }

double resample_filter(ResampleMode mode, double x)
{
    x = std::abs(x);
    if (mode == ResampleMode::LINEAR || mode == ResampleMode::AA_LINEAR) return x < 1.0 ? 1.0 - x : 0.0;
    double a = mode == ResampleMode::AA_CUBIC ? -0.5 : -0.75;
    if (x < 1.0) return cubic_convolution1(x, a);
    if (x < 2.0) return cubic_convolution2(x, a);
    return 0.0;
}

FixedPointTaps fixed_point_taps(int64_t input, int64_t output, bool align_corners, c10::optional<double> opt_scale, ResampleMode mode)
{
    bool antialias = is_aa(mode);
    int64_t interp_size = (mode == ResampleMode::LINEAR || mode == ResampleMode::AA_LINEAR) ? 2 : 4;
    double scale = at::native::area_pixel_compute_scale<double>(input, output, align_corners, opt_scale);
    double support = antialias && scale >= 1.0 ? interp_size * 0.5 * scale : interp_size * 0.5;
    int64_t max_interp = antialias ? static_cast<int64_t>(std::ceil(support)) * 2 + 1 : interp_size;

    FixedPointTaps t;
    t.taps = max_interp;
    std::vector<double> weights(output * max_interp, 0.0);
    double wt_max = 0.0;
    for (int64_t i = 0; i < output; i++) {
        double* w = weights.data() + i * max_interp;
        int64_t xmin = 0, xsize = 0;
        if (antialias) {
            double center = scale * (i + 0.5);
            double invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
            xmin = std::max(static_cast<int64_t>(center - support + 0.5), int64_t(0));
            xsize = std::clamp(std::min(static_cast<int64_t>(center + support + 0.5), input) - xmin, int64_t(0), max_interp);
            double total = 0.0;
            for (int64_t j = 0; j < xsize; j++) {
                w[j] = resample_filter(mode, (j + xmin - center + 0.5) * invscale);
                total += w[j];
            }
            for (int64_t j = 0; j < xsize && total != 0.0; j++) w[j] /= total;
        } else {
            // out-of-range taps fold onto the edge pixel, as _compute_indices_min_size_weights does
            bool cubic = max_interp > 2;
            double real = at::native::area_pixel_compute_source_index<double>(scale, i, align_corners, cubic);
            int64_t input_index = std::min(static_cast<int64_t>(std::floor(static_cast<float>(real))), input - 1);
            double lambda = std::min(std::max(real - input_index, 0.0), 1.0);
            int64_t half = static_cast<int64_t>(max_interp * 0.5);
            int64_t unbound_min = input_index - half + 1;
            int64_t unbound_max = input_index + half + 1;
            xmin = std::max(unbound_min, int64_t(0));
            xsize = std::clamp(std::min(unbound_max, input) - xmin, int64_t(0), max_interp);
            int64_t w_index = 0;
            for (int64_t j = 0; j < max_interp; j++) {
                double value = resample_filter(mode, static_cast<double>(j + 1 - half) - lambda);
                if (unbound_min + j <= 0) w_index = 0;
                else if (unbound_min + j >= input - 1) w_index = xsize - 1;
                w[w_index] += value;
                w_index++;
            }
        }
        for (int64_t j = 0; j < max_interp; j++) wt_max = std::max(wt_max, w[j]);
        t.index_min.push_back(xmin);
        t.index_size.push_back(xsize);
    }

    int64_t precision = 0;
    for (precision = 0; precision < 22; precision++) {
        int next = static_cast<int>(0.5 + wt_max * (1 << (precision + 1)));
        if (next >= (1 << 15)) break;
    }
    t.precision = precision;
    for (double v : weights) {
        double scaled = v * (1 << precision);
        t.weights.push_back(static_cast<int16_t>(scaled < 0 ? static_cast<int>(-0.5 + scaled) : static_cast<int>(0.5 + scaled)));
    }
    return t;
}

void dispatch_resample_u8_shader(const at::Tensor& input, const at::Tensor& output, const FixedPointTaps& t, int64_t in_length, int64_t out_length, int64_t inner)
{
    uint64_t count = output.numel();
    if (count == 0) return;
    DeviceContext* device = get_device_context(output);
    at::Tensor index_min = at::tensor(t.index_min, at::kLong).to(output.device());
    at::Tensor index_size = at::tensor(t.index_size, at::kLong).to(output.device());
    at::Tensor weights = at::tensor(t.weights, at::kShort).to(output.device());
    uint32_t workgroupSizeX = get_dtype_workgroup_size(at::kByte, 1, device);

    SpecializationBuilder spd{};
    spd.push(workgroupSizeX);

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(input))
       .push(get_tensor_address(output))
       .push(get_tensor_address(index_min))
       .push(get_tensor_address(index_size))
       .push(get_tensor_address(weights))
       .push(count)
       .push(static_cast<uint32_t>(in_length))
       .push(static_cast<uint32_t>(out_length))
       .push(static_cast<uint32_t>(inner))
       .push(static_cast<uint32_t>(t.taps))
       .push(static_cast<uint32_t>(t.precision));

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::RESAMPLE_U8, at::kByte},
        device->default_subgroup_size,
        spd.build(),
        pcs.build(),
        /* read = */ {input, index_min, index_size, weights},
        /* write = */ {output},
        get_group_count(count, workgroupSizeX), 1, 1
    );
}

// ATen's fixed-point separable uint8 passes
at::Tensor resample_forward_u8(const at::Tensor& self, const std::vector<int64_t>& output_size, const Scales& scales, ResampleMode mode, bool align_corners)
{
    int64_t dims = static_cast<int64_t>(output_size.size());
    at::Tensor current = self.contiguous();
    for (int64_t a = dims - 1; a >= 0; a--) {
        int64_t axis = 2 + a;
        int64_t in_length = current.size(axis);
        int64_t out_length = output_size[a];
        if (in_length == out_length) continue;
        int64_t inner = 1;
        for (int64_t d = axis + 1; d < current.dim(); d++) inner *= current.size(d);
        std::vector<int64_t> shape = current.sizes().vec();
        shape[axis] = out_length;
        at::Tensor next = at::empty(shape, current.options());
        dispatch_resample_u8_shader(current, next, fixed_point_taps(in_length, out_length, align_corners, scales[a], mode), in_length, out_length, inner);
        current = next;
    }
    return current.is_same(self) ? self.clone() : current;
}

void check_resample_dtype(const at::Tensor& self, ResampleMode mode, int64_t dims)
{
    at::ScalarType dtype = self.scalar_type();
    bool floating = at::isFloatingType(dtype);
    bool nearest = mode == ResampleMode::NEAREST || mode == ResampleMode::NEAREST_EXACT;
    bool allowed = floating;
    if (dtype == at::kByte) allowed = nearest || dims == 2;
    if (is_aa(mode) && is_low_prec(dtype)) allowed = false;
    TORCH_CHECK_NOT_IMPLEMENTED(allowed, "\"upsample\" not implemented for '", dtype, "'");
}

// last axis first, as the CPU's separable path
at::Tensor resample_forward(const at::Tensor& self, const std::vector<int64_t>& output_size, const Scales& scales, ResampleMode mode, bool align_corners)
{
    int64_t dims = static_cast<int64_t>(output_size.size());
    TORCH_CHECK(self.dim() == dims + 2, "Expected a ", dims + 2, "D input, but got ", self.sizes());
    for (int64_t s : output_size) TORCH_CHECK(s > 0, "Input and output sizes should be greater than 0, but got output ", output_size);
    for (int64_t d = 1; d < self.dim(); d++) {
        TORCH_CHECK(self.size(d) > 0, "Non-empty ", self.dim(), "D data tensor expected but got a tensor with sizes ", self.sizes());
    }

    std::vector<int64_t> out_shape = {self.size(0), self.size(1)};
    out_shape.insert(out_shape.end(), output_size.begin(), output_size.end());
    check_resample_dtype(self, mode, dims);
    if (self.numel() == 0) return at::empty(out_shape, self.options());
    bool nearest = mode == ResampleMode::NEAREST || mode == ResampleMode::NEAREST_EXACT;
    if (self.scalar_type() == at::kByte && !nearest) return resample_forward_u8(self, output_size, scales, mode, align_corners);

    at::ScalarType dtype = compute_dtype(self);
    at::Tensor current = self.to(dtype).contiguous();
    for (int64_t a = dims - 1; a >= 0; a--) {
        int64_t axis = 2 + a;
        int64_t in_length = current.size(axis);
        int64_t out_length = output_size[a];
        if (is_aa(mode) && in_length == out_length) continue;

        int64_t inner = 1;
        for (int64_t d = axis + 1; d < current.dim(); d++) inner *= current.size(d);
        std::vector<int64_t> shape = current.sizes().vec();
        shape[axis] = out_length;
        at::Tensor next = at::empty(shape, current.options());

        bool nearest = mode == ResampleMode::NEAREST || mode == ResampleMode::NEAREST_EXACT;
        double scale = axis_scale(in_length, out_length, nearest ? false : align_corners, scales[a], dtype == at::kDouble);
        double support = is_aa(mode) ? aa_support(mode, scale) : 0.0;
        dispatch_resample_shader(current, next, in_length, out_length, inner, scale, support, mode, align_corners, false);
        current = next;
    }
    return current.to(self.scalar_type());
}

at::Tensor resample_backward(
    const at::Tensor& grad_output,
    const std::vector<int64_t>& output_size,
    const std::vector<int64_t>& input_size,
    const Scales& scales,
    ResampleMode mode,
    bool align_corners)
{
    int64_t dims = static_cast<int64_t>(output_size.size());
    TORCH_CHECK(static_cast<int64_t>(input_size.size()) == dims + 2, "It is expected input_size equals to ", dims + 2, ", but got size ", input_size.size());
    if (grad_output.numel() == 0) return at::zeros(input_size, grad_output.options());

    at::ScalarType dtype = compute_dtype(grad_output);
    at::Tensor current = grad_output.to(dtype).contiguous();
    for (int64_t a = 0; a < dims; a++) {
        int64_t axis = 2 + a;
        int64_t in_length = input_size[axis];
        int64_t out_length = output_size[a];
        if (is_aa(mode) && in_length == out_length) continue;

        int64_t inner = 1;
        for (int64_t d = axis + 1; d < current.dim(); d++) inner *= current.size(d);
        std::vector<int64_t> shape = current.sizes().vec();
        shape[axis] = in_length;
        at::Tensor next = at::empty(shape, current.options());

        bool nearest = mode == ResampleMode::NEAREST || mode == ResampleMode::NEAREST_EXACT;
        double scale = axis_scale(in_length, out_length, nearest ? false : align_corners, scales[a], dtype == at::kDouble);
        double support = is_aa(mode) ? aa_support(mode, scale) : 0.0;
        dispatch_resample_shader(current, next, in_length, out_length, inner, scale, support, mode, align_corners, true);
        current = next;
    }
    return current.to(grad_output.scalar_type());
}

at::Tensor& upsample_nearest1d_out_vulkan(const at::Tensor& self, c10::SymIntArrayRef output_size, c10::optional<double> scales, at::Tensor& out)
{
    return fill_out(out, resample_forward(self, to_ints(output_size), {scales}, ResampleMode::NEAREST, false));
}

at::Tensor& upsample_nearest_exact1d_out_vulkan(const at::Tensor& self, c10::SymIntArrayRef output_size, c10::optional<double> scales, at::Tensor& out)
{
    return fill_out(out, resample_forward(self, to_ints(output_size), {scales}, ResampleMode::NEAREST_EXACT, false));
}

at::Tensor& upsample_nearest2d_out_vulkan(
    const at::Tensor& self,
    c10::SymIntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& out)
{
    return fill_out(out, resample_forward(self, to_ints(output_size), {scales_h, scales_w}, ResampleMode::NEAREST, false));
}

at::Tensor& upsample_nearest_exact2d_out_vulkan(
    const at::Tensor& self,
    c10::SymIntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& out)
{
    return fill_out(out, resample_forward(self, to_ints(output_size), {scales_h, scales_w}, ResampleMode::NEAREST_EXACT, false));
}

at::Tensor& upsample_nearest3d_out_vulkan(
    const at::Tensor& self,
    c10::SymIntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& out)
{
    return fill_out(out, resample_forward(self, to_ints(output_size), {scales_d, scales_h, scales_w}, ResampleMode::NEAREST, false));
}

at::Tensor& upsample_nearest_exact3d_out_vulkan(
    const at::Tensor& self,
    c10::SymIntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& out)
{
    return fill_out(out, resample_forward(self, to_ints(output_size), {scales_d, scales_h, scales_w}, ResampleMode::NEAREST_EXACT, false));
}

at::Tensor& upsample_linear1d_out_vulkan(const at::Tensor& self, c10::SymIntArrayRef output_size, bool align_corners, c10::optional<double> scales, at::Tensor& out)
{
    return fill_out(out, resample_forward(self, to_ints(output_size), {scales}, ResampleMode::LINEAR, align_corners));
}

at::Tensor& upsample_bilinear2d_out_vulkan(
    const at::Tensor& self,
    c10::SymIntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& out)
{
    return fill_out(out, resample_forward(self, to_ints(output_size), {scales_h, scales_w}, ResampleMode::LINEAR, align_corners));
}

at::Tensor& upsample_trilinear3d_out_vulkan(
    const at::Tensor& self,
    c10::SymIntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& out)
{
    return fill_out(out, resample_forward(self, to_ints(output_size), {scales_d, scales_h, scales_w}, ResampleMode::LINEAR, align_corners));
}

at::Tensor& upsample_bicubic2d_out_vulkan(
    const at::Tensor& self,
    c10::SymIntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& out)
{
    return fill_out(out, resample_forward(self, to_ints(output_size), {scales_h, scales_w}, ResampleMode::CUBIC, align_corners));
}

at::Tensor& upsample_bilinear2d_aa_out_vulkan(
    const at::Tensor& self,
    c10::SymIntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& out)
{
    return fill_out(out, resample_forward(self, to_ints(output_size), {scales_h, scales_w}, ResampleMode::AA_LINEAR, align_corners));
}

at::Tensor& upsample_bicubic2d_aa_out_vulkan(
    const at::Tensor& self,
    c10::SymIntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& out)
{
    return fill_out(out, resample_forward(self, to_ints(output_size), {scales_h, scales_w}, ResampleMode::AA_CUBIC, align_corners));
}

at::Tensor& upsample_nearest1d_backward_out_vulkan(
    const at::Tensor& grad_output,
    c10::SymIntArrayRef output_size,
    c10::SymIntArrayRef input_size,
    c10::optional<double> scales,
    at::Tensor& grad_input)
{
    return fill_out(grad_input, resample_backward(grad_output, to_ints(output_size), to_ints(input_size), {scales}, ResampleMode::NEAREST, false));
}

at::Tensor& upsample_nearest_exact1d_backward_out_vulkan(
    const at::Tensor& grad_output,
    c10::SymIntArrayRef output_size,
    c10::SymIntArrayRef input_size,
    c10::optional<double> scales,
    at::Tensor& grad_input)
{
    return fill_out(grad_input, resample_backward(grad_output, to_ints(output_size), to_ints(input_size), {scales}, ResampleMode::NEAREST_EXACT, false));
}

at::Tensor& upsample_nearest2d_backward_out_vulkan(
    const at::Tensor& grad_output,
    c10::SymIntArrayRef output_size,
    c10::SymIntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& grad_input)
{
    return fill_out(grad_input, resample_backward(grad_output, to_ints(output_size), to_ints(input_size), {scales_h, scales_w}, ResampleMode::NEAREST, false));
}

at::Tensor& upsample_nearest_exact2d_backward_out_vulkan(
    const at::Tensor& grad_output,
    c10::SymIntArrayRef output_size,
    c10::SymIntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& grad_input)
{
    return fill_out(grad_input, resample_backward(grad_output, to_ints(output_size), to_ints(input_size), {scales_h, scales_w}, ResampleMode::NEAREST_EXACT, false));
}

at::Tensor& upsample_nearest3d_backward_out_vulkan(
    const at::Tensor& grad_output,
    c10::SymIntArrayRef output_size,
    c10::SymIntArrayRef input_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& grad_input)
{
    return fill_out(grad_input, resample_backward(grad_output, to_ints(output_size), to_ints(input_size), {scales_d, scales_h, scales_w}, ResampleMode::NEAREST, false));
}

at::Tensor& upsample_nearest_exact3d_backward_out_vulkan(
    const at::Tensor& grad_output,
    c10::SymIntArrayRef output_size,
    c10::SymIntArrayRef input_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& grad_input)
{
    return fill_out(grad_input, resample_backward(grad_output, to_ints(output_size), to_ints(input_size), {scales_d, scales_h, scales_w}, ResampleMode::NEAREST_EXACT, false));
}

at::Tensor& upsample_linear1d_backward_out_vulkan(
    const at::Tensor& grad_output,
    c10::SymIntArrayRef output_size,
    c10::SymIntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales,
    at::Tensor& grad_input)
{
    return fill_out(grad_input, resample_backward(grad_output, to_ints(output_size), to_ints(input_size), {scales}, ResampleMode::LINEAR, align_corners));
}

at::Tensor& upsample_bilinear2d_backward_out_vulkan(
    const at::Tensor& grad_output,
    c10::SymIntArrayRef output_size,
    c10::SymIntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& grad_input)
{
    return fill_out(grad_input, resample_backward(grad_output, to_ints(output_size), to_ints(input_size), {scales_h, scales_w}, ResampleMode::LINEAR, align_corners));
}

at::Tensor& upsample_trilinear3d_backward_out_vulkan(
    const at::Tensor& grad_output,
    c10::SymIntArrayRef output_size,
    c10::SymIntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& grad_input)
{
    return fill_out(grad_input, resample_backward(grad_output, to_ints(output_size), to_ints(input_size), {scales_d, scales_h, scales_w}, ResampleMode::LINEAR, align_corners));
}

at::Tensor& upsample_bicubic2d_backward_out_vulkan(
    const at::Tensor& grad_output,
    c10::SymIntArrayRef output_size,
    c10::SymIntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& grad_input)
{
    return fill_out(grad_input, resample_backward(grad_output, to_ints(output_size), to_ints(input_size), {scales_h, scales_w}, ResampleMode::CUBIC, align_corners));
}

at::Tensor& upsample_bilinear2d_aa_backward_out_vulkan(
    const at::Tensor& grad_output,
    c10::SymIntArrayRef output_size,
    c10::SymIntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& grad_input)
{
    return fill_out(grad_input, resample_backward(grad_output, to_ints(output_size), to_ints(input_size), {scales_h, scales_w}, ResampleMode::AA_LINEAR, align_corners));
}

at::Tensor& upsample_bicubic2d_aa_backward_out_vulkan(
    const at::Tensor& grad_output,
    c10::SymIntArrayRef output_size,
    c10::SymIntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& grad_input)
{
    return fill_out(grad_input, resample_backward(grad_output, to_ints(output_size), to_ints(input_size), {scales_h, scales_w}, ResampleMode::AA_CUBIC, align_corners));
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("upsample_nearest1d.out", &upsample_nearest1d_out_vulkan);
    m.impl("_upsample_nearest_exact1d.out", &upsample_nearest_exact1d_out_vulkan);
    m.impl("upsample_nearest2d.out", &upsample_nearest2d_out_vulkan);
    m.impl("_upsample_nearest_exact2d.out", &upsample_nearest_exact2d_out_vulkan);
    m.impl("upsample_nearest3d.out", &upsample_nearest3d_out_vulkan);
    m.impl("_upsample_nearest_exact3d.out", &upsample_nearest_exact3d_out_vulkan);
    m.impl("upsample_linear1d.out", &upsample_linear1d_out_vulkan);
    m.impl("upsample_bilinear2d.out", &upsample_bilinear2d_out_vulkan);
    m.impl("upsample_trilinear3d.out", &upsample_trilinear3d_out_vulkan);
    m.impl("upsample_bicubic2d.out", &upsample_bicubic2d_out_vulkan);
    m.impl("_upsample_bilinear2d_aa.out", &upsample_bilinear2d_aa_out_vulkan);
    m.impl("_upsample_bicubic2d_aa.out", &upsample_bicubic2d_aa_out_vulkan);
    m.impl("upsample_nearest1d_backward.grad_input", &upsample_nearest1d_backward_out_vulkan);
    m.impl("_upsample_nearest_exact1d_backward.grad_input", &upsample_nearest_exact1d_backward_out_vulkan);
    m.impl("upsample_nearest2d_backward.grad_input", &upsample_nearest2d_backward_out_vulkan);
    m.impl("_upsample_nearest_exact2d_backward.grad_input", &upsample_nearest_exact2d_backward_out_vulkan);
    m.impl("upsample_nearest3d_backward.grad_input", &upsample_nearest3d_backward_out_vulkan);
    m.impl("_upsample_nearest_exact3d_backward.grad_input", &upsample_nearest_exact3d_backward_out_vulkan);
    m.impl("upsample_linear1d_backward.grad_input", &upsample_linear1d_backward_out_vulkan);
    m.impl("upsample_bilinear2d_backward.grad_input", &upsample_bilinear2d_backward_out_vulkan);
    m.impl("upsample_trilinear3d_backward.grad_input", &upsample_trilinear3d_backward_out_vulkan);
    m.impl("upsample_bicubic2d_backward.grad_input", &upsample_bicubic2d_backward_out_vulkan);
    m.impl("_upsample_bilinear2d_aa_backward.grad_input", &upsample_bilinear2d_aa_backward_out_vulkan);
    m.impl("_upsample_bicubic2d_aa_backward.grad_input", &upsample_bicubic2d_aa_backward_out_vulkan);
}
