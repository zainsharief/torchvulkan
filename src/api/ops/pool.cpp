#include <torch/extension.h>
#include <ATen/native/Pool.h>
#include <ATen/native/FractionalMaxPooling.h>
#include <array>
#include <tuple>
#include <vector>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "api/ops/helpers.h"

namespace {

enum class PoolMode {
    MAX = 0,
    AVG = 1,
    ADAPTIVE_MAX = 2,
    ADAPTIVE_AVG = 3,
    FRACTIONAL_MAX = 4,
    AVG_BACKWARD = 5,
    ADAPTIVE_AVG_BACKWARD = 6
};

// must match pool.slang
struct PoolGeometry {
    std::array<int64_t, 3> in{1, 1, 1};
    std::array<int64_t, 3> out{1, 1, 1};
    std::array<int64_t, 3> kernel{1, 1, 1};
    std::array<int64_t, 3> stride{1, 1, 1};
    std::array<int64_t, 3> pad{0, 0, 0};
    std::array<int64_t, 3> dilation{1, 1, 1};
    bool count_include_pad = true;
    int64_t divisor_override = 0;
    int64_t sample_dims = 0;
};

int64_t inputs_per_plane(const PoolGeometry& g) { return g.in[0] * g.in[1] * g.in[2]; }
int64_t outputs_per_plane(const PoolGeometry& g) { return g.out[0] * g.out[1] * g.out[2]; }

void dispatch_pool_shader(
    const at::Tensor& input,
    const at::Tensor& output,
    const at::Tensor& indices,
    const at::Tensor& samples,
    const PoolGeometry& g,
    PoolMode mode,
    uint64_t count)
{
    if (count == 0) return;
    DeviceContext* device = get_device_context(output);
    at::ScalarType dtype = output.scalar_type();
    uint32_t workgroupSizeX = get_dtype_workgroup_size(dtype, 1, device);

    uint32_t params[21] = {0};
    for (int a = 0; a < 3; a++) {
        params[a] = static_cast<uint32_t>(g.in[a]);
        params[3 + a] = static_cast<uint32_t>(g.out[a]);
        params[6 + a] = static_cast<uint32_t>(g.kernel[a]);
        params[9 + a] = static_cast<uint32_t>(g.stride[a]);
        params[12 + a] = static_cast<uint32_t>(g.pad[a]);
        params[15 + a] = static_cast<uint32_t>(g.dilation[a]);
    }
    params[18] = g.count_include_pad ? 1u : 0u;
    params[19] = static_cast<uint32_t>(g.divisor_override);
    params[20] = static_cast<uint32_t>(g.sample_dims);

    MetadataBuilder metadataBuilder{};
    metadataBuilder.push_array(params, 21);
    Metadata metadata = metadataBuilder.build();

    SpecializationBuilder spd{};
    spd.push(static_cast<uint32_t>(mode))
       .push(workgroupSizeX);

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(input))
       .push(get_tensor_address(output))
       .push(indices.defined() ? get_tensor_address(indices) : uint64_t(0))
       .push(samples.defined() ? get_tensor_address(samples) : uint64_t(0))
       .push_metadata(&metadata)
       .push(count);

    std::vector<at::Tensor> reads = {input};
    if (samples.defined()) reads.push_back(samples);
    std::vector<at::Tensor> writes = {output};
    if (indices.defined()) writes.push_back(indices);

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::POOL, dtype},
        device->default_subgroup_size,
        spd.build(),
        pcs.build(),
        /* read = */ reads,
        /* write = */ writes,
        get_group_count(count, workgroupSizeX), 1, 1
    );
}

std::vector<int64_t> expand_param(at::IntArrayRef values, int64_t dims, const char* name, const char* param)
{
    TORCH_CHECK(values.size() == 1 || static_cast<int64_t>(values.size()) == dims, name, ": ", param, " must either be a single int, or a tuple of ", dims, " ints");
    std::vector<int64_t> out(dims);
    for (int64_t i = 0; i < dims; i++) out[i] = values.size() == 1 ? values[0] : values[i];
    return out;
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>> split_spatial(const at::Tensor& self, int64_t dims, const char* name)
{
    TORCH_CHECK(self.dim() == dims + 1 || self.dim() == dims + 2, name, ": Expected ", dims + 1, "D or ", dims + 2, "D tensor for input, but got: ", self.sizes());
    std::vector<int64_t> leading(self.sizes().begin(), self.sizes().end() - dims);
    std::vector<int64_t> spatial(self.sizes().end() - dims, self.sizes().end());
    return std::make_tuple(leading, spatial);
}

void set_axes(std::array<int64_t, 3>& target, const std::vector<int64_t>& values)
{
    int64_t offset = 3 - static_cast<int64_t>(values.size());
    for (size_t i = 0; i < values.size(); i++) target[offset + i] = values[i];
}

std::vector<int64_t> concat(const std::vector<int64_t>& a, const std::vector<int64_t>& b)
{
    std::vector<int64_t> r = a;
    r.insert(r.end(), b.begin(), b.end());
    return r;
}

PoolGeometry window_geometry(
    const at::Tensor& self,
    int64_t dims,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    const char* name)
{
    auto [leading, spatial] = split_spatial(self, dims, name);
    std::vector<int64_t> k = expand_param(kernel_size, dims, name, "kernel_size");
    std::vector<int64_t> s = stride.empty() ? k : expand_param(stride, dims, name, "stride");
    std::vector<int64_t> p = expand_param(padding, dims, name, "padding");
    std::vector<int64_t> d = expand_param(dilation, dims, name, "dilation");
    std::vector<int64_t> o(dims);
    for (int64_t a = 0; a < dims; a++) {
        TORCH_CHECK(k[a] > 0 && s[a] > 0 && d[a] > 0, name, ": kernel size, stride and dilation should be greater than zero");
        o[a] = at::native::pooling_output_shape<int64_t>(spatial[a], k[a], p[a], s[a], d[a], ceil_mode);
    }

    int64_t planes_dim = self.dim() - dims - 1;
    if (dims == 2) {
        at::native::pool2d_shape_check(self, k[0], k[1], s[0], s[1], p[0], p[1], d[0], d[1], self.size(planes_dim), spatial[0], spatial[1], o[0], o[1],
            at::MemoryFormat::Contiguous);
    } else {
        at::native::pool3d_shape_check(self, self.size(planes_dim), k[0], k[1], k[2], s[0], s[1], s[2], p[0], p[1], p[2], d[0], d[1], d[2],
            spatial[0], spatial[1], spatial[2], o[0], o[1], o[2], name);
    }

    PoolGeometry g;
    set_axes(g.in, spatial);
    set_axes(g.out, o);
    set_axes(g.kernel, k);
    set_axes(g.stride, s);
    set_axes(g.pad, p);
    set_axes(g.dilation, d);
    return g;
}

// bfloat16 runs in float32: MoltenVK can't build its constants
at::ScalarType pool_dtype(const at::Tensor& self)
{
    at::ScalarType dtype = self.scalar_type();
    return (dtype == at::kBFloat16 || !is_dtype_supported(self, dtype)) ? at::kFloat : dtype;
}

int64_t planes_of(const at::Tensor& self, int64_t dims)
{
    int64_t planes = 1;
    for (int64_t i = 0; i < self.dim() - dims; i++) planes *= self.size(i);
    return planes;
}

std::vector<int64_t> output_shape(const at::Tensor& self, int64_t dims, const PoolGeometry& g)
{
    std::vector<int64_t> shape(self.sizes().begin(), self.sizes().end() - dims);
    for (int64_t a = 3 - dims; a < 3; a++) shape.push_back(g.out[a]);
    return shape;
}

std::tuple<at::Tensor, at::Tensor> run_max_pool(const at::Tensor& self, int64_t dims, const PoolGeometry& g, PoolMode mode, const at::Tensor& samples)
{
    std::vector<int64_t> shape = output_shape(self, dims, g);
    at::ScalarType dtype = pool_dtype(self);
    at::Tensor output = at::empty(shape, self.options().dtype(dtype));
    at::Tensor indices = at::empty(shape, self.options().dtype(at::kLong));
    int64_t planes = planes_of(self, dims);
    dispatch_pool_shader(self.to(dtype).contiguous(), output, indices, samples.defined() ? samples.to(dtype) : samples, g, mode,
        static_cast<uint64_t>(planes * outputs_per_plane(g)));
    return std::make_tuple(output.to(self.scalar_type()), indices);
}

at::Tensor max_pool_backward(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& indices, int64_t dims)
{
    int64_t planes = planes_of(self, dims);
    int64_t inputs = self.numel() / std::max<int64_t>(planes, 1);
    at::Tensor grad_input = at::zeros({planes, inputs}, grad_output.options());
    if (grad_output.numel() == 0 || planes == 0) return grad_input.view(self.sizes());
    return grad_input.scatter_add_(1, indices.reshape({planes, -1}), grad_output.reshape({planes, -1})).view(self.sizes());
}

std::tuple<at::Tensor&, at::Tensor&> max_pool2d_with_indices_out_vulkan(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    at::Tensor& out,
    at::Tensor& indices)
{
    PoolGeometry g = window_geometry(self, 2, kernel_size, stride, padding, dilation, ceil_mode, "max_pool2d");
    auto [o, i] = run_max_pool(self, 2, g, PoolMode::MAX, at::Tensor());
    fill_out(out, o);
    fill_out(indices, i);
    return std::forward_as_tuple(out, indices);
}

std::tuple<at::Tensor, at::Tensor> max_pool3d_with_indices_vulkan(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode)
{
    PoolGeometry g = window_geometry(self, 3, kernel_size, stride, padding, dilation, ceil_mode, "max_pool3d");
    return run_max_pool(self, 3, g, PoolMode::MAX, at::Tensor());
}

std::tuple<at::Tensor&, at::Tensor&> max_pool3d_with_indices_out_vulkan(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    at::Tensor& out,
    at::Tensor& indices)
{
    auto [o, i] = max_pool3d_with_indices_vulkan(self, kernel_size, stride, padding, dilation, ceil_mode);
    fill_out(out, o);
    fill_out(indices, i);
    return std::forward_as_tuple(out, indices);
}

at::Tensor& max_pool2d_with_indices_backward_out_vulkan(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::IntArrayRef /* kernel_size */,
    at::IntArrayRef /* stride */,
    at::IntArrayRef /* padding */,
    at::IntArrayRef /* dilation */,
    bool /* ceil_mode */,
    const at::Tensor& indices,
    at::Tensor& grad_input)
{
    return fill_out(grad_input, max_pool_backward(grad_output, self, indices, 2));
}

at::Tensor max_pool3d_with_indices_backward_vulkan(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::IntArrayRef /* kernel_size */,
    at::IntArrayRef /* stride */,
    at::IntArrayRef /* padding */,
    at::IntArrayRef /* dilation */,
    bool /* ceil_mode */,
    const at::Tensor& indices)
{
    return max_pool_backward(grad_output, self, indices, 3);
}

at::Tensor& max_pool3d_with_indices_backward_out_vulkan(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    const at::Tensor& indices,
    at::Tensor& grad_input)
{
    return fill_out(grad_input, max_pool3d_with_indices_backward_vulkan(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices));
}

PoolGeometry avg_geometry(
    const at::Tensor& self,
    int64_t dims,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    const char* name)
{
    TORCH_CHECK(!divisor_override.has_value() || *divisor_override != 0, "divisor must be not zero");
    PoolGeometry g = window_geometry(self, dims, kernel_size, stride, padding, {1}, ceil_mode, name);
    g.count_include_pad = count_include_pad;
    g.divisor_override = divisor_override.value_or(0);
    return g;
}

at::Tensor avg_pool_forward(const at::Tensor& self, int64_t dims, const PoolGeometry& g)
{
    at::ScalarType dtype = pool_dtype(self);
    at::Tensor output = at::empty(output_shape(self, dims, g), self.options().dtype(dtype));
    dispatch_pool_shader(self.to(dtype).contiguous(), output, at::Tensor(), at::Tensor(), g, PoolMode::AVG, static_cast<uint64_t>(planes_of(self, dims) * outputs_per_plane(g)));
    return output.to(self.scalar_type());
}

at::Tensor avg_pool_backward(const at::Tensor& grad_output, const at::Tensor& self, int64_t dims, const PoolGeometry& g)
{
    at::ScalarType dtype = pool_dtype(grad_output);
    at::Tensor grad_input = at::empty(self.sizes(), grad_output.options().dtype(dtype));
    dispatch_pool_shader(grad_output.to(dtype).contiguous(), grad_input, at::Tensor(), at::Tensor(), g, PoolMode::AVG_BACKWARD,
        static_cast<uint64_t>(planes_of(self, dims) * inputs_per_plane(g)));
    return grad_input.to(grad_output.scalar_type());
}

at::Tensor& avg_pool2d_out_vulkan(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    at::Tensor& out)
{
    PoolGeometry g = avg_geometry(self, 2, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, "avg_pool2d");
    return fill_out(out, avg_pool_forward(self, 2, g));
}

at::Tensor& avg_pool3d_out_vulkan(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    at::Tensor& out)
{
    PoolGeometry g = avg_geometry(self, 3, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, "avg_pool3d");
    return fill_out(out, avg_pool_forward(self, 3, g));
}

at::Tensor& avg_pool2d_backward_out_vulkan(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    at::Tensor& grad_input)
{
    PoolGeometry g = avg_geometry(self, 2, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, "avg_pool2d_backward");
    return fill_out(grad_input, avg_pool_backward(grad_output, self, 2, g));
}

at::Tensor& avg_pool3d_backward_out_vulkan(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    at::Tensor& grad_input)
{
    PoolGeometry g = avg_geometry(self, 3, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, "avg_pool3d_backward");
    return fill_out(grad_input, avg_pool_backward(grad_output, self, 3, g));
}

PoolGeometry adaptive_geometry(const at::Tensor& self, int64_t dims, at::IntArrayRef output_size, const char* name)
{
    auto [leading, spatial] = split_spatial(self, dims, name);
    TORCH_CHECK(static_cast<int64_t>(output_size.size()) == dims, name, ": output_size must be ", dims);
    for (int64_t i = 1; i < self.dim(); i++) {
        TORCH_CHECK(self.size(i) > 0, name, "(): Expected input to have non-zero size for non-batch dimensions, but input has sizes ",
            self.sizes(), " with dimension ", i, " being empty"
        );
    }
    PoolGeometry g;
    set_axes(g.in, spatial);
    set_axes(g.out, output_size.vec());
    return g;
}

at::Tensor adaptive_avg_pool_forward(const at::Tensor& self, int64_t dims, at::IntArrayRef output_size, const char* name)
{
    PoolGeometry g = adaptive_geometry(self, dims, output_size, name);
    at::ScalarType dtype = pool_dtype(self);
    at::Tensor output = at::empty(output_shape(self, dims, g), self.options().dtype(dtype));
    dispatch_pool_shader(self.to(dtype).contiguous(), output, at::Tensor(), at::Tensor(), g, PoolMode::ADAPTIVE_AVG,
        static_cast<uint64_t>(planes_of(self, dims) * outputs_per_plane(g)));
    return output.to(self.scalar_type());
}

at::Tensor adaptive_avg_pool_backward(const at::Tensor& grad_output, const at::Tensor& self, int64_t dims)
{
    std::vector<int64_t> output_size(grad_output.sizes().end() - dims, grad_output.sizes().end());
    PoolGeometry g = adaptive_geometry(self, dims, output_size, "adaptive_avg_pool_backward");
    at::ScalarType dtype = pool_dtype(grad_output);
    at::Tensor grad_input = at::empty(self.sizes(), grad_output.options().dtype(dtype));
    dispatch_pool_shader(grad_output.to(dtype).contiguous(), grad_input, at::Tensor(), at::Tensor(), g, PoolMode::ADAPTIVE_AVG_BACKWARD,
        static_cast<uint64_t>(planes_of(self, dims) * inputs_per_plane(g)));
    return grad_input.to(grad_output.scalar_type());
}

at::Tensor adaptive_avg_pool2d_vulkan(const at::Tensor& self, c10::SymIntArrayRef output_size)
{
    return adaptive_avg_pool_forward(self, 2, to_ints(output_size), "adaptive_avg_pool2d");
}

at::Tensor adaptive_avg_pool3d_vulkan(const at::Tensor& self, c10::SymIntArrayRef output_size)
{
    return adaptive_avg_pool_forward(self, 3, to_ints(output_size), "adaptive_avg_pool3d");
}

at::Tensor adaptive_avg_pool2d_backward_vulkan(const at::Tensor& grad_output, const at::Tensor& self)
{
    return adaptive_avg_pool_backward(grad_output, self, 2);
}

at::Tensor adaptive_avg_pool3d_backward_vulkan(const at::Tensor& grad_output, const at::Tensor& self)
{
    return adaptive_avg_pool_backward(grad_output, self, 3);
}

at::Tensor& adaptive_avg_pool2d_out_vulkan(const at::Tensor& self, c10::SymIntArrayRef output_size, at::Tensor& out)
{
    return fill_out(out, adaptive_avg_pool2d_vulkan(self, output_size));
}

at::Tensor& adaptive_avg_pool3d_out_vulkan(const at::Tensor& self, c10::SymIntArrayRef output_size, at::Tensor& out)
{
    return fill_out(out, adaptive_avg_pool3d_vulkan(self, output_size));
}

at::Tensor& adaptive_avg_pool3d_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& self, at::Tensor& grad_input)
{
    return fill_out(grad_input, adaptive_avg_pool3d_backward_vulkan(grad_output, self));
}

std::tuple<at::Tensor&, at::Tensor&> adaptive_max_pool2d_out_vulkan(const at::Tensor& self, at::IntArrayRef output_size, at::Tensor& out, at::Tensor& indices)
{
    PoolGeometry g = adaptive_geometry(self, 2, output_size, "adaptive_max_pool2d");
    auto [o, i] = run_max_pool(self, 2, g, PoolMode::ADAPTIVE_MAX, at::Tensor());
    fill_out(out, o);
    fill_out(indices, i);
    return std::forward_as_tuple(out, indices);
}

std::tuple<at::Tensor&, at::Tensor&> adaptive_max_pool3d_out_vulkan(const at::Tensor& self, at::IntArrayRef output_size, at::Tensor& out, at::Tensor& indices)
{
    PoolGeometry g = adaptive_geometry(self, 3, output_size, "adaptive_max_pool3d");
    auto [o, i] = run_max_pool(self, 3, g, PoolMode::ADAPTIVE_MAX, at::Tensor());
    fill_out(out, o);
    fill_out(indices, i);
    return std::forward_as_tuple(out, indices);
}

at::Tensor& adaptive_max_pool2d_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& indices, at::Tensor& grad_input)
{
    return fill_out(grad_input, max_pool_backward(grad_output, self, indices, 2));
}

at::Tensor& adaptive_max_pool3d_backward_out_vulkan(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& indices, at::Tensor& grad_input)
{
    return fill_out(grad_input, max_pool_backward(grad_output, self, indices, 3));
}

std::tuple<at::Tensor, at::Tensor> fractional_max_pool(
    const at::Tensor& self,
    int64_t dims,
    at::IntArrayRef kernel_size,
    at::IntArrayRef output_size,
    const at::Tensor& random_samples,
    const char* name)
{
    if (dims == 2) at::native::fractional_max_pool_check_shape<2>(self, random_samples);
    else at::native::fractional_max_pool_check_shape<3>(self, random_samples);
    auto [leading, spatial] = split_spatial(self, dims, name);
    std::vector<int64_t> k = expand_param(kernel_size, dims, name, "kernel_size");
    TORCH_CHECK(static_cast<int64_t>(output_size.size()) == dims, name, ": output_size must be ", dims);
    for (int64_t a = 0; a < dims; a++) {
        TORCH_CHECK(output_size[a] + k[a] - 1 < spatial[a] + 1, name, "(): pool size ", k[a], " too large relative to input size ", spatial[a]);
    }
    for (int64_t i = 1; i < self.dim(); i++) {
        TORCH_CHECK(self.size(i) > 0, name, "(): Expected input to have non-zero size for non-batch dimensions, but got ", self.sizes());
    }

    PoolGeometry g;
    set_axes(g.in, spatial);
    set_axes(g.out, output_size.vec());
    set_axes(g.kernel, k);
    g.sample_dims = dims;
    int64_t planes = planes_of(self, dims);
    at::Tensor samples = random_samples.reshape({-1, dims}).narrow(0, 0, planes).contiguous();
    return run_max_pool(self, dims, g, PoolMode::FRACTIONAL_MAX, samples);
}

std::tuple<at::Tensor&, at::Tensor&> fractional_max_pool2d_out_vulkan(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef output_size,
    const at::Tensor& random_samples,
    at::Tensor& output,
    at::Tensor& indices)
{
    auto [o, i] = fractional_max_pool(self, 2, kernel_size, output_size, random_samples, "fractional_max_pool2d");
    fill_out(output, o);
    fill_out(indices, i);
    return std::forward_as_tuple(output, indices);
}

std::tuple<at::Tensor&, at::Tensor&> fractional_max_pool3d_out_vulkan(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef output_size,
    const at::Tensor& random_samples,
    at::Tensor& output,
    at::Tensor& indices)
{
    auto [o, i] = fractional_max_pool(self, 3, kernel_size, output_size, random_samples, "fractional_max_pool3d");
    fill_out(output, o);
    fill_out(indices, i);
    return std::forward_as_tuple(output, indices);
}

at::Tensor& fractional_max_pool2d_backward_out_vulkan(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::IntArrayRef /* kernel_size */,
    at::IntArrayRef /* output_size */,
    const at::Tensor& indices,
    at::Tensor& grad_input)
{
    return fill_out(grad_input, max_pool_backward(grad_output, self, indices, 2));
}

at::Tensor fractional_max_pool3d_backward_vulkan(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::IntArrayRef /* kernel_size */,
    at::IntArrayRef /* output_size */,
    const at::Tensor& indices)
{
    return max_pool_backward(grad_output, self, indices, 3);
}

at::Tensor& fractional_max_pool3d_backward_out_vulkan(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef output_size,
    const at::Tensor& indices,
    at::Tensor& grad_input)
{
    return fill_out(grad_input, fractional_max_pool3d_backward_vulkan(grad_output, self, kernel_size, output_size, indices));
}

at::Tensor max_unpool(const at::Tensor& self, const at::Tensor& indices, at::IntArrayRef output_size, int64_t dims, const char* name)
{
    TORCH_CHECK(indices.scalar_type() == at::kLong, "elements in indices should be type int64 but got: ", indices.scalar_type());
    TORCH_CHECK(static_cast<int64_t>(output_size.size()) == dims, "There should be exactly ", dims, " elements (height, width) in output_size, but got ", output_size.size(), " elements.");
    TORCH_CHECK(self.dim() == dims + 1 || self.dim() == dims + 2, "Input to ", name, " should be a ", dims + 1, "d or ", dims + 2, "d Tensor, but got a tensor with ", self.dim(), " dimensions.");
    TORCH_CHECK(self.sizes() == indices.sizes(), "Expected shape of indices to be same as that of the input tensor (", self.sizes(), ") but got indices tensor with shape: ", indices.sizes());

    int64_t planes = planes_of(self, dims);
    int64_t outputs = 1;
    for (int64_t s : output_size) outputs *= s;
    std::vector<int64_t> shape(self.sizes().begin(), self.sizes().end() - dims);
    shape.insert(shape.end(), output_size.begin(), output_size.end());

    at::Tensor flat_indices = indices.reshape({planes, -1});
    if (flat_indices.numel() > 0) {
        bool invalid = at::logical_or(at::lt(flat_indices, 0), at::ge(flat_indices, outputs)).any().item<bool>();
        TORCH_CHECK(!invalid, "Found an invalid max index (output volumes are of size ", output_size, ")");
    }
    at::Tensor output = at::zeros({planes, outputs}, self.options());
    if (flat_indices.numel() > 0) output.scatter_(1, flat_indices, self.reshape({planes, -1}));
    return output.view(shape);
}

at::Tensor max_unpool2d_vulkan(const at::Tensor& self, const at::Tensor& indices, c10::SymIntArrayRef output_size)
{
    return max_unpool(self, indices, to_ints(output_size), 2, "max_unpool2d");
}

at::Tensor& max_unpool2d_out_vulkan(const at::Tensor& self, const at::Tensor& indices, c10::SymIntArrayRef output_size, at::Tensor& out)
{
    return fill_out(out, max_unpool2d_vulkan(self, indices, output_size));
}

at::Tensor max_unpool3d_vulkan(const at::Tensor& self, const at::Tensor& indices, c10::SymIntArrayRef output_size, at::IntArrayRef /* stride */, at::IntArrayRef /* padding */)
{
    return max_unpool(self, indices, to_ints(output_size), 3, "max_unpool3d");
}

at::Tensor& max_unpool3d_out_vulkan(
    const at::Tensor& self,
    const at::Tensor& indices,
    c10::SymIntArrayRef output_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::Tensor& out)
{
    return fill_out(out, max_unpool3d_vulkan(self, indices, output_size, stride, padding));
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("max_pool2d_with_indices.out", &max_pool2d_with_indices_out_vulkan);
    m.impl("max_pool3d_with_indices", &max_pool3d_with_indices_vulkan);
    m.impl("max_pool3d_with_indices.out", &max_pool3d_with_indices_out_vulkan);
    m.impl("max_pool2d_with_indices_backward.grad_input", &max_pool2d_with_indices_backward_out_vulkan);
    m.impl("max_pool3d_with_indices_backward", &max_pool3d_with_indices_backward_vulkan);
    m.impl("max_pool3d_with_indices_backward.grad_input", &max_pool3d_with_indices_backward_out_vulkan);
    m.impl("avg_pool2d.out", &avg_pool2d_out_vulkan);
    m.impl("avg_pool3d.out", &avg_pool3d_out_vulkan);
    m.impl("avg_pool2d_backward.grad_input", &avg_pool2d_backward_out_vulkan);
    m.impl("avg_pool3d_backward.grad_input", &avg_pool3d_backward_out_vulkan);
    m.impl("_adaptive_avg_pool2d", &adaptive_avg_pool2d_vulkan);
    m.impl("_adaptive_avg_pool3d", &adaptive_avg_pool3d_vulkan);
    m.impl("_adaptive_avg_pool2d_backward", &adaptive_avg_pool2d_backward_vulkan);
    m.impl("_adaptive_avg_pool3d_backward", &adaptive_avg_pool3d_backward_vulkan);
    m.impl("adaptive_avg_pool2d.out", &adaptive_avg_pool2d_out_vulkan);
    m.impl("adaptive_avg_pool3d.out", &adaptive_avg_pool3d_out_vulkan);
    m.impl("adaptive_avg_pool3d_backward.grad_input", &adaptive_avg_pool3d_backward_out_vulkan);
    m.impl("adaptive_max_pool2d.out", &adaptive_max_pool2d_out_vulkan);
    m.impl("adaptive_max_pool3d.out", &adaptive_max_pool3d_out_vulkan);
    m.impl("adaptive_max_pool2d_backward.grad_input", &adaptive_max_pool2d_backward_out_vulkan);
    m.impl("adaptive_max_pool3d_backward.grad_input", &adaptive_max_pool3d_backward_out_vulkan);
    m.impl("fractional_max_pool2d.output", &fractional_max_pool2d_out_vulkan);
    m.impl("fractional_max_pool3d.output", &fractional_max_pool3d_out_vulkan);
    m.impl("fractional_max_pool2d_backward.grad_input", &fractional_max_pool2d_backward_out_vulkan);
    m.impl("fractional_max_pool3d_backward", &fractional_max_pool3d_backward_vulkan);
    m.impl("fractional_max_pool3d_backward.grad_input", &fractional_max_pool3d_backward_out_vulkan);
    m.impl("max_unpool2d", &max_unpool2d_vulkan);
    m.impl("max_unpool2d.out", &max_unpool2d_out_vulkan);
    m.impl("max_unpool3d", &max_unpool3d_vulkan);
    m.impl("max_unpool3d.out", &max_unpool3d_out_vulkan);
}
