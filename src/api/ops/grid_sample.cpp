#include <torch/extension.h>
#include <array>
#include <tuple>
#include <vector>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "api/ops/helpers.h"

namespace {

enum class GridSampleMode {
    FORWARD = 0,
    TAPS = 1,
    GRID_GRAD = 2,
    INPUT_GRAD = 3
};

struct GridSampleProblem {
    at::Tensor input;
    at::Tensor grid;
    int64_t dims;
    int64_t n;
    int64_t channels;
    std::array<int64_t, 3> in{1, 1, 1};
    std::array<int64_t, 3> out{1, 1, 1};
};

at::ScalarType grid_sample_dtype(const at::Tensor& input)
{
    return (input.scalar_type() == at::kDouble && is_dtype_supported(input, at::kDouble)) ? at::kDouble : at::kFloat;
}

GridSampleProblem make_problem(const at::Tensor& input, const at::Tensor& grid, int64_t dims)
{
    TORCH_CHECK(input.dim() == dims + 2, "grid_sampler(): expected ", dims + 2, "D input, but got input with sizes ", input.sizes());
    TORCH_CHECK(grid.dim() == dims + 2 && grid.size(-1) == dims, "grid_sampler(): expected grid to have size ", dims, " in last dimension, but got grid with sizes ", grid.sizes());
    TORCH_CHECK(input.size(0) == grid.size(0), "grid_sampler(): expected grid and input to have same batch size, but got input with sizes ", input.sizes(),
        " and grid with sizes ", grid.sizes()
    );
    at::ScalarType dtype = grid_sample_dtype(input);
    GridSampleProblem p;
    p.input = input.to(dtype).contiguous();
    p.grid = grid.to(dtype).contiguous();
    p.dims = dims;
    p.n = input.size(0);
    p.channels = input.size(1);
    for (int64_t a = 0; a < dims; a++) {
        p.in[3 - dims + a] = input.size(2 + a);
        p.out[3 - dims + a] = grid.size(1 + a);
    }
    return p;
}

int64_t outputs_of(const GridSampleProblem& p) { return p.out[0] * p.out[1] * p.out[2]; }

int64_t tap_count(int64_t interpolation, int64_t dims)
{
    if (interpolation == 1) return 1;
    if (interpolation == 2) return dims == 3 ? 64 : 16;
    return dims == 3 ? 8 : 4;
}

void dispatch_grid_sample_shader(
    const GridSampleProblem& p,
    const at::Tensor& output,
    const at::Tensor& grad_output,
    const at::Tensor& taps,
    const at::Tensor& weights,
    const at::Tensor& starts,
    GridSampleMode mode,
    int64_t interpolation,
    int64_t padding,
    bool align_corners)
{
    uint64_t count = mode == GridSampleMode::INPUT_GRAD ? static_cast<uint64_t>(p.input.numel()) : static_cast<uint64_t>(p.n * outputs_of(p));
    if (count == 0) return;
    DeviceContext* device = get_device_context(p.input);
    at::ScalarType dtype = p.input.scalar_type();
    uint32_t workgroupSizeX = get_dtype_workgroup_size(dtype, 1, device);

    uint32_t params[8] = {
        static_cast<uint32_t>(p.n), static_cast<uint32_t>(p.channels),
        static_cast<uint32_t>(p.in[0]), static_cast<uint32_t>(p.in[1]), static_cast<uint32_t>(p.in[2]),
        static_cast<uint32_t>(p.out[0]), static_cast<uint32_t>(p.out[1]), static_cast<uint32_t>(p.out[2]),
    };
    MetadataBuilder metadataBuilder{};
    metadataBuilder.push_array(params, 8);
    Metadata metadata = metadataBuilder.build();

    SpecializationBuilder spd{};
    spd.push(static_cast<uint32_t>(mode))
       .push(static_cast<uint32_t>(p.dims))
       .push(static_cast<uint32_t>(interpolation))
       .push(static_cast<uint32_t>(padding))
       .push(static_cast<uint32_t>(align_corners))
       .push(workgroupSizeX);

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(p.input))
       .push(get_tensor_address(p.grid))
       .push(output.defined() ? get_tensor_address(output) : uint64_t(0))
       .push(grad_output.defined() ? get_tensor_address(grad_output) : uint64_t(0))
       .push(taps.defined() ? get_tensor_address(taps) : uint64_t(0))
       .push(weights.defined() ? get_tensor_address(weights) : uint64_t(0))
       .push(starts.defined() ? get_tensor_address(starts) : uint64_t(0))
       .push_metadata(&metadata)
       .push(count);

    std::vector<at::Tensor> reads = {p.input, p.grid};
    if (grad_output.defined()) reads.push_back(grad_output);
    std::vector<at::Tensor> writes;
    if (output.defined()) writes.push_back(output);
    if (mode == GridSampleMode::INPUT_GRAD) {
        reads.push_back(taps);
        reads.push_back(weights);
        reads.push_back(starts);
    } else {
        if (taps.defined()) writes.push_back(taps);
        if (weights.defined()) writes.push_back(weights);
    }

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::GRID_SAMPLE, dtype},
        spd.build(),
        pcs.build(),
        /* read = */ reads,
        /* write = */ writes,
        get_group_count(count, workgroupSizeX), 1, 1
    );
}

std::vector<int64_t> output_shape(const GridSampleProblem& p)
{
    std::vector<int64_t> shape = {p.n, p.channels};
    for (int64_t a = 3 - p.dims; a < 3; a++) shape.push_back(p.out[a]);
    return shape;
}

at::Tensor grid_sampler_forward(const at::Tensor& input, const at::Tensor& grid, int64_t dims, int64_t interpolation, int64_t padding, bool align_corners)
{
    GridSampleProblem p = make_problem(input, grid, dims);
    at::Tensor output = at::empty(output_shape(p), p.input.options());
    if (p.channels > 0) dispatch_grid_sample_shader(p, output, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), GridSampleMode::FORWARD, interpolation, padding, align_corners);
    return output.to(input.scalar_type());
}

// sorted stably so each input position sums in the CPU's order
std::tuple<at::Tensor, at::Tensor> grid_sampler_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& grid,
    int64_t dims,
    int64_t interpolation,
    int64_t padding,
    bool align_corners,
    std::array<bool, 2> output_mask)
{
    GridSampleProblem p = make_problem(input, grid, dims);
    at::Tensor go = grad_output.to(p.input.scalar_type()).contiguous();
    int64_t outputs = outputs_of(p);
    int64_t inputs = p.in[0] * p.in[1] * p.in[2];

    at::Tensor grad_input, grad_grid;
    if (output_mask[0]) {
        int64_t k = tap_count(interpolation, dims);
        at::Tensor taps = at::empty({p.n, outputs, k}, p.input.options().dtype(at::kLong));
        at::Tensor weights = at::empty({p.n, outputs, k}, p.input.options());
        dispatch_grid_sample_shader(p, at::Tensor(), at::Tensor(), taps, weights, at::Tensor(), GridSampleMode::TAPS, interpolation, padding, align_corners);

        int64_t positions = p.n * inputs;
        at::TensorOptions long_options = taps.options();
        at::Tensor batch_base = at::mul(at::arange(p.n, long_options), inputs).view({p.n, 1, 1});
        at::Tensor keys = at::where(at::ge(taps, 0), at::add(batch_base, taps), at::full({}, positions, long_options)).reshape({-1});
        auto [sorted, order] = at::sort(keys, /*stable=*/true, 0, /*descending=*/false);
        at::Tensor starts = at::searchsorted(sorted, at::arange(positions + 1, long_options));
        grad_input = at::empty(p.input.sizes(), p.input.options());
        if (grad_input.numel() > 0) {
            dispatch_grid_sample_shader(p, grad_input, go, order, weights, starts, GridSampleMode::INPUT_GRAD, interpolation, padding, align_corners);
        }
        grad_input = grad_input.to(input.scalar_type());
    }
    if (output_mask[1]) {
        grad_grid = at::empty(p.grid.sizes(), p.grid.options());
        if (p.channels > 0) dispatch_grid_sample_shader(p, grad_grid, go, at::Tensor(), at::Tensor(), at::Tensor(), GridSampleMode::GRID_GRAD, interpolation, padding, align_corners);
        else grad_grid.zero_();
        grad_grid = grad_grid.to(grid.scalar_type());
    }
    return std::make_tuple(grad_input, grad_grid);
}

at::Tensor grid_sampler_2d_vulkan(const at::Tensor& input, const at::Tensor& grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners)
{
    return grid_sampler_forward(input, grid, 2, interpolation_mode, padding_mode, align_corners);
}

at::Tensor grid_sampler_3d_vulkan(const at::Tensor& input, const at::Tensor& grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners)
{
    return grid_sampler_forward(input, grid, 3, interpolation_mode, padding_mode, align_corners);
}

std::tuple<at::Tensor, at::Tensor> grid_sampler_2d_backward_vulkan(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    std::array<bool, 2> output_mask)
{
    return grid_sampler_backward(grad_output, input, grid, 2, interpolation_mode, padding_mode, align_corners, output_mask);
}

std::tuple<at::Tensor, at::Tensor> grid_sampler_3d_backward_vulkan(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    std::array<bool, 2> output_mask)
{
    return grid_sampler_backward(grad_output, input, grid, 3, interpolation_mode, padding_mode, align_corners, output_mask);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("grid_sampler_2d", &grid_sampler_2d_vulkan);
    m.impl("grid_sampler_3d", &grid_sampler_3d_vulkan);
    m.impl("grid_sampler_2d_backward", &grid_sampler_2d_backward_vulkan);
    m.impl("grid_sampler_3d_backward", &grid_sampler_3d_backward_vulkan);
}
