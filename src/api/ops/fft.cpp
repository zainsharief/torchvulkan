#include <torch/extension.h>
#include <ATen/WrapDimUtils.h>
#include <cmath>
#include <vector>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "api/ops/helpers.h"

namespace {

enum class FftMode {
    STAGE = 0,
    CHIRP_IN = 1,
    CHIRP_KERNEL = 2,
    POINTWISE = 3,
    CHIRP_OUT = 4
};

enum class FftNorm {
    NONE = 0,
    BY_ROOT_N = 1,
    BY_N = 2
};

void dispatch_fft_shader(
    FftMode mode,
    bool inverse,
    const at::Tensor& input,
    const at::Tensor& output,
    const at::Tensor& aux,
    uint64_t count,
    int64_t n,
    int64_t m,
    int64_t half_span)
{
    if (count == 0) return;
    DeviceContext* device = get_device_context(output);
    at::ScalarType real = output.scalar_type() == at::kComplexDouble ? at::kDouble : at::kFloat;
    uint32_t workgroupSizeX = get_dtype_workgroup_size(real, 1, device);

    SpecializationBuilder spd{};
    spd.push(static_cast<uint32_t>(mode))
       .push(static_cast<uint32_t>(inverse))
       .push(workgroupSizeX);

    PushConstantBuilder pcs{};
    pcs.push(input.defined() ? get_tensor_address(input) : uint64_t(0))
       .push(get_tensor_address(output))
       .push(aux.defined() ? get_tensor_address(aux) : uint64_t(0))
       .push(count)
       .push(static_cast<uint32_t>(n))
       .push(static_cast<uint32_t>(m))
       .push(static_cast<uint32_t>(half_span))
       .push(uint32_t(0));
    std::vector<at::Tensor> reads;
    if (input.defined()) reads.push_back(input);
    if (aux.defined()) reads.push_back(aux);

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::FFT, real},
        device->default_subgroup_size,
        spd.build(),
        pcs.build(),
        /* read = */ reads,
        /* write = */ {output},
        get_group_count(count, workgroupSizeX), 1, 1
    );
}

bool is_power_of_two(int64_t n) { return n > 0 && (n & (n - 1)) == 0; }

at::Tensor stockham(const at::Tensor& rows, bool inverse)
{
    int64_t n = rows.size(1);
    at::Tensor current = rows;
    for (int64_t span = 1; span < n; span *= 2) {
        at::Tensor next = at::empty_like(rows);
        dispatch_fft_shader(FftMode::STAGE, inverse, current, next, at::Tensor(), static_cast<uint64_t>(rows.size(0) * (n / 2)), n, n, span);
        current = next;
    }
    return current.is_same(rows) ? rows.clone() : current;
}

at::Tensor bluestein(const at::Tensor& rows, bool inverse)
{
    int64_t n = rows.size(1);
    int64_t m = 1;
    while (m < 2 * n - 1) m *= 2;
    at::Tensor a = at::empty({rows.size(0), m}, rows.options());
    dispatch_fft_shader(FftMode::CHIRP_IN, inverse, rows, a, at::Tensor(), static_cast<uint64_t>(rows.size(0) * m), n, m, 0);
    at::Tensor b = at::empty({1, m}, rows.options());
    dispatch_fft_shader(FftMode::CHIRP_KERNEL, inverse, at::Tensor(), b, at::Tensor(), static_cast<uint64_t>(m), n, m, 0);

    at::Tensor fa = stockham(a, false);
    at::Tensor fb = stockham(b, false);
    at::Tensor product = at::empty_like(fa);
    dispatch_fft_shader(FftMode::POINTWISE, inverse, fa, product, fb, static_cast<uint64_t>(rows.size(0) * m), n, m, 0);
    at::Tensor convolved = stockham(product, true);
    at::Tensor out = at::empty_like(rows);
    dispatch_fft_shader(FftMode::CHIRP_OUT, inverse, convolved, out, at::Tensor(), static_cast<uint64_t>(rows.size(0) * n), n, m, 0);
    return out;
}

at::ScalarType fft_dtype(const at::Tensor& self)
{
    return (self.scalar_type() == at::kComplexDouble && is_dtype_supported(self, at::kDouble)) ? at::kComplexDouble : at::kComplexFloat;
}

at::Tensor transform_dim(const at::Tensor& self, int64_t dim, bool inverse)
{
    int64_t n = self.size(dim);
    if (n <= 1 || self.numel() == 0) return self.clone();
    at::Tensor moved = self.movedim(dim, -1).contiguous();
    std::vector<int64_t> shape = moved.sizes().vec();
    at::Tensor rows = moved.view({-1, n});
    at::Tensor result = is_power_of_two(n) ? stockham(rows, inverse) : bluestein(rows, inverse);
    return result.view(shape).movedim(-1, dim);
}

double norm_scale(int64_t normalization, int64_t elements)
{
    if (normalization == static_cast<int64_t>(FftNorm::BY_ROOT_N)) return 1.0 / std::sqrt(static_cast<double>(elements));
    if (normalization == static_cast<int64_t>(FftNorm::BY_N)) return 1.0 / static_cast<double>(elements);
    return 1.0;
}

at::Tensor apply_scale(const at::Tensor& x, double scale)
{
    if (scale == 1.0) return x;
    at::Tensor result = x.clone();
    at::view_as_real(result).mul_(scale);
    return result;
}

at::Tensor c2c(const at::Tensor& self, at::IntArrayRef dims, int64_t normalization, bool forward)
{
    TORCH_CHECK(self.is_complex(), "_fft_c2c expects a complex input tensor, but got ", self.scalar_type());
    at::Tensor x = self.resolve_conj().to(fft_dtype(self));
    int64_t elements = 1;
    for (int64_t d : dims) {
        int64_t w = c10::maybe_wrap_dim(d, x.dim());
        elements *= x.size(w);
        x = transform_dim(x, w, !forward);
    }
    return apply_scale(x, norm_scale(normalization, elements)).to(self.scalar_type());
}

at::Tensor fft_c2c_vulkan(const at::Tensor& self, c10::SymIntArrayRef dim, int64_t normalization, bool forward)
{
    std::vector<int64_t> dims;
    for (const c10::SymInt& d : dim) dims.push_back(d.guard_int(__FILE__, __LINE__));
    return c2c(self, dims, normalization, forward);
}

at::Tensor& fft_c2c_out_vulkan(const at::Tensor& self, c10::SymIntArrayRef dim, int64_t normalization, bool forward, at::Tensor& out)
{
    return fill_out(out, fft_c2c_vulkan(self, dim, normalization, forward));
}

at::Tensor fft_r2c_vulkan(const at::Tensor& self, at::IntArrayRef dim, int64_t normalization, bool onesided)
{
    TORCH_CHECK(self.is_floating_point(), "_fft_r2c expects a real input tensor, but got ", self.scalar_type());
    at::Tensor x = at::empty(self.sizes(), self.options().dtype(get_dtype_complex(self.scalar_type())));
    at::Tensor pairs = at::view_as_real(x);
    pairs.select(-1, 0).copy_(self);
    pairs.select(-1, 1).zero_();
    at::Tensor result = c2c(x, dim, normalization, true);
    if (onesided && !dim.empty()) {
        int64_t last = c10::maybe_wrap_dim(dim.back(), self.dim());
        result = result.narrow(last, 0, self.size(last) / 2 + 1).contiguous();
    }
    return result;
}

at::Tensor& fft_r2c_out_vulkan(const at::Tensor& self, at::IntArrayRef dim, int64_t normalization, bool onesided, at::Tensor& out)
{
    return fill_out(out, fft_r2c_vulkan(self, dim, normalization, onesided));
}

at::Tensor fft_c2r_vulkan(const at::Tensor& self, at::IntArrayRef dim, int64_t normalization, c10::SymInt last_dim_size_)
{
    TORCH_CHECK(self.is_complex(), "_fft_c2r expects a complex input tensor, but got ", self.scalar_type());
    TORCH_CHECK(!dim.empty(), "_fft_c2r expects at least one dim");
    int64_t length = last_dim_size_.guard_int(__FILE__, __LINE__);
    TORCH_CHECK(length >= 1, "Invalid number of data points (", length, ") specified");
    at::Tensor x = self.resolve_conj().to(fft_dtype(self));
    int64_t last = c10::maybe_wrap_dim(dim.back(), x.dim());

    int64_t elements = length;
    for (size_t i = 0; i + 1 < dim.size(); i++) {
        int64_t w = c10::maybe_wrap_dim(dim[i], x.dim());
        elements *= x.size(w);
        x = transform_dim(x, w, true);
    }

    int64_t stored = std::min<int64_t>(x.size(last), length / 2 + 1);
    at::Tensor half = x.narrow(last, 0, stored);
    at::Tensor full = half;
    int64_t mirrored = length - stored;
    if (mirrored > 0) {
        std::vector<int64_t> source(mirrored);
        for (int64_t i = 0; i < mirrored; i++) source[i] = mirrored - i;
        at::Tensor index = at::tensor(source, at::kLong).to(x.device());
        at::Tensor tail = at::index_select(x, last, index).clone();
        at::view_as_real(tail).select(-1, 1).neg_();
        full = at::cat({half, tail}, last);
    }
    at::Tensor result = transform_dim(full, last, true);
    at::Tensor real = at::view_as_real(result).select(-1, 0).mul(norm_scale(normalization, elements));
    return real.to(self.scalar_type() == at::kComplexDouble ? at::kDouble : at::kFloat).contiguous();
}

at::Tensor& fft_c2r_out_vulkan(const at::Tensor& self, at::IntArrayRef dim, int64_t normalization, c10::SymInt last_dim_size, at::Tensor& out)
{
    return fill_out(out, fft_c2r_vulkan(self, dim, normalization, last_dim_size));
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("_fft_c2c", &fft_c2c_vulkan);
    m.impl("_fft_c2c.out", &fft_c2c_out_vulkan);
    m.impl("_fft_r2c", &fft_r2c_vulkan);
    m.impl("_fft_r2c.out", &fft_r2c_out_vulkan);
    m.impl("_fft_c2r", &fft_c2r_vulkan);
    m.impl("_fft_c2r.out", &fft_c2r_out_vulkan);
}
