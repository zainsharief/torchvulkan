#include <torch/extension.h>
#include <ATen/WrapDimUtils.h>
#include <algorithm>
#include <vector>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "api/ops/helpers.h"
#include "api/ops/internal.h"
#include "api/fallback.h"

namespace {

enum class ScanOp { SUM = 0, PROD = 1, LOGADDEXP = 2 };

// complex scans (ops/complex.cpp): a sum scans the (real, imag) pairs; a product or logcumsumexp combines
// pairwise in log2(n) Hillis-Steele rounds
at::Tensor complex_cumscan(const at::Tensor& x, int64_t d, ScanOp op)
{
    if (op == ScanOp::SUM) return unview(at::view_as_complex(at::cumsum(at::view_as_real(x.resolve_conj().resolve_neg()), d).contiguous()));
    torchvulkan::ComplexParts z = torchvulkan::complex_parts(x);
    int64_t n = x.size(d);
    double identity_re = op == ScanOp::PROD ? 1.0 : -std::numeric_limits<double>::infinity();
    for (int64_t offset = 1; offset < n; offset *= 2) {
        std::vector<int64_t> pad_shape = z.re.sizes().vec();
        pad_shape[d] = offset;
        torchvulkan::ComplexParts shifted{
            at::cat({at::full(pad_shape, identity_re, z.re.options()), z.re.narrow(d, 0, n - offset)}, d),
            at::cat({at::zeros(pad_shape, z.im.options()), z.im.narrow(d, 0, n - offset)}, d)};
        z = op == ScanOp::PROD ? torchvulkan::complex_mul(z, shifted) : torchvulkan::complex_logaddexp(shifted, z);
    }
    return torchvulkan::complex_from(z, x.scalar_type());
}

at::Tensor cumscan(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype,
    ScanOp op)
{
    c10::ScalarType out_dtype = reduce_compute_dtype(self, dtype);
    c10::ScalarType compute = reduce_accumulate_dtype(out_dtype);

    int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    at::Tensor x = self.to(compute);
    if (x.dim() == 0 || x.numel() == 0) return x.clone().to(out_dtype);

    if (x.is_complex()) return complex_cumscan(x, d, op).to(out_dtype);
    if (!is_dtype_supported(x, compute) || x.dim() > MAX_DIMS || !fits_32bit_indexing({x})) {
        TORCHVULKAN_FALLBACK("A ", compute, " cumulative op of this shape can't run on the Vulkan device. Falling back to CPU.");
        at::Tensor cpu = op == ScanOp::SUM ? at::cumsum(x.cpu(), d) : op == ScanOp::PROD ? at::cumprod(x.cpu(), d) : at::logcumsumexp(x.cpu(), d);
        return cpu.to(out_dtype).to(self.device());
    }

    int64_t ndim = x.dim();
    int64_t scan_size = x.size(d);
    int64_t num_lines = scan_size > 0 ? x.numel() / scan_size : 0;
    at::Tensor out = at::empty(x.sizes(), x.options());
    if (num_lines == 0) return out.to(out_dtype);

    DeviceContext* device = get_device_context(x);
    uint32_t workgroupSizeX = std::min<uint32_t>(256u, device->properties.limits.maxComputeWorkGroupInvocations);
    ShaderKey shader_key{torchvulkan::Kernel::SCAN, compute};
    uint32_t opv = static_cast<uint32_t>(op);
    int32_t ndim32 = static_cast<int32_t>(ndim);

    SpecializationBuilder spd{};
    spd.push(opv)
       .push(ndim32)
       .push(workgroupSizeX);
    SpecializationArgs specialization = spd.build();

    std::vector<int64_t> sizes_kept(x.sizes().begin(), x.sizes().end());
    sizes_kept[d] = 1;

    MetadataBuilder metadataBuilder = get_shape_metadata(sizes_kept, {x.strides(), out.strides()});
    Metadata metadata = metadataBuilder.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(x))
       .push(get_output_address(out))
       .push_metadata(&metadata)
       .push(static_cast<uint64_t>(num_lines))
       .push(static_cast<uint32_t>(scan_size))
       .push(static_cast<uint32_t>(x.stride(d)))
       .push(static_cast<uint32_t>(out.stride(d)));

    // one workgroup per line - the shader itself distributes each line across workgroupSizeX threads
    uint32_t groupX = static_cast<uint32_t>(num_lines);

    device->shader_manager->dispatchShader(
        shader_key,
        device->default_subgroup_size,
        specialization,
        pcs.build(),
        /* read = */ {x},
        /* write = */ {out},
        groupX, 1, 1
    );

    return out.to(out_dtype);
}

at::Tensor& cumsum_out_vulkan(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype,
    at::Tensor& out)
{
    return fill_out(out, cumscan(self, dim, dtype, ScanOp::SUM));
}

at::Tensor& cumprod_out_vulkan(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype,
    at::Tensor& out)
{
    return fill_out(out, cumscan(self, dim, dtype, ScanOp::PROD));
}

at::Tensor logcumsumexp_vulkan(const at::Tensor& self, int64_t dim)
{
    if (!self.is_complex()) check_floating(self, "logcumsumexp");
    return cumscan(self, dim, self.scalar_type(), ScanOp::LOGADDEXP);
}

at::Tensor& logcumsumexp_out_vulkan(const at::Tensor& self, int64_t dim, at::Tensor& out)
{
    return fill_out(out, logcumsumexp_vulkan(self, dim));
}

enum class CumArgOp { MAX = 0, MIN = 1 };

void cumscanarg(const at::Tensor& self, at::Tensor& values, at::Tensor& indices, int64_t dim, CumArgOp op);

// past MAX_DIMS or 32-bit indexing: the scan dim moved last and the rest flattened into rows, scanned in pieces that fit.
// A row longer than a piece carries the running value across with torch's step: the later element wins on NaN, or on
// >= (<= for cummin) unless the running value is NaN
void cumscanarg_in_pieces(const at::Tensor& self, at::Tensor& values, at::Tensor& indices, int64_t d, CumArgOp op)
{
    at::Tensor moved = self.movedim(d, -1).contiguous();
    std::vector<int64_t> moved_shape = moved.sizes().vec();
    int64_t n = moved.size(-1);
    at::Tensor rows = moved.view({-1, n});
    at::Tensor val = at::empty_like(rows);
    at::Tensor idx = at::empty(rows.sizes(), rows.options().dtype(at::kLong));
    constexpr int64_t piece = int64_t(1) << 30;
    if (n <= piece) {
        int64_t step = std::max<int64_t>(1, piece / n);
        for (int64_t r = 0; r < rows.size(0); r += step) {
            int64_t count = std::min(step, rows.size(0) - r);
            at::Tensor v = val.narrow(0, r, count);
            at::Tensor i = idx.narrow(0, r, count);
            cumscanarg(rows.narrow(0, r, count), v, i, 1, op);
        }
    } else {
        for (int64_t r = 0; r < rows.size(0); r++) {
            at::Tensor carry_v;
            at::Tensor carry_i;
            for (int64_t start = 0; start < n; start += piece) {
                int64_t length = std::min(piece, n - start);
                at::Tensor v = val.select(0, r).narrow(0, start, length);
                at::Tensor i = idx.select(0, r).narrow(0, start, length);
                cumscanarg(rows.select(0, r).narrow(0, start, length), v, i, 0, op);
                i.add_(start);
                if (carry_v.defined()) {
                    at::Tensor wins = op == CumArgOp::MAX ? at::ge(v, carry_v) : at::le(v, carry_v);
                    at::Tensor later = at::logical_or(at::isnan(v), at::logical_and(at::logical_not(at::isnan(carry_v)), wins));
                    v.copy_(at::where(later, v, carry_v));
                    i.copy_(at::where(later, i, carry_i));
                }
                carry_v = v.narrow(0, length - 1, 1).clone();
                carry_i = i.narrow(0, length - 1, 1).clone();
            }
        }
    }
    values.copy_(val.view(moved_shape).movedim(-1, d));
    indices.copy_(idx.view(moved_shape).movedim(-1, d));
}

void cumscanarg(
    const at::Tensor& self,
    at::Tensor& values,
    at::Tensor& indices,
    int64_t dim,
    CumArgOp op)
{
    TORCH_CHECK_NOT_IMPLEMENTED(!self.is_complex(), "\"", op == CumArgOp::MAX ? "cummax" : "cummin", "\" not implemented for '",
        self.scalar_type(), "'");
    if (self.dim() == 0 || self.numel() == 0) {
        values.copy_(self);
        if (indices.numel() > 0) indices.zero_();
        return;
    }
    int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    if (is_dtype_supported(self, self.scalar_type()) && (self.dim() > MAX_DIMS || !fits_32bit_indexing({self}))) {
        cumscanarg_in_pieces(self, values, indices, d, op);
        return;
    }
    if (!is_dtype_supported(self, self.scalar_type())) {
        TORCHVULKAN_FALLBACK("A ", self.scalar_type(), op == CumArgOp::MAX ? " cummax" : " cummin", " can't run on the Vulkan device. Falling back to CPU.");
        auto r = op == CumArgOp::MAX ? at::cummax(self.cpu(), dim) : at::cummin(self.cpu(), dim);
        values.copy_(std::get<0>(r));
        indices.copy_(std::get<1>(r));
        return;
    }

    int64_t ndim = self.dim();
    int64_t scan_size = self.size(d);
    int64_t num_lines = scan_size > 0 ? self.numel() / scan_size : 0;
    at::Tensor val_t = at::empty(self.sizes(), self.options());
    at::Tensor idx_t = at::empty(self.sizes(), self.options().dtype(at::kLong));
    if (num_lines == 0) return;

    DeviceContext* device = get_device_context(self);
    uint32_t workgroupSizeX = std::min<uint32_t>(256u, device->properties.limits.maxComputeWorkGroupInvocations);
    ShaderKey shader_key{torchvulkan::Kernel::SCAN_ARG, self.scalar_type()};
    uint32_t opv = static_cast<uint32_t>(op);
    int32_t ndim32 = static_cast<int32_t>(ndim);

    SpecializationBuilder spd{};
    spd.push(opv)
       .push(ndim32)
       .push(workgroupSizeX);
    SpecializationArgs specialization = spd.build();

    std::vector<int64_t> sizes_kept(self.sizes().begin(), self.sizes().end());
    sizes_kept[d] = 1;

    MetadataBuilder metadataBuilder = get_shape_metadata(sizes_kept, {self.strides(), val_t.strides()});
    Metadata metadata = metadataBuilder.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(self))
       .push(get_output_address(val_t))
       .push(get_output_address(idx_t))
       .push_metadata(&metadata)
       .push(static_cast<uint64_t>(num_lines))
       .push(static_cast<uint32_t>(scan_size))
       .push(static_cast<uint32_t>(self.stride(d)))
       .push(static_cast<uint32_t>(val_t.stride(d)));

    // one workgroup per line - the shader itself distributes each line across workgroupSizeX threads
    uint32_t groupX = static_cast<uint32_t>(num_lines);

    device->shader_manager->dispatchShader(
        shader_key,
        device->default_subgroup_size,
        specialization,
        pcs.build(),
        /* read = */ {self},
        /* write = */ {val_t, idx_t},
        groupX, 1, 1
    );

    values.copy_(val_t);
    indices.copy_(idx_t);
}

void cummax_helper_vulkan(
    const at::Tensor& self,
    at::Tensor& values,
    at::Tensor& indices,
    int64_t dim)
{
    cumscanarg(normalize_bool(self), values, indices, dim, CumArgOp::MAX);
}

void cummin_helper_vulkan(
    const at::Tensor& self,
    at::Tensor& values,
    at::Tensor& indices,
    int64_t dim)
{
    cumscanarg(normalize_bool(self), values, indices, dim, CumArgOp::MIN);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("cumsum.out", &cumsum_out_vulkan);
    m.impl("cumprod.out", &cumprod_out_vulkan);
    m.impl("_logcumsumexp", &logcumsumexp_vulkan);
    m.impl("_logcumsumexp.out", &logcumsumexp_out_vulkan);
    m.impl("_cummax_helper", &cummax_helper_vulkan);
    m.impl("_cummin_helper", &cummin_helper_vulkan);
}
