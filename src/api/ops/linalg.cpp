#include <torch/extension.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/ExpandUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/ops/linalg_matrix_sqrth_native.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <tuple>
#include <vector>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "api/ops/helpers.h"
#include "api/ops/internal.h"
#include "api/fallback.h"

namespace {

enum class HouseholderMode {
    GEQRF = 0,
    ORGQR = 1,
    ORMQR = 2
};

enum class JacobiMode {
    SVD = 0,
    EIGH = 1
};

enum class LdlMode {
    FACTOR = 0,
    SOLVE = 1
};

at::ScalarType linalg_dtype(const at::Tensor& t, const char* name)
{
    TORCH_CHECK(at::isFloatingType(t.scalar_type()), name, ": Expected a floating point tensor as input. Got ", t.scalar_type());
    return (t.scalar_type() == at::kDouble && is_dtype_supported(t, at::kDouble)) ? at::kDouble : at::kFloat;
}

void check_square(const at::Tensor& A, const char* name)
{
    TORCH_CHECK(A.dim() >= 2, name, ": The input tensor A must have at least 2 dimensions.");
    TORCH_CHECK(A.size(-1) == A.size(-2), name, ": A must be batches of square matrices, but they are ", A.size(-2), " by ", A.size(-1), " matrices");
}

// a structured op's check on an out= tensor's dtype
void check_out_dtype(const at::Tensor& out, at::ScalarType expected)
{
    TORCH_CHECK(out.scalar_type() == expected, "Expected out tensor to have dtype ", c10::scalarTypeToTypeMeta(expected), ", but got ",
                out.dtype(), " instead");
}

// checkLinalgCompatibleDtype: an out= tensor must be able to take the result
void check_castable(const char* fn, const char* name, const at::Tensor& out, at::ScalarType result)
{
    TORCH_CHECK(at::canCast(result, out.scalar_type()), fn, ": Expected ", name, " to be safely castable from ", result, " dtype, but got ", name,
                " with dtype ", out.scalar_type());
}

std::vector<int64_t> batch_shape(const at::Tensor& A)
{
    return std::vector<int64_t>(A.sizes().begin(), A.sizes().end() - 2);
}

// the number of matrices in a batch (a -1 reshape can't infer it when the batch is empty)
int64_t batch_count(const at::Tensor& A)
{
    int64_t count = 1;
    for (int64_t s : batch_shape(A)) count *= s;
    return count;
}

std::vector<int64_t> with_matrix(std::vector<int64_t> batch, int64_t rows, int64_t cols)
{
    batch.push_back(rows);
    batch.push_back(cols);
    return batch;
}

at::Tensor batched_copy(const at::Tensor& A, at::ScalarType dtype)
{
    int64_t batches = 1;
    for (int64_t d = 0; d < A.dim() - 2; d++) batches *= A.size(d);
    at::Tensor out = at::empty({batches, A.size(-2), A.size(-1)}, A.options().dtype(dtype));
    if (out.numel() > 0) out.copy_(A.reshape({batches, A.size(-2), A.size(-1)}));
    return out;
}

uint32_t matrix_workgroup(const DeviceContext* device)
{
    return std::min<uint32_t>(256u, device->properties.limits.maxComputeWorkGroupInvocations);
}

void for_each_chunk(int64_t batches, const DeviceContext* device, const std::function<void(int64_t, int64_t)>& fn)
{
    int64_t limit = device->properties.limits.maxComputeWorkGroupCount[0];
    for (int64_t start = 0; start < batches; start += limit) fn(start, std::min(limit, batches - start));
}

void dispatch_cholesky_shader(const at::Tensor& a, const at::Tensor& info)
{
    DeviceContext* device = get_device_context(a);
    uint32_t workgroupSizeX = matrix_workgroup(device);
    for_each_chunk(a.size(0), device, [&](int64_t start, int64_t count) {
        at::Tensor a_part = a.narrow(0, start, count);
        at::Tensor info_part = info.narrow(0, start, count);

        SpecializationBuilder spd{};
        spd.push(workgroupSizeX);

        PushConstantBuilder pcs{};
        pcs.push(get_output_address(a_part))
           .push(get_output_address(info_part))
           .push(static_cast<uint32_t>(a.size(1)));

        device->shader_manager->dispatchShader(
            ShaderKey{torchvulkan::Kernel::CHOLESKY, a.scalar_type()},
            device->default_subgroup_size,
            spd.build(),
            pcs.build(),
            /* read = */ {a_part},
            /* write = */ {a_part, info_part},
            static_cast<uint32_t>(count), 1, 1
        );
    });
}

void dispatch_lu_shader(const at::Tensor& a, const at::Tensor& pivots)
{
    DeviceContext* device = get_device_context(a);
    uint32_t workgroupSizeX = matrix_workgroup(device);
    for_each_chunk(a.size(0), device, [&](int64_t start, int64_t count) {
        at::Tensor a_part = a.narrow(0, start, count);
        at::Tensor pivots_part = pivots.narrow(0, start, count);

        SpecializationBuilder spd{};
        spd.push(workgroupSizeX);

        PushConstantBuilder pcs{};
        pcs.push(get_tensor_address(a_part))
           .push(pivots.numel() > 0 ? get_tensor_address(pivots_part) : uint64_t(0))
           .push(static_cast<uint32_t>(a.size(1)))
           .push(static_cast<uint32_t>(a.size(2)));
        std::vector<at::Tensor> writes = {a_part};
        if (pivots.numel() > 0) writes.push_back(pivots_part);

        device->shader_manager->dispatchShader(
            ShaderKey{torchvulkan::Kernel::LU, a.scalar_type()},
            device->default_subgroup_size,
            spd.build(),
            pcs.build(),
            /* read = */ {a_part},
            /* write = */ writes,
            static_cast<uint32_t>(count), 1, 1
        );
    });
}

void dispatch_triangular_solve_shader(const at::Tensor& a, const at::Tensor& x, bool upper, bool unit)
{
    uint64_t count = static_cast<uint64_t>(x.size(0) * x.size(2));
    if (count == 0 || x.size(1) == 0) return;
    DeviceContext* device = get_device_context(x);
    uint32_t workgroupSizeX = get_dtype_workgroup_size(x.scalar_type(), 1, device);

    SpecializationBuilder spd{};
    spd.push(static_cast<uint32_t>(upper))
       .push(static_cast<uint32_t>(unit))
       .push(workgroupSizeX);

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(a))
       .push(get_output_address(x))
       .push(count)
       .push(static_cast<uint32_t>(x.size(1)))
       .push(static_cast<uint32_t>(x.size(2)));

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::TRIANGULAR_SOLVE, x.scalar_type()},
        device->default_subgroup_size,
        spd.build(),
        pcs.build(),
        /* read = */ {a},
        /* write = */ {x},
        get_group_count(count, workgroupSizeX), 1, 1
    );
}

void dispatch_householder_shader(HouseholderMode mode, const at::Tensor& a, const at::Tensor& tau, const at::Tensor& c, bool left, bool transpose)
{
    DeviceContext* device = get_device_context(a);
    uint32_t workgroupSizeX = matrix_workgroup(device);
    for_each_chunk(a.size(0), device, [&](int64_t start, int64_t count) {
        at::Tensor a_part = a.narrow(0, start, count);
        at::Tensor tau_part = tau.narrow(0, start, count);
        at::Tensor c_part = c.defined() ? c.narrow(0, start, count) : at::Tensor();

        SpecializationBuilder spd{};
        spd.push(static_cast<uint32_t>(mode))
           .push(static_cast<uint32_t>(left))
           .push(static_cast<uint32_t>(transpose))
           .push(workgroupSizeX);

        PushConstantBuilder pcs{};
        pcs.push(get_tensor_address(a_part))
           .push(tau.numel() > 0 ? get_tensor_address(tau_part) : uint64_t(0))
           .push(c.defined() ? get_tensor_address(c_part) : uint64_t(0))
           .push(static_cast<uint32_t>(a.size(1)))
           .push(static_cast<uint32_t>(a.size(2)))
           .push(static_cast<uint32_t>(tau.size(1)))
           .push(static_cast<uint32_t>(c.defined() ? c.size(1) : 0))
           .push(static_cast<uint32_t>(c.defined() ? c.size(2) : 0));
        std::vector<at::Tensor> reads = {a_part};
        if (tau.numel() > 0) reads.push_back(tau_part);
        std::vector<at::Tensor> writes;
        if (mode == HouseholderMode::ORMQR) writes = {c_part};
        else writes = {a_part};
        if (mode == HouseholderMode::GEQRF && tau.numel() > 0) writes.push_back(tau_part);

        device->shader_manager->dispatchShader(
            ShaderKey{torchvulkan::Kernel::HOUSEHOLDER, a.scalar_type()},
            device->default_subgroup_size,
            spd.build(),
            pcs.build(),
            /* read = */ reads,
            /* write = */ writes,
            static_cast<uint32_t>(count), 1, 1
        );
    });
}

// a complex a and v (SVD only) go to the shader as (re, im) pairs
void dispatch_jacobi_shader(JacobiMode mode, const at::Tensor& a, const at::Tensor& v)
{
    int64_t n = a.size(2);
    int64_t pairs = (n + 1) / 2;
    at::ScalarType real = c10::toRealValueType(a.scalar_type());
    // per pair: the rotation's cos and sin, and the phase it divides out
    at::Tensor rotations = at::empty({a.size(0), std::max<int64_t>(pairs, 1), 4}, a.options().dtype(real));
    DeviceContext* device = get_device_context(a);
    uint32_t workgroupSizeX = matrix_workgroup(device);
    for_each_chunk(a.size(0), device, [&](int64_t start, int64_t count) {
        at::Tensor a_part = a.narrow(0, start, count);
        at::Tensor v_part = v.narrow(0, start, count);
        at::Tensor r_part = rotations.narrow(0, start, count);

        SpecializationBuilder spd{};
        spd.push(static_cast<uint32_t>(mode))
           .push(workgroupSizeX)
           .push(static_cast<uint32_t>(a.is_complex()));

        PushConstantBuilder pcs{};
        pcs.push(get_output_address(a_part))
           .push(get_output_address(v_part))
           .push(get_output_address(r_part))
           .push(static_cast<uint32_t>(a.size(1)))
           .push(static_cast<uint32_t>(n))
           .push(static_cast<uint32_t>(pairs));

        device->shader_manager->dispatchShader(
            ShaderKey{torchvulkan::Kernel::JACOBI, real},
            device->default_subgroup_size,
            spd.build(),
            pcs.build(),
            /* read = */ {a_part},
            /* write = */ {a_part, v_part, r_part},
            static_cast<uint32_t>(count), 1, 1
        );
    });
}

// a complex a and b go to the shader as (re, im) pairs
void dispatch_ldl_shader(LdlMode mode, const at::Tensor& a, const at::Tensor& pivots, const at::Tensor& info, const at::Tensor& b, bool hermitian)
{
    at::ScalarType real = c10::toRealValueType(a.scalar_type());
    int64_t n = a.size(1);
    int64_t k = b.defined() ? b.size(2) : 1;
    uint64_t count = static_cast<uint64_t>(mode == LdlMode::FACTOR ? a.size(0) : b.size(0) * k);
    if (count == 0 || n == 0) return;
    DeviceContext* device = get_device_context(a);
    uint32_t workgroupSizeX = get_dtype_workgroup_size(real, 1, device);

    SpecializationBuilder spd{};
    spd.push(static_cast<uint32_t>(mode))
       .push(workgroupSizeX)
       .push(static_cast<uint32_t>(a.is_complex()))
       .push(static_cast<uint32_t>(hermitian));
    at::Tensor scratch = at::empty({static_cast<int64_t>(count), 4}, a.options().dtype(real));

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(a))
       .push(get_tensor_address(pivots))
       .push(info.defined() ? get_tensor_address(info) : uint64_t(0))
       .push(b.defined() ? get_tensor_address(b) : uint64_t(0))
       .push(get_tensor_address(scratch))
       .push(count)
       .push(static_cast<uint32_t>(n))
       .push(static_cast<uint32_t>(k));
    std::vector<at::Tensor> writes = {scratch};
    if (mode == LdlMode::FACTOR) writes.insert(writes.end(), {a, pivots, info});
    else writes.push_back(b);

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::LDL, real},
        device->default_subgroup_size,
        spd.build(),
        pcs.build(),
        /* read = */ {a, pivots},
        /* write = */ writes,
        get_group_count(count, workgroupSizeX), 1, 1
    );
}

// one thread per workgroup, as eig.slang's barriers require
void dispatch_eig_shader(const at::Tensor& h, const at::Tensor& v, const at::Tensor& d, const at::Tensor& e)
{
    if (h.size(0) == 0 || h.size(1) == 0) return;
    // per matrix: Householder scalars, gebal's scaling, and two elements of swap scratch
    at::Tensor ort = at::empty({h.size(0), 2 * h.size(1) + 2}, h.options());
    DeviceContext* device = get_device_context(h);
    for_each_chunk(h.size(0), device, [&](int64_t start, int64_t count) {
        auto part = [&](const at::Tensor& t) { return t.narrow(0, start, count); };

        SpecializationBuilder spd{};
        spd.push(uint32_t(1));

        PushConstantBuilder pcs{};
        pcs.push(get_tensor_address(part(h)))
           .push(get_tensor_address(part(v)))
           .push(get_tensor_address(part(d)))
           .push(get_tensor_address(part(e)))
           .push(get_tensor_address(part(ort)))
           .push(static_cast<uint64_t>(count))
           .push(static_cast<uint32_t>(h.size(1)));

        device->shader_manager->dispatchShader(
            ShaderKey{torchvulkan::Kernel::EIG, h.scalar_type()},
            device->default_subgroup_size,
            spd.build(),
            pcs.build(),
            /* read = */ {part(h)},
            /* write = */ {part(h), part(v), part(d), part(e), part(ort)},
            static_cast<uint32_t>(count), 1, 1
        );
    });
}

std::tuple<at::Tensor, at::Tensor> broadcast_batches(const at::Tensor& A, const at::Tensor& B)
{
    std::vector<int64_t> batch = at::infer_size(A.sizes().slice(0, A.dim() - 2), B.sizes().slice(0, B.dim() - 2));
    return std::make_tuple(A.expand(with_matrix(batch, A.size(-2), A.size(-1))), B.expand(with_matrix(batch, B.size(-2), B.size(-1))));
}

// a complex matrix as the real one [[re, -im], [im, re]]: products, inverses and solutions carry over
at::Tensor real_embedding(const torchvulkan::ComplexParts& a)
{
    return at::cat({at::cat({a.re, at::neg(a.im)}, -1), at::cat({a.im, a.re}, -1)}, -2);
}

// a complex A X = B through the real system [[Ar, -Ai], [Ai, Ar]] [Xr; Xi] = [Br; Bi]
at::Tensor complex_solve_by_embedding(const at::Tensor& a, const at::Tensor& b, at::ScalarType complex_dtype)
{
    torchvulkan::ComplexParts ap = torchvulkan::complex_parts(a.to(complex_dtype)), bp = torchvulkan::complex_parts(b.to(complex_dtype));
    // no error of its own: a singular A is the caller's to report (from its LU's info), in torch's words
    at::Tensor x = std::get<0>(at::linalg_solve_ex(real_embedding(ap), at::cat({bp.re, bp.im}, -2), /*left=*/true, /*check_errors=*/false));
    int64_t n = a.size(-1);
    return torchvulkan::complex_from({x.narrow(-2, 0, n), x.narrow(-2, n, n)}, complex_dtype);
}

at::Tensor solve_triangular_left(const at::Tensor& A, const at::Tensor& B, bool upper, bool unit)
{
    auto [a, b] = broadcast_batches(A, B);
    if (b.numel() == 0 || a.size(-1) == 0) return b.clone();
    if (a.is_complex() || b.is_complex()) {
        at::ScalarType type = at::promote_types(a.scalar_type(), b.scalar_type());
        at::Tensor t = upper ? at::triu(a) : at::tril(a);
        if (unit) {
            int64_t n = t.size(-1);
            t = at::add(at::sub(t, at::diag_embed(at::diagonal(t, 0, -2, -1))), at::eye(n, t.options()));
        }
        return complex_solve_by_embedding(t, b, type);
    }
    at::ScalarType dtype = linalg_dtype(B, "linalg.solve_triangular");
    std::vector<int64_t> out_shape = b.sizes().vec();
    at::Tensor a_batched = batched_copy(a, dtype);
    at::Tensor x = batched_copy(b, dtype);
    dispatch_triangular_solve_shader(a_batched, x, upper, unit);
    return x.view(out_shape).to(B.scalar_type());
}

// from LAPACK's sequential row swaps
at::Tensor permutation_from_pivots(const at::Tensor& pivots, int64_t m)
{
    std::vector<int64_t> batch(pivots.sizes().begin(), pivots.sizes().end() - 1);
    int64_t k = pivots.size(-1);
    int64_t batches = 1;
    for (int64_t s : batch) batches *= s;
    at::Tensor perm = at::arange(m, pivots.options().dtype(at::kLong)).repeat({batches, 1});
    at::Tensor piv = at::sub(pivots.reshape({batches, k}).to(at::kLong), 1);
    for (int64_t i = 0; i < k; i++) {
        at::Tensor j = piv.narrow(1, i, 1);
        at::Tensor at_i = perm.narrow(1, i, 1).clone();
        at::Tensor at_j = at::gather(perm, 1, j);
        perm.narrow(1, i, 1).copy_(at_j);
        perm.scatter_(1, j, at_i);
    }
    batch.push_back(m);
    return perm.view(batch);
}

// LU one column per step on the real and imaginary planes (real input uses the real one), with LAPACK's getrf pivoting
// (the largest |re| + |im|) or none; a zero pivot leaves its column and sets info
std::tuple<at::Tensor, at::Tensor, at::Tensor> complex_lu_factor(const at::Tensor& A, bool pivot = true)
{
    int64_t m = A.size(-2), n = A.size(-1), k = std::min(m, n);
    std::vector<int64_t> batch = batch_shape(A);
    at::ScalarType dtype = A.scalar_type() == at::kComplexHalf ? at::kComplexFloat : A.scalar_type();
    torchvulkan::ComplexParts z = torchvulkan::complex_parts(A.to(dtype).reshape({batch_count(A), m, n}));
    at::Tensor re = z.re.contiguous().clone(), im = z.im.contiguous().clone();
    int64_t batches = re.size(0);
    at::Tensor pivots = at::empty({batches, k}, A.options().dtype(at::kLong));
    at::Tensor info = at::zeros({batches}, A.options().dtype(at::kLong));
    at::Tensor rows = at::arange(m, pivots.options());
    for (int64_t j = 0; j < k; j++) {
        // LAPACK's izamax measures |re| + |im|
        at::Tensor magnitude = at::add(at::abs(re.narrow(1, j, m - j).select(2, j)), at::abs(im.narrow(1, j, m - j).select(2, j)));
        at::Tensor p = pivot ? at::add(at::argmax(magnitude, -1), j) : at::full({batches}, j, pivots.options());
        pivots.select(1, j).copy_(at::add(p, 1));
        // swap rows j and p
        at::Tensor order = rows.expand({batches, m}).clone();
        order.select(1, j).copy_(p);
        order.scatter_(1, p.unsqueeze(1), j);
        at::Tensor gather_index = order.unsqueeze(2).expand({batches, m, n});
        re = at::gather(re, 1, gather_index);
        im = at::gather(im, 1, gather_index);
        torchvulkan::ComplexParts pivot{re.select(1, j).select(1, j), im.select(1, j).select(1, j)};
        at::Tensor zero = at::logical_and(at::eq(pivot.re, 0), at::eq(pivot.im, 0));
        info = at::where(at::logical_and(zero, at::eq(info, 0)), at::full_like(info, j + 1), info);
        if (j + 1 >= m) continue;
        torchvulkan::ComplexParts column{re.narrow(1, j + 1, m - j - 1).select(2, j), im.narrow(1, j + 1, m - j - 1).select(2, j)};
        torchvulkan::ComplexParts l = torchvulkan::complex_div(column, {pivot.re.unsqueeze(1), pivot.im.unsqueeze(1)});
        l.re = at::where(zero.unsqueeze(1), column.re, l.re);
        l.im = at::where(zero.unsqueeze(1), column.im, l.im);
        re.narrow(1, j + 1, m - j - 1).select(2, j).copy_(l.re);
        im.narrow(1, j + 1, m - j - 1).select(2, j).copy_(l.im);
        if (j + 1 >= n) continue;
        torchvulkan::ComplexParts u{re.select(1, j).narrow(1, j + 1, n - j - 1).unsqueeze(1), im.select(1, j).narrow(1, j + 1, n - j - 1).unsqueeze(1)};
        torchvulkan::ComplexParts outer = torchvulkan::complex_mul({l.re.unsqueeze(2), l.im.unsqueeze(2)}, u);
        re.narrow(1, j + 1, m - j - 1).narrow(2, j + 1, n - j - 1).sub_(outer.re);
        im.narrow(1, j + 1, m - j - 1).narrow(2, j + 1, n - j - 1).sub_(outer.im);
    }
    at::Tensor lu = (A.is_complex() ? torchvulkan::complex_from({re, im}, dtype) : re).to(A.scalar_type());
    std::vector<int64_t> pivot_shape = batch;
    pivot_shape.push_back(k);
    return std::make_tuple(lu.view(with_matrix(batch, m, n)), pivots.to(at::kInt).view(pivot_shape), info.to(at::kInt).view(batch));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lu_factor_impl(const at::Tensor& A, const char* name)
{
    TORCH_CHECK(A.dim() >= 2, name, ": Expected tensor with 2 or more dimensions. Got size: ", A.sizes(), " instead");
    if (A.is_complex()) return complex_lu_factor(A);
    at::ScalarType dtype = linalg_dtype(A, name);
    int64_t m = A.size(-2), n = A.size(-1), k = std::min(m, n);
    std::vector<int64_t> batch = batch_shape(A);
    at::Tensor lu = batched_copy(A, dtype);
    at::Tensor pivots = at::empty({lu.size(0), k}, A.options().dtype(at::kInt));
    at::Tensor info = at::zeros({lu.size(0)}, A.options().dtype(at::kInt));
    if (lu.numel() > 0) {
        dispatch_lu_shader(lu, pivots);
        // LAPACK's info is the first zero pivot
        at::Tensor zero = at::eq(at::diagonal(lu, 0, -2, -1), 0);
        at::Tensor first = at::add(at::argmax(zero.to(at::kInt), -1), 1);
        info = at::where(at::any(zero, -1), first, at::zeros_like(first)).to(at::kInt);
    }
    std::vector<int64_t> pivot_shape = batch;
    pivot_shape.push_back(k);
    return std::make_tuple(lu.view(with_matrix(batch, m, n)).to(A.scalar_type()), pivots.view(pivot_shape), info.view(batch));
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> linalg_lu_factor_ex_out_vulkan(
    const at::Tensor& A,
    bool pivot,
    bool check_errors,
    at::Tensor& LU,
    at::Tensor& pivots,
    at::Tensor& info)
{
    // without pivoting (as CUDA allows): the column-by-column factorisation with identity pivots
    auto [lu, piv, inf] = pivot ? lu_factor_impl(A, "torch.linalg.lu_factor_ex") : complex_lu_factor(A, false);
    fill_out(LU, lu);
    fill_out(pivots, piv);
    fill_out(info, inf);
    if (check_errors) at::_linalg_check_errors(info, "torch.linalg.lu_factor_ex", A.dim() == 2);
    return std::forward_as_tuple(LU, pivots, info);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> unpack_lu(const at::Tensor& LU, const at::Tensor& pivots, bool unpack_data, bool unpack_pivots)
{
    int64_t m = LU.size(-2), n = LU.size(-1), k = std::min(m, n);
    at::Tensor P, L, U;
    if (unpack_data) {
        L = at::add(at::tril(LU.narrow(-1, 0, k), -1), at::eye(m, k, LU.options()));
        U = at::triu(LU.narrow(-2, 0, k));
    } else {
        L = at::empty({0}, LU.options());
        U = at::empty({0}, LU.options());
    }
    if (unpack_pivots) {
        at::Tensor perm = permutation_from_pivots(pivots, m);
        std::vector<int64_t> shape = batch_shape(LU);
        P = at::zeros(with_matrix(shape, m, m), LU.options()).scatter_(-2, perm.unsqueeze(-2), 1);
    } else {
        P = at::empty({0}, LU.options());
    }
    return std::make_tuple(P, L, U);
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> linalg_lu_out_vulkan(const at::Tensor& A, bool pivot, at::Tensor& P, at::Tensor& L, at::Tensor& U)
{
    auto [lu, pivots, info] = pivot ? lu_factor_impl(A, "torch.linalg.lu") : complex_lu_factor(A, false);
    auto [p, l, u] = unpack_lu(lu, pivots, true, pivot);
    fill_out(P, p);
    fill_out(L, l);
    fill_out(U, u);
    return std::forward_as_tuple(P, L, U);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lu_unpack_vulkan(const at::Tensor& LU_data, const at::Tensor& LU_pivots, bool unpack_data, bool unpack_pivots)
{
    return unpack_lu(LU_data, LU_pivots, unpack_data, unpack_pivots);
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> lu_unpack_out_vulkan(
    const at::Tensor& LU_data,
    const at::Tensor& LU_pivots,
    bool unpack_data,
    bool unpack_pivots,
    at::Tensor& P,
    at::Tensor& L,
    at::Tensor& U)
{
    auto [p, l, u] = unpack_lu(LU_data, LU_pivots, unpack_data, unpack_pivots);
    fill_out(P, p);
    fill_out(L, l);
    fill_out(U, u);
    return std::forward_as_tuple(P, L, U);
}

// LAPACK returns batched column-major matrices, and so do torch's meta functions, which FakeTensor and
// torch.compile take strides from
at::Tensor column_major(const at::Tensor& t)
{
    if (t.dim() < 2) return t;
    return unview(t.mT().contiguous().mT());
}

at::Tensor lu_solve_impl(const at::Tensor& LU, const at::Tensor& pivots, const at::Tensor& B, bool left, bool adjoint)
{
    if (!left) {
        return lu_solve_impl(LU, pivots, B.mH(), true, !adjoint).mH();
    }
    int64_t n = LU.size(-1);
    auto [lu, b] = broadcast_batches(LU, B);
    if (b.numel() == 0 || n == 0) return b.clone();
    std::vector<int64_t> pivot_batch(b.sizes().begin(), b.sizes().end() - 2);
    pivot_batch.push_back(pivots.size(-1));
    at::Tensor perm = permutation_from_pivots(pivots.expand(pivot_batch), n);
    at::Tensor index = perm.unsqueeze(-1).expand(b.sizes());
    if (!adjoint) {
        at::Tensor permuted = at::gather(b, -2, index);
        at::Tensor y = solve_triangular_left(lu, permuted, /*upper=*/false, /*unit=*/true);
        return solve_triangular_left(lu, y, /*upper=*/true, /*unit=*/false);
    }
    at::Tensor lu_t = lu.mH();
    at::Tensor y = solve_triangular_left(lu_t, b, /*upper=*/false, /*unit=*/false);
    at::Tensor z = solve_triangular_left(lu_t, y, /*upper=*/true, /*unit=*/true);
    return at::empty_like(z).scatter_(-2, index, z);
}

at::Tensor& linalg_lu_solve_out_vulkan(const at::Tensor& LU, const at::Tensor& pivots, const at::Tensor& B, bool left, bool adjoint, at::Tensor& result)
{
    return fill_out(result, lu_solve_impl(LU, pivots, B, left, adjoint));
}

// as the meta function: complex with left=False comes back as a conjugate view (same values)
at::Tensor linalg_lu_solve_vulkan(const at::Tensor& LU, const at::Tensor& pivots, const at::Tensor& B, bool left, bool adjoint)
{
    // the meta function's layout: column-major, row-major for left=False
    at::Tensor solved = lu_solve_impl(LU, pivots, B, left, adjoint);
    at::Tensor result = left ? column_major(solved) : unview(solved.contiguous());
    if (LU.is_complex() && !left && result.numel() > 0) return result.conj_physical().conj();
    return result.resolve_conj();  // an empty B comes back as a clone of its conjugate view
}

bool is_vector_rhs(const at::Tensor& A, const at::Tensor& B)
{
    return B.dim() == 1 || (A.dim() - 1 == B.dim() && B.sizes() == A.sizes().slice(0, A.dim() - 1));
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> linalg_solve_ex_out_vulkan(
    const at::Tensor& A,
    const at::Tensor& B,
    bool left,
    bool check_errors,
    at::Tensor& result,
    at::Tensor& LU,
    at::Tensor& pivots,
    at::Tensor& info)
{
    check_square(A, "torch.linalg.solve");
    bool vector_case = is_vector_rhs(A, B);
    at::Tensor b = vector_case ? B.unsqueeze(left ? -1 : -2) : B;
    auto [lu, piv, inf] = lu_factor_impl(A, "torch.linalg.solve");
    at::Tensor x = lu_solve_impl(lu, piv, b, left, /*adjoint=*/false);
    if (vector_case) x = x.squeeze(left ? -1 : -2);
    fill_out(result, x);
    fill_out(LU, lu);
    fill_out(pivots, piv);
    fill_out(info, inf);
    if (check_errors) at::_linalg_check_errors(info, "torch.linalg.solve_ex", A.dim() == 2);
    return std::forward_as_tuple(result, LU, pivots, info);
}

std::tuple<at::Tensor&, at::Tensor&> linalg_inv_ex_out_vulkan(const at::Tensor& A, bool check_errors, at::Tensor& inverse, at::Tensor& info)
{
    check_square(A, "linalg.inv");
    auto [lu, piv, inf] = lu_factor_impl(A, "linalg.inv");
    at::Tensor eye = at::eye(A.size(-1), A.options()).expand(A.sizes());
    fill_out(inverse, lu_solve_impl(lu, piv, eye, true, false));
    fill_out(info, inf);
    if (check_errors) at::_linalg_check_errors(info, "linalg.inv_ex", A.dim() == 2);
    return std::forward_as_tuple(inverse, info);
}

at::Tensor permutation_sign(const at::Tensor& pivots, const at::TensorOptions& options)
{
    int64_t k = pivots.size(-1);
    at::Tensor identity = at::arange(1, k + 1, pivots.options().dtype(at::kInt));
    at::Tensor swaps = at::sum(at::ne(pivots, identity), -1, /*keepdim=*/false, at::kLong);
    return at::rsub(at::mul(at::remainder(swaps, 2), 2), 1).to(options.dtype());
}

// ATen factors A^H when A is contiguous and real, and the det and slogdet backwards rely on it
at::Tensor det_factor_input(const at::Tensor& A)
{
    return A.is_contiguous() && !A.is_complex() ? A.mH() : A;
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> linalg_det_out_vulkan(const at::Tensor& A, at::Tensor& result, at::Tensor& LU, at::Tensor& pivots)
{
    check_square(A, "linalg.det");
    auto [lu, piv, info] = lu_factor_impl(det_factor_input(A), "linalg.det");
    at::Tensor diag = at::diagonal(lu, 0, -2, -1);
    at::Tensor det = at::mul(at::prod(diag, -1), permutation_sign(piv, lu.options()));
    fill_out(result, det);
    fill_out(LU, lu);
    fill_out(pivots, piv);
    return std::forward_as_tuple(result, LU, pivots);
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> linalg_slogdet_out_vulkan(
    const at::Tensor& A,
    at::Tensor& sign,
    at::Tensor& logabsdet,
    at::Tensor& LU,
    at::Tensor& pivots)
{
    check_square(A, "linalg.slogdet");
    auto [lu, piv, info] = lu_factor_impl(det_factor_input(A), "linalg.slogdet");
    at::Tensor diag = at::diagonal(lu, 0, -2, -1);
    fill_out(sign, at::mul(at::prod(lu.is_complex() ? at::sgn(diag) : at::sign(diag), -1), permutation_sign(piv, lu.options())));
    fill_out(logabsdet, at::sum(at::log(at::abs(diag)), -1));
    fill_out(LU, lu);
    fill_out(pivots, piv);
    return std::forward_as_tuple(sign, logabsdet, LU, pivots);
}

// complex Cholesky, one column per step: L[j,j] = sqrt(A[j,j] - |L[j,:j]|^2), L[i,j] = (A[i,j] - L[i,:j] conj(L[j,:j])) / L[j,j]
std::tuple<at::Tensor, at::Tensor> complex_cholesky(const at::Tensor& A, bool upper)
{
    int64_t n = A.size(-1);
    std::vector<int64_t> batch = batch_shape(A);
    at::ScalarType dtype = A.scalar_type() == at::kComplexHalf ? at::kComplexFloat : A.scalar_type();
    torchvulkan::ComplexParts a = torchvulkan::complex_parts(A.to(dtype).reshape({batch_count(A), n, n}));
    int64_t batches = a.re.size(0);
    at::Tensor re = at::zeros({batches, n, n}, a.re.options()), im = at::zeros_like(re);
    at::Tensor info = at::zeros({batches}, A.options().dtype(at::kLong));
    for (int64_t j = 0; j < n; j++) {
        at::Tensor row_re = re.select(1, j).narrow(1, 0, j), row_im = im.select(1, j).narrow(1, 0, j);
        at::Tensor d = at::sub(a.re.select(1, j).select(1, j), at::sum(at::add(at::mul(row_re, row_re), at::mul(row_im, row_im)), -1));
        info = at::where(at::logical_and(at::logical_not(at::gt(d, 0)), at::eq(info, 0)), at::full_like(info, j + 1), info);
        at::Tensor diagonal = at::sqrt(d);
        re.select(1, j).select(1, j).copy_(diagonal);
        if (j + 1 >= n) continue;
        // L[j+1:, :j] conj(L[j, :j])
        at::Tensor below_re = re.narrow(1, j + 1, n - j - 1).narrow(2, 0, j), below_im = im.narrow(1, j + 1, n - j - 1).narrow(2, 0, j);
        torchvulkan::ComplexParts dot = torchvulkan::complex_mul({below_re, below_im}, {row_re.unsqueeze(1), at::neg(row_im).unsqueeze(1)});
        at::Tensor col_re = at::sub(a.re.narrow(1, j + 1, n - j - 1).select(2, j), at::sum(dot.re, -1));
        at::Tensor col_im = at::sub(a.im.narrow(1, j + 1, n - j - 1).select(2, j), at::sum(dot.im, -1));
        re.narrow(1, j + 1, n - j - 1).select(2, j).copy_(at::div(col_re, diagonal.unsqueeze(1)));
        im.narrow(1, j + 1, n - j - 1).select(2, j).copy_(at::div(col_im, diagonal.unsqueeze(1)));
    }
    at::Tensor L = torchvulkan::complex_from({re, im}, dtype).view(A.sizes()).to(A.scalar_type());
    if (upper) L = L.mH().resolve_conj().contiguous();
    return std::make_tuple(L, info.to(at::kInt).view(batch));
}

std::tuple<at::Tensor, at::Tensor> cholesky_impl(const at::Tensor& A, bool upper, const char* name)
{
    check_square(A, name);
    if (A.is_complex()) return complex_cholesky(A, upper);
    at::ScalarType dtype = linalg_dtype(A, name);
    std::vector<int64_t> batch = batch_shape(A);
    at::Tensor l = batched_copy(A, dtype);
    at::Tensor info = at::zeros({l.size(0)}, A.options().dtype(at::kInt));
    if (l.numel() > 0) dispatch_cholesky_shader(l, info);
    at::Tensor L = l.view(A.sizes()).to(A.scalar_type());
    if (upper) L = L.mT().contiguous();
    return std::make_tuple(L, info.view(batch));
}

std::tuple<at::Tensor&, at::Tensor&> linalg_cholesky_ex_out_vulkan(const at::Tensor& A, bool upper, bool check_errors, at::Tensor& L, at::Tensor& info)
{
    check_out_dtype(L, A.scalar_type());
    auto [l, inf] = cholesky_impl(A, upper, "linalg.cholesky");
    fill_out(L, l);
    fill_out(info, inf);
    if (check_errors) at::_linalg_check_errors(info, "linalg.cholesky_ex", A.dim() == 2);
    return std::forward_as_tuple(L, info);
}

at::Tensor cholesky_vulkan(const at::Tensor& self, bool upper)
{
    auto [l, info] = cholesky_impl(self, upper, "cholesky");
    at::_linalg_check_errors(info, "cholesky", self.dim() == 2);
    return l;
}

at::Tensor& cholesky_out_vulkan(const at::Tensor& self, bool upper, at::Tensor& out)
{
    return fill_out(out, cholesky_vulkan(self, upper));
}

at::Tensor cholesky_solve_helper_vulkan(const at::Tensor& self, const at::Tensor& A, bool upper)
{
    // conjugate transposes, which complex needs (for real they are transposes)
    at::Tensor lower = upper ? A.mH() : A;
    at::Tensor y = solve_triangular_left(lower, self, /*upper=*/false, /*unit=*/false);
    return column_major(solve_triangular_left(lower.mH(), y, /*upper=*/true, /*unit=*/false));
}

at::Tensor cholesky_inverse_vulkan(const at::Tensor& self, bool upper)
{
    TORCH_CHECK(self.dim() >= 2, "cholesky_inverse: The input tensor A must have at least 2 dimensions.");
    TORCH_CHECK(self.size(-1) == self.size(-2), "cholesky_inverse: A must be batches of square matrices, but they are ", self.size(-2), " by ",
                self.size(-1), " matrices");
    at::Tensor eye = at::eye(self.size(-1), self.options()).expand(self.sizes());
    return cholesky_solve_helper_vulkan(eye, self, upper);
}

at::Tensor& cholesky_inverse_out_vulkan(const at::Tensor& self, bool upper, at::Tensor& out)
{
    return fill_out(out, cholesky_inverse_vulkan(self, upper));
}

at::Tensor solve_triangular(const at::Tensor& A, const at::Tensor& B, bool upper, bool left, bool unitriangular)
{
    if (left) return solve_triangular_left(A, B, upper, unitriangular);
    return solve_triangular_left(A.mT(), B.mT(), !upper, unitriangular).mT();
}

at::Tensor linalg_solve_triangular_vulkan(const at::Tensor& A, const at::Tensor& B, bool upper, bool left, bool unitriangular)
{
    TORCH_CHECK(A.dim() >= 2 && B.dim() >= 2, "linalg.solve_triangular: The input tensors must have at least 2 dimensions.");
    // the result takes A's dtype, as on the CPU and in the meta function
    return column_major(solve_triangular(A, B.to(A.scalar_type()), upper, left, unitriangular));
}

at::Tensor& linalg_solve_triangular_out_vulkan(const at::Tensor& A, const at::Tensor& B, bool upper, bool left, bool unitriangular, at::Tensor& out)
{
    return fill_out(out, linalg_solve_triangular_vulkan(A, B, upper, left, unitriangular));
}

std::tuple<at::Tensor&, at::Tensor&> triangular_solve_out_vulkan(
    const at::Tensor& self,
    const at::Tensor& A,
    bool upper,
    bool transpose,
    bool unitriangular,
    at::Tensor& X,
    at::Tensor& M)
{
    TORCH_CHECK(self.dim() >= 2 && A.dim() >= 2, "torch.triangular_solve: Expected b and A to have at least 2 dimensions");
    check_out_dtype(X, at::promote_types(self.scalar_type(), A.scalar_type()));
    check_out_dtype(M, at::promote_types(self.scalar_type(), A.scalar_type()));
    at::Tensor coefficient = transpose ? A.mT() : A;
    fill_out(X, solve_triangular_left(coefficient, self, transpose ? !upper : upper, unitriangular));
    auto [a, b] = broadcast_batches(A, self);
    fill_out(M, a.clone());
    return std::forward_as_tuple(X, M);
}

at::ScalarType complex_compute_dtype(at::ScalarType t)
{
    return t == at::kComplexHalf ? at::kComplexFloat : t;
}

// LAPACK's zgeqrf, one column per step: beta = -sign(re(alpha)) |(alpha, x)|, tau = ((beta - re) / beta, -im / beta),
// x /= alpha - beta, then H^H = I - conj(tau) v v^H on the trailing columns
std::tuple<at::Tensor, at::Tensor> complex_geqrf(const at::Tensor& A)
{
    int64_t m = A.size(-2), n = A.size(-1), k = std::min(m, n);
    at::ScalarType dtype = complex_compute_dtype(A.scalar_type());
    at::Tensor q = A.to(dtype).reshape({batch_count(A), m, n}).clone();
    int64_t batches = q.size(0);
    at::Tensor tau = at::zeros({batches, k}, q.options());
    for (int64_t j = 0; j < k; j++) {
        at::Tensor alpha = q.select(1, j).select(1, j);
        at::Tensor x = q.narrow(1, j + 1, m - j - 1).select(2, j);
        at::Tensor xnorm = at::linalg_vector_norm(x, 2, {-1}, false);
        at::Tensor re = at::real(alpha), im = at::imag(alpha);
        at::Tensor beta = at::neg(at::copysign(at::sqrt(at::add(at::add(at::mul(re, re), at::mul(im, im)), at::mul(xnorm, xnorm))), re));
        at::Tensor skip = at::logical_and(at::eq(xnorm, 0), at::eq(im, 0));
        at::Tensor t = at::complex(at::div(at::sub(beta, re), beta), at::div(at::neg(im), beta));
        at::Tensor scaled = at::div(x, at::sub(alpha, beta).unsqueeze(1));
        at::Tensor v_tail = at::where(skip.unsqueeze(1), x, scaled);
        t = at::where(skip, at::zeros_like(t), t);
        q.narrow(1, j + 1, m - j - 1).select(2, j).copy_(v_tail);
        q.select(1, j).select(1, j).copy_(at::where(skip, alpha, at::complex(beta, at::zeros_like(beta))));
        tau.select(1, j).copy_(t);
        if (j + 1 >= n) continue;
        at::Tensor v = at::cat({at::ones({batches, 1}, q.options()), v_tail}, 1).unsqueeze(2);
        at::Tensor c = q.narrow(1, j, m - j).narrow(2, j + 1, n - j - 1);
        at::Tensor w = at::matmul(v.mH(), c);
        c.sub_(at::mul(at::conj(t).view({batches, 1, 1}), at::matmul(v, w)).resolve_conj());
    }
    std::vector<int64_t> tau_shape = batch_shape(A);
    tau_shape.push_back(k);
    return std::make_tuple(q.view(A.sizes()).to(A.scalar_type()), tau.view(tau_shape).to(A.scalar_type()));
}

// the first `columns` columns of H_1 ... H_k (H_j = I - tau_j v_j v_j^H), applied backwards to the identity as zung2r does
at::Tensor complex_householder_product(const at::Tensor& reflectors, const at::Tensor& tau, int64_t columns)
{
    int64_t m = reflectors.size(-2), k = tau.size(-1);
    at::ScalarType dtype = complex_compute_dtype(reflectors.scalar_type());
    at::Tensor r = reflectors.to(dtype).reshape({batch_count(reflectors), m, reflectors.size(-1)});
    int64_t batches = r.size(0);
    at::Tensor t = tau.to(dtype).reshape({batches, k});
    at::Tensor q = at::eye(m, columns, r.options()).expand({batches, m, columns}).contiguous();
    for (int64_t j = k - 1; j >= 0; j--) {
        at::Tensor v = at::cat({at::ones({batches, 1}, r.options()), r.narrow(1, j + 1, m - j - 1).select(2, j)}, 1).unsqueeze(2);
        at::Tensor rows = q.narrow(1, j, m - j);
        rows.sub_(at::mul(t.select(1, j).view({batches, 1, 1}), at::matmul(v, at::matmul(v.mH(), rows))).resolve_conj());
    }
    return q;
}

std::tuple<at::Tensor, at::Tensor> geqrf_impl(const at::Tensor& A)
{
    TORCH_CHECK(A.dim() >= 2, "torch.geqrf: input must have at least 2 dimensions.");
    if (A.is_complex()) return complex_geqrf(A);
    at::ScalarType dtype = linalg_dtype(A, "torch.geqrf");
    int64_t m = A.size(-2), n = A.size(-1), k = std::min(m, n);
    std::vector<int64_t> batch = batch_shape(A);
    at::Tensor qr = batched_copy(A, dtype);
    at::Tensor tau = at::zeros({qr.size(0), k}, qr.options());
    if (qr.numel() > 0 && k > 0) dispatch_householder_shader(HouseholderMode::GEQRF, qr, tau, at::Tensor(), true, false);
    std::vector<int64_t> tau_shape = batch;
    tau_shape.push_back(k);
    return std::make_tuple(qr.view(A.sizes()).to(A.scalar_type()), tau.view(tau_shape).to(A.scalar_type()));
}

std::tuple<at::Tensor&, at::Tensor&> geqrf_out_vulkan(const at::Tensor& self, at::Tensor& a, at::Tensor& tau)
{
    auto [qr, t] = geqrf_impl(self);
    fill_out(a, qr);
    fill_out(tau, t);
    return std::forward_as_tuple(a, tau);
}

std::tuple<at::Tensor, at::Tensor> geqrf_vulkan(const at::Tensor& self)
{
    // column-major reflectors, as LAPACK's geqrf
    auto [reflectors, tau] = geqrf_impl(self);
    return std::make_tuple(reflectors.mT().contiguous().mT(), tau);
}

at::Tensor orgqr_impl(const at::Tensor& input, const at::Tensor& tau)
{
    TORCH_CHECK(input.size(-2) >= input.size(-1), "torch.linalg.householder_product: input.shape[-2] must be greater than or equal to input.shape[-1]");
    TORCH_CHECK(input.size(-1) >= tau.size(-1), "torch.linalg.householder_product: input.shape[-1] must be greater than or equal to tau.shape[-1]");
    if (input.is_complex()) return complex_householder_product(input, tau, input.size(-1)).view(input.sizes()).to(input.scalar_type());
    at::ScalarType dtype = linalg_dtype(input, "torch.linalg.householder_product");
    at::Tensor q = batched_copy(input, dtype);
    at::Tensor t = tau.to(dtype).reshape({q.size(0), tau.size(-1)}).contiguous();
    if (t.size(1) == 0) return at::eye(input.size(-2), input.size(-1), input.options()).expand(input.sizes()).contiguous();
    if (q.numel() > 0) dispatch_householder_shader(HouseholderMode::ORGQR, q, t, at::Tensor(), true, false);
    return q.view(input.sizes()).to(input.scalar_type());
}

at::Tensor linalg_householder_product_vulkan(const at::Tensor& input, const at::Tensor& tau)
{
    TORCH_CHECK(input.dim() >= 2, "torch.linalg.householder_product: input must have at least 2 dimensions.");
    // column-major matrices, as LAPACK's result (and the meta function's)
    return orgqr_impl(input, tau).mT().contiguous().mT();
}

at::Tensor& linalg_householder_product_out_vulkan(const at::Tensor& input, const at::Tensor& tau, at::Tensor& out)
{
    return fill_out(out, linalg_householder_product_vulkan(input, tau));
}

at::Tensor ormqr_vulkan(const at::Tensor& input, const at::Tensor& tau, const at::Tensor& other, bool left, bool transpose)
{
    TORCH_CHECK(input.dim() >= 2, "torch.ormqr: input must have at least 2 dimensions.");
    TORCH_CHECK(other.dim() >= 2, "torch.ormqr: other must have at least 2 dimensions.");
    // torch's checks, in its order
    int64_t side = left ? -2 : -1;
    TORCH_CHECK(other.size(side) == input.size(-2), "torch.ormqr: other.shape[", side, "] must be equal to input.shape[-2]");
    TORCH_CHECK(tau.size(-1) == std::min(other.size(side), input.size(-1)),
        "torch.ormqr: tau.shape[-1] must be equal to min(other.shape[-2], input.shape[-1])");
    TORCH_CHECK(input.dim() - tau.dim() == 1, "torch.ormqr: Expected tau to have one dimension less than input, but got tau.ndim equal to ",
        tau.dim(), " and input.ndim is equal to ", input.dim());
    TORCH_CHECK(input.dim() == other.dim(), "torch.ormqr: Expected other to have the same number of dimensions as input, but got other.ndim equal to ",
        other.dim(), " and input.ndim is equal to ", input.dim());
    if (input.dim() > 2) {
        at::IntArrayRef batch = input.sizes().slice(0, input.dim() - 2);
        TORCH_CHECK(tau.sizes().slice(0, tau.dim() - 1).equals(batch), "torch.ormqr: Expected batch dimensions of tau to be equal to input.shape[:-2], but got ",
            tau.sizes().slice(0, tau.dim() - 1));
        TORCH_CHECK(other.sizes().slice(0, other.dim() - 2).equals(batch), "torch.ormqr: Expected batch dimensions of other to be equal to input.shape[:-2], but got ",
            other.sizes().slice(0, other.dim() - 2));
    }
    if (input.is_complex() || other.is_complex()) {
        // through the explicit m x m Q
        int64_t m = input.size(-2);
        at::ScalarType dtype = at::promote_types(input.scalar_type(), other.scalar_type());
        at::Tensor q = complex_householder_product(input, tau, m).view(with_matrix(batch_shape(input), m, m)).to(dtype);
        at::Tensor op = transpose ? q.mH() : q;
        at::Tensor c = other.to(dtype);
        return column_major((left ? at::matmul(op, c) : at::matmul(c, op)).resolve_conj());
    }
    at::ScalarType dtype = linalg_dtype(other, "torch.ormqr");
    int64_t batches = other.numel() / std::max<int64_t>(other.size(-1) * other.size(-2), 1);
    at::Tensor a = batched_copy(input.expand(with_matrix(batch_shape(other), input.size(-2), input.size(-1))), dtype);
    at::Tensor t = tau.to(dtype).unsqueeze(-2).expand(with_matrix(batch_shape(other), 1, tau.size(-1))).reshape({batches, tau.size(-1)}).contiguous();
    at::Tensor c = batched_copy(other, dtype);
    if (c.numel() > 0 && t.size(1) > 0) dispatch_householder_shader(HouseholderMode::ORMQR, a, t, c, left, transpose);
    return column_major(c.view(other.sizes()).to(other.scalar_type()));
}

at::Tensor& ormqr_out_vulkan(const at::Tensor& input, const at::Tensor& tau, const at::Tensor& other, bool left, bool transpose, at::Tensor& out)
{
    return fill_out(out, ormqr_vulkan(input, tau, other, left, transpose));
}

std::tuple<at::Tensor&, at::Tensor&> linalg_qr_out_vulkan(const at::Tensor& A, c10::string_view mode, at::Tensor& Q, at::Tensor& R)
{
    TORCH_CHECK(mode == "reduced" || mode == "complete" || mode == "r", "qr received unrecognized mode '", mode, "' but expected one of 'reduced' (default), 'r', or 'complete'");
    TORCH_CHECK(A.dim() >= 2, "linalg.qr: The input tensor A must have at least 2 dimensions.");
    int64_t m = A.size(-2), n = A.size(-1), k = std::min(m, n);
    auto [qr, tau] = geqrf_impl(A);
    bool complete = mode == "complete";
    if (complete) fill_out(R, at::triu(qr));
    else fill_out(R, at::triu(qr.narrow(-2, 0, k)));
    if (mode == "r") {
        fill_out(Q, at::empty({0}, A.options()));
    } else if (complete && m > n) {
        at::Tensor padded = at::cat({qr, at::zeros(with_matrix(batch_shape(A), m, m - n), qr.options())}, -1);
        fill_out(Q, orgqr_impl(padded, tau));
    } else {
        fill_out(Q, orgqr_impl(complete ? qr.narrow(-1, 0, m) : qr.narrow(-1, 0, k), tau));
    }
    return std::forward_as_tuple(Q, R);
}

// one-sided Jacobi
std::tuple<at::Tensor, at::Tensor, at::Tensor> svd_impl(const at::Tensor& A, bool full_matrices, bool compute_uv);

std::tuple<at::Tensor, at::Tensor, at::Tensor> svd_impl(const at::Tensor& A, bool full_matrices, bool compute_uv)
{
    TORCH_CHECK(A.dim() >= 2, "linalg.svd: The input tensor A must have at least 2 dimensions.");
    bool complex = A.is_complex();
    at::ScalarType dtype = complex ? complex_compute_dtype(A.scalar_type()) : linalg_dtype(A, "linalg.svd");
    int64_t m = A.size(-2), n = A.size(-1), k = std::min(m, n);
    std::vector<int64_t> batch = batch_shape(A);
    // a wide matrix through its (conjugate) transpose, A^H = V S U^H
    bool wide = m < n;
    auto adjoint = [complex](const at::Tensor& t) { return complex ? t.mH() : t.mT(); };
    at::Tensor tall = wide ? adjoint(A) : A;
    int64_t rows = std::max(m, n);

    at::Tensor g = batched_copy(tall, dtype);
    // each matrix scaled by a power of two to largest entry ~1, as gesvd scales: the rotations square entries, which
    // overflow or flush near 1e19 or 1e-19 in float32
    at::Tensor scale;
    if (g.numel() > 0) {
        at::Tensor largest = at::amax(at::abs(g), {1, 2}, /*keepdim=*/true);
        at::Tensor exponent = std::get<1>(at::frexp(at::where(at::isfinite(largest), largest, at::zeros_like(largest))));
        scale = at::ldexp(at::ones_like(largest), at::neg(exponent));
        g = at::mul(g, scale);
    }
    at::Tensor v = at::empty({g.size(0), k, k}, g.options());
    if (g.numel() > 0) dispatch_jacobi_shader(JacobiMode::SVD, g, v);

    at::Tensor s = at::linalg_vector_norm(g, 2, {1}, false);
    at::Tensor order = std::get<1>(at::sort(s, /*stable=*/true, -1, /*descending=*/true));
    s = at::gather(s, 1, order);
    std::vector<int64_t> s_shape = batch;
    s_shape.push_back(k);
    at::Tensor S = (g.numel() > 0 ? at::div(s, scale.squeeze(2)) : s).view(s_shape).to(c10::toRealValueType(A.scalar_type()));
    if (!compute_uv) return std::make_tuple(at::empty({0}, A.options()), S, at::empty({0}, A.options()));

    at::Tensor column_order = order.unsqueeze(1);
    at::Tensor g_sorted = at::gather(g, 2, column_order.expand({g.size(0), rows, k}));
    at::Tensor v_sorted = at::gather(v, 2, column_order.expand({g.size(0), k, k}));
    // a singular value at rounding level of the largest is zero: its column of g is noise, which / s isn't orthogonal
    double eps = dtype == at::kDouble || dtype == at::kComplexDouble ? 2.220446049250313e-16 : 1.1920928955078125e-07;
    at::Tensor tolerance = s.numel() > 0 ? at::mul(s.narrow(1, 0, std::min<int64_t>(k, 1)), eps * static_cast<double>(rows)) : s;
    at::Tensor nonzero = at::gt(s, tolerance).unsqueeze(1);
    at::Tensor u = at::where(nonzero, at::div(g_sorted, at::where(nonzero, s.unsqueeze(1), at::ones_like(s.unsqueeze(1)))), at::zeros_like(g_sorted));

    // zero singular values and full matrices take orthonormal columns from u's QR
    bool full = full_matrices && rows > k;
    bool has_zero = s.numel() > 0 && at::logical_not(nonzero).any().item<bool>();
    if (u.numel() > 0 && (full || has_zero)) {
        auto [qr, tau] = geqrf_impl(u);
        at::Tensor padded = full ? at::cat({qr, at::zeros({u.size(0), rows, rows - k}, qr.options())}, -1) : qr;
        at::Tensor q = orgqr_impl(padded, tau);
        at::Tensor first = at::where(nonzero, u, q.narrow(2, 0, k));
        u = full ? at::cat({first, q.narrow(2, k, rows - k)}, 2) : first;
    } else if (full && k == 0) {
        u = at::eye(rows, g.options()).expand({g.size(0), rows, rows}).contiguous();
    }
    int64_t u_cols = full ? rows : k;
    at::Tensor U_tall = u.view(with_matrix(batch, rows, u_cols)).to(A.scalar_type());
    at::Tensor V = v_sorted.view(with_matrix(batch, k, k)).to(A.scalar_type());
    if (!wide) return std::make_tuple(U_tall, S, adjoint(V).resolve_conj().contiguous());
    return std::make_tuple(V.contiguous(), S, adjoint(U_tall).resolve_conj().contiguous());
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> linalg_svd_out_vulkan(
    const at::Tensor& A,
    bool full_matrices,
    bool compute_uv,
    c10::optional<c10::string_view> /* driver */,
    at::Tensor& U,
    at::Tensor& S,
    at::Tensor& Vh)
{
    auto [u, s, vh] = svd_impl(A, full_matrices, compute_uv);
    fill_out(U, u);
    fill_out(S, s);
    fill_out(Vh, vh);
    return std::forward_as_tuple(U, S, Vh);
}

// two-sided Jacobi
std::tuple<at::Tensor&, at::Tensor&> linalg_eigh_out_vulkan(const at::Tensor& A, c10::string_view uplo, bool compute_v, at::Tensor& eigenvalues, at::Tensor& eigenvectors)
{
    check_out_dtype(eigenvalues, c10::toRealValueType(A.scalar_type()));
    if (compute_v) check_out_dtype(eigenvectors, A.scalar_type());
    check_square(A, "linalg.eigh");
    TORCH_CHECK(uplo == "L" || uplo == "U" || uplo == "l" || uplo == "u", "Expected UPLO argument to be 'L' or 'U', but got ", uplo);
    at::ScalarType dtype = A.is_complex() ? at::kFloat : linalg_dtype(A, "linalg.eigh");  // complex goes through its real embedding
    int64_t n = A.size(-1);
    std::vector<int64_t> batch = batch_shape(A);
    bool lower = uplo == "L" || uplo == "l";
    if (A.is_complex()) {
        // the Hermitian matrix's real embedding is symmetric with every eigenvalue twice: every other eigenpair, its
        // vector [x; y] read as x + iy
        int64_t m = A.size(-1);
        at::ScalarType dtype = A.scalar_type() == at::kComplexHalf ? at::kComplexFloat : A.scalar_type();
        at::Tensor h = lower ? at::add(at::tril(A), at::tril(A, -1).mH()) : at::add(at::triu(A), at::triu(A, 1).mH());
        torchvulkan::ComplexParts hp = torchvulkan::complex_parts(h.resolve_conj().to(dtype));
        hp.im = at::sub(hp.im, at::diag_embed(at::diagonal(hp.im, 0, -2, -1)));  // a Hermitian diagonal is real (LAPACK ignores its imaginary part)
        auto [w, vectors] = at::linalg_eigh(real_embedding(hp));
        fill_out(eigenvalues, w.slice(-1, 0, 2 * m, 2).to(c10::toRealValueType(A.scalar_type())));
        if (compute_v) {
            at::Tensor v = vectors.slice(-1, 0, 2 * m, 2);
            fill_out(eigenvectors, torchvulkan::complex_from({v.narrow(-2, 0, m), v.narrow(-2, m, m)}, dtype).to(A.scalar_type()));
        } else {
            fill_out(eigenvectors, at::empty({0}, A.options()));
        }
        return std::forward_as_tuple(eigenvalues, eigenvectors);
    }
    at::Tensor symmetric = lower ? at::add(at::tril(A), at::tril(A, -1).mT()) : at::add(at::triu(A), at::triu(A, 1).mT());

    at::Tensor a = batched_copy(symmetric, dtype);
    at::Tensor v = at::empty({a.size(0), n, n}, a.options());
    if (a.numel() > 0) dispatch_jacobi_shader(JacobiMode::EIGH, a, v);
    at::Tensor values = at::diagonal(a, 0, -2, -1);
    auto [sorted, order] = at::sort(values, /*stable=*/true, -1, /*descending=*/false);
    std::vector<int64_t> value_shape = batch;
    value_shape.push_back(n);
    fill_out(eigenvalues, sorted.view(value_shape).to(A.scalar_type()));
    if (compute_v) fill_out(eigenvectors, at::gather(v, 2, order.unsqueeze(1).expand({a.size(0), n, n})).view(A.sizes()).to(A.scalar_type()));
    else fill_out(eigenvectors, at::empty({0}, A.options()));
    return std::forward_as_tuple(eigenvalues, eigenvectors);
}

// LAPACK's sytrf/hetrf take float, double and their complex types
at::ScalarType ldl_dtype(const at::Tensor& t, const char* name)
{
    if (!t.is_complex()) return linalg_dtype(t, name);
    return (t.scalar_type() == at::kComplexDouble && is_dtype_supported(t, at::kDouble)) ? at::kComplexDouble : at::kComplexFloat;
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> linalg_ldl_factor_ex_out_vulkan(
    const at::Tensor& self,
    bool hermitian,
    bool check_errors,
    at::Tensor& LD,
    at::Tensor& pivots,
    at::Tensor& info)
{
    check_square(self, "torch.linalg.ldl_factor_ex");
    at::ScalarType dtype = ldl_dtype(self, "torch.linalg.ldl_factor_ex");
    std::vector<int64_t> batch = batch_shape(self);
    int64_t n = self.size(-1);
    at::Tensor a = batched_copy(self, dtype);
    at::Tensor piv = at::zeros({a.size(0), n}, self.options().dtype(at::kInt));
    at::Tensor inf = at::zeros({a.size(0)}, self.options().dtype(at::kInt));
    if (a.numel() > 0) dispatch_ldl_shader(LdlMode::FACTOR, a, piv, inf, at::Tensor(), hermitian);
    std::vector<int64_t> pivot_shape = batch;
    pivot_shape.push_back(n);
    fill_out(LD, at::tril(a).view(self.sizes()).to(self.scalar_type()));
    fill_out(pivots, piv.view(pivot_shape));
    fill_out(info, inf.view(batch));
    if (check_errors) at::_linalg_check_errors(info, "torch.linalg.ldl_factor_ex", self.dim() == 2);
    return std::forward_as_tuple(LD, pivots, info);
}

at::Tensor& linalg_ldl_solve_out_vulkan(const at::Tensor& LD, const at::Tensor& pivots, const at::Tensor& B, bool hermitian, at::Tensor& result)
{
    TORCH_CHECK(LD.dim() >= 2 && B.dim() >= 2, "torch.linalg.ldl_solve: LD and B must have at least 2 dimensions");
    at::ScalarType dtype = ldl_dtype(B, "torch.linalg.ldl_solve");
    auto [ld, b] = broadcast_batches(LD, B);
    std::vector<int64_t> pivot_batch(b.sizes().begin(), b.sizes().end() - 2);
    pivot_batch.push_back(pivots.size(-1));
    at::Tensor a = batched_copy(ld, dtype);
    at::Tensor piv = pivots.expand(pivot_batch).reshape({a.size(0), pivots.size(-1)}).contiguous();
    at::Tensor x = batched_copy(b, dtype);
    if (x.numel() > 0) dispatch_ldl_shader(LdlMode::SOLVE, a, piv, at::Tensor(), x, hermitian);
    return fill_out(result, x.view(b.sizes()).to(B.scalar_type()));
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> linalg_lstsq_out_vulkan(
    const at::Tensor& self,
    const at::Tensor& b,
    c10::optional<double> rcond,
    c10::optional<c10::string_view> driver_opt,
    at::Tensor& solution,
    at::Tensor& residuals,
    at::Tensor& rank,
    at::Tensor& singular_values)
{
    TORCH_CHECK(self.dim() >= 2, "torch.linalg.lstsq: input must have at least 2 dimensions.");
    TORCH_CHECK(b.dim() >= 1, "torch.linalg.lstsq: other must have at least 1 dimension.");
    int64_t dim_diff = self.dim() - b.dim();
    TORCH_CHECK(dim_diff >= 0 && dim_diff <= 1,
        "torch.linalg.lstsq: input.dim() must be greater or equal to other.dim() and (input.dim() - other.dim()) <= 1");
    bool vector_rhs = b.dim() == 1 || (dim_diff == 1 && b.sizes().equals(self.sizes().slice(0, self.dim() - 1)));
    TORCH_CHECK(self.size(-2) == (vector_rhs ? b.size(-1) : b.size(-2)),
        vector_rhs ? "torch.linalg.lstsq: input.size(-2) should match other.size(-1)" : "torch.linalg.lstsq: input.size(-2) should match other.size(-2)");
    c10::string_view driver = driver_opt.value_or("gelsy");
    TORCH_CHECK(driver == "gels" || driver == "gelsy" || driver == "gelsd" || driver == "gelss",
        "torch.linalg.lstsq: parameter `driver` should be one of (gels, gelsy, gelsd, gelss)"
    );
    int64_t m = self.size(-2), n = self.size(-1);
    bool vector_case = b.dim() == self.dim() - 1;
    at::Tensor rhs = vector_case ? b.unsqueeze(-1) : b;
    auto [a, bb] = broadcast_batches(self, rhs);

    at::Tensor x;
    if (m >= n) {
        auto [qr, tau] = geqrf_impl(a);
        at::Tensor qtb = ormqr_vulkan(qr, tau, bb, /*left=*/true, /*transpose=*/true).narrow(-2, 0, n);
        x = solve_triangular_left(qr.narrow(-2, 0, n), qtb, /*upper=*/true, /*unit=*/false);
    } else {
        // A = R^H Q^H from the QR of A^H (conjugate transposes, which complex needs)
        auto [qr, tau] = geqrf_impl(a.mH().resolve_conj());
        at::Tensor r = qr.narrow(-2, 0, m);
        at::Tensor y = solve_triangular_left(r.mH(), bb, /*upper=*/false, /*unit=*/false);
        at::Tensor padded = at::cat({y, at::zeros(with_matrix(batch_shape(y), n - m, y.size(-1)), y.options())}, -2);
        x = ormqr_vulkan(qr, tau, padded, /*left=*/true, /*transpose=*/false);
    }

    std::vector<int64_t> batch = batch_shape(x);
    bool needs_svd = driver != "gels";
    at::Tensor sv = needs_svd ? std::get<1>(svd_impl(a, false, false)) : at::Tensor();
    if (driver == "gelsd" || driver == "gelss") fill_out(singular_values, sv);
    else fill_out(singular_values, at::empty({0}, self.options().dtype(c10::toRealValueType(self.scalar_type()))));

    if (needs_svd) {
        double eps = std::numeric_limits<float>::epsilon();
        if (self.scalar_type() == at::kDouble || self.scalar_type() == at::kComplexDouble) eps = std::numeric_limits<double>::epsilon();
        double threshold = rcond.has_value() && *rcond > 0 ? *rcond : eps * static_cast<double>(std::max(m, n));
        at::Tensor largest = sv.numel() > 0 ? std::get<0>(at::max(sv, -1, /*keepdim=*/true)) : sv;
        fill_out(rank, sv.numel() > 0 ? at::sum(at::gt(sv, at::mul(largest, threshold)), -1, false, at::kLong) : at::zeros(batch, self.options().dtype(at::kLong)));
    } else {
        fill_out(rank, at::empty({0}, self.options().dtype(at::kLong)));
    }

    bool full_rank = needs_svd ? (rank.numel() == 0 || at::eq(rank, std::min(m, n)).all().item<bool>()) : true;
    at::ScalarType real = c10::toRealValueType(self.scalar_type());
    if (m > n && driver != "gelsy" && full_rank) fill_out(residuals, at::sum(at::square(at::abs(at::sub(at::matmul(a, x), bb))), -2));
    else fill_out(residuals, at::empty({0}, self.options().dtype(real)));

    fill_out(solution, vector_case ? x.squeeze(-1) : x);
    return std::forward_as_tuple(solution, residuals, rank, singular_values);
}

at::Tensor linear_combination(const std::vector<at::Tensor>& matrices, const std::vector<double>& coefficients)
{
    at::Tensor result = at::mul(matrices[0], coefficients[0]);
    for (size_t i = 1; i < coefficients.size(); i++) result = at::add(result, at::mul(matrices[i], coefficients[i]));
    return result;
}

// as ATen's _fill_matrix_powers
std::vector<at::Tensor> matrix_powers(const at::Tensor& A, int count)
{
    std::vector<at::Tensor> powers = {at::eye(A.size(-1), A.options()).expand(A.sizes()).contiguous(), A};
    if (count > 2) powers.push_back(at::matmul(A, A));
    if (count > 3) powers.push_back(at::matmul(A, powers[2]));
    if (count > 4) powers.push_back(at::matmul(powers[3], powers[3]));
    return powers;
}

// degree-18 Taylor approximant (Bader, Blanes and Casas), as ATen's LinearAlgebra.cpp
at::Tensor matrix_exp_t18(const at::Tensor& A)
{
    static const double b[5][5] = {
        {0., -1.00365581030144618291e-01, -8.02924648241156932449e-03, -8.92138498045729985177e-04, 0.},
        {0., 3.97849749499645077844e-01, 1.36783778460411720168e+00, 4.98289622525382669416e-01, -6.37898194594723280150e-04},
        {-1.09676396052962061844e+01, 1.68015813878906206114e+00, 5.71779846478865511061e-02, -6.98210122488052056106e-03, 3.34975017086070470649e-05},
        {-9.04316832390810593223e-02, -6.76404519071381882256e-02, 6.75961301770459654925e-02, 2.95552570429315521194e-02, -1.39180257516060693404e-05},
        {0., 0., -9.23364619367118555360e-02, -1.69364939002081722752e-02, -1.40086798182036094347e-05},
    };
    std::vector<at::Tensor> As = matrix_powers(A, 5);
    std::vector<at::Tensor> Bs;
    for (int i = 0; i < 5; i++) Bs.push_back(linear_combination(As, std::vector<double>(b[i], b[i] + 5)));
    Bs[3] = at::add(Bs[3], at::matmul(Bs[0], Bs[4]));
    return at::add(Bs[1], at::matmul(at::add(Bs[2], Bs[3]), Bs[3]));
}

at::Tensor matrix_exp_t12(const at::Tensor& A)
{
    static const double b[4][4] = {
        {9.0198e-16, 0.46932117595418237389, -0.20099424927047284052, -0.04623946134063071740},
        {5.31597895759871264183, 1.19926790417132231573, 0.01179296240992997031, 0.01108844528519167989},
        {0.18188869982170434744, 0.05502798439925399070, 0.09351590770535414968, 0.00610700528898058230},
        {-2.0861320e-13, -0.13181061013830184015, -0.02027855540589259079, -0.00675951846863086359},
    };
    std::vector<at::Tensor> As = matrix_powers(A, 4);
    std::vector<at::Tensor> Bs;
    for (int i = 0; i < 4; i++) Bs.push_back(linear_combination(As, std::vector<double>(b[i], b[i] + 4)));
    Bs[2] = at::add(Bs[2], at::matmul(Bs[3], Bs[3]));
    return at::add(Bs[0], at::matmul(at::add(Bs[1], Bs[2]), Bs[2]));
}

at::Tensor matrix_exp_t8(const at::Tensor& A)
{
    const double sqrt_177 = 0.1330413469565007072504e+2;
    const double x3 = 2. / 3.;
    const double x1 = x3 * ((1. + sqrt_177) / 88.);
    const double x2 = x3 * ((1. + sqrt_177) / 352.);
    const double x4 = (-271. + 29. * sqrt_177) / (315. * x3);
    const double x5 = (-11. + 11. * sqrt_177) / (1260. * x3);
    const double x6 = (-99. + 11. * sqrt_177) / (5040. * x3);
    const double x7 = (89. - sqrt_177) / (5040. * x3 * x3);
    const double y2 = (857. - 58. * sqrt_177) / 630.;
    std::vector<at::Tensor> As = matrix_powers(A, 3);
    at::Tensor A3 = at::matmul(As[2], linear_combination({As[1], As[2]}, {x1, x2}));
    std::vector<at::Tensor> four = {As[0], As[1], As[2], A3};
    at::Tensor A4 = at::matmul(linear_combination({As[2], A3}, {x3, 1.0}), linear_combination(four, {x4, x5, x6, x7}));
    return linear_combination({As[0], As[1], As[2], A3, A4}, {1.0, 1.0, y2, 0.0, 1.0});
}

at::Tensor matrix_exp_t4(const at::Tensor& A)
{
    std::vector<at::Tensor> As = matrix_powers(A, 3);
    at::Tensor A3 = at::matmul(As[2], linear_combination({As[0], As[1], As[2]}, {1 / 2.0, 1 / 6.0, 1 / 24.0}));
    return linear_combination({As[0], As[1], As[2], A3}, {1.0, 1.0, 0.0, 1.0});
}

at::Tensor matrix_exp_t2(const at::Tensor& A)
{
    std::vector<at::Tensor> As = matrix_powers(A, 3);
    return at::add(at::add(As[0], As[1]), at::div(As[2], 2.0));
}

at::Tensor matrix_exp_t1(const at::Tensor& A)
{
    return at::add(at::eye(A.size(-1), A.options()).expand(A.sizes()), A);
}

at::Tensor matrix_exp_scale_square(const at::Tensor& a, const at::Tensor& norm, double theta)
{
    at::Tensor s = at::clamp_min(at::ceil(at::log2(at::div(norm, theta))), 0);
    at::Tensor scaled = at::mul(a, at::pow(2.0, at::neg(s)).view({-1, 1, 1}));
    at::Tensor result = matrix_exp_t18(scaled);
    at::Tensor squarings = s.to(at::kLong).cpu();
    for (int64_t i = 0; i < a.size(0); i++) {
        at::Tensor m = result.narrow(0, i, 1);
        for (int64_t j = 0; j < squarings[i].item<int64_t>(); j++) m = at::matmul(m, m);
        result.narrow(0, i, 1).copy_(m);
    }
    return result;
}

at::Tensor linalg_matrix_exp_impl(const at::Tensor& self, bool is_float);

at::Tensor linalg_matrix_exp_vulkan(const at::Tensor& self)
{
    check_square(self, "linalg.matrix_exp");
    TORCH_CHECK(at::isFloatingType(self.scalar_type()) || at::isComplexType(self.scalar_type()),
                "linalg.matrix_exp: Expected a floating point or complex tensor as input. Got ", self.scalar_type());
    int64_t n = self.size(-1);
    if (n == 0) return self.clone();
    if (n == 1) return at::exp(self);

    if (is_low_prec(self.scalar_type())) {
        return linalg_matrix_exp_impl(self.to(at::kFloat), true).to(self.scalar_type());
    }
    if (self.scalar_type() == at::kComplexHalf) return linalg_matrix_exp_impl(self.to(at::kComplexFloat), true).to(self.scalar_type());
    bool double_precision = self.scalar_type() == at::kDouble || self.scalar_type() == at::kComplexDouble;
    return linalg_matrix_exp_impl(self, !double_precision);
}

at::Tensor linalg_matrix_exp_impl(const at::Tensor& self, bool is_float)
{
    int64_t n = self.size(-1);
    std::array<double, 6> thetas = is_float
        ? std::array<double, 6>{1.192092800768788e-07, 5.978858893805233e-04, 5.116619363445086e-02, 5.800524627688768e-01, 1.461661507209034e+00, 3.010066362817634e+00}
        : std::array<double, 6>{2.220446049250313e-16, 2.580956802971767e-08, 3.397168839976962e-04, 4.991228871115323e-02, 2.996158913811580e-01, 1.090863719290036e+00};
    at::Tensor a = self.reshape({-1, n, n});
    at::Tensor norm = std::get<0>(at::max(at::sum(at::abs(a), -2), -1));
    if (a.size(0) > 1) return matrix_exp_scale_square(a, norm, thetas[5]).view(self.sizes());

    double value = norm.item<double>();
    at::Tensor result;
    if (value <= thetas[0]) result = matrix_exp_t1(a);
    else if (value <= thetas[1]) result = matrix_exp_t2(a);
    else if (value <= thetas[2]) result = matrix_exp_t4(a);
    else if (value <= thetas[3]) result = matrix_exp_t8(a);
    else if (value <= thetas[4]) result = matrix_exp_t12(a);
    else result = matrix_exp_scale_square(a, norm, thetas[5]);
    return result.view(self.sizes());
}

at::Tensor& linalg_cross_out_vulkan(const at::Tensor& self, const at::Tensor& other, int64_t dim, at::Tensor& out)
{
    at::assert_no_internal_overlap(out);
    at::assert_no_overlap(out, self);
    at::assert_no_overlap(out, other);
    TORCH_CHECK(self.dim() == other.dim(), "linalg.cross: inputs must have the same number of dimensions.");
    std::vector<int64_t> shape = at::infer_size(self.sizes(), other.sizes());
    int64_t d = c10::maybe_wrap_dim(dim, static_cast<int64_t>(shape.size()));
    TORCH_CHECK(self.size(d) == 3 && other.size(d) == 3, "linalg.cross: inputs dimension ", dim, " must have length 3. Got ", self.size(d), " and ", other.size(d));
    at::Tensor a = self.expand(shape), b = other.expand(shape);
    auto c = [&](const at::Tensor& t, int64_t i) { return t.narrow(d, i, 1); };
    at::Tensor x = at::sub(at::mul(c(a, 1), c(b, 2)), at::mul(c(a, 2), c(b, 1)));
    at::Tensor y = at::sub(at::mul(c(a, 2), c(b, 0)), at::mul(c(a, 0), c(b, 2)));
    at::Tensor z = at::sub(at::mul(c(a, 0), c(b, 1)), at::mul(c(a, 1), c(b, 0)));
    return fill_out(out, at::cat({x, y, z}, d));
}

at::Tensor linalg_pinv_vulkan(const at::Tensor& self, const c10::optional<at::Tensor>& atol, const c10::optional<at::Tensor>& rtol, bool hermitian)
{
    TORCH_CHECK(self.dim() >= 2, "linalg.pinv(", self.scalar_type(), "{", self.sizes(), "}): expected a tensor with 2 or more dimensions of float, double, cfloat or cdouble types");
    int64_t m = self.size(-2), n = self.size(-1);
    if (self.numel() == 0) return at::empty(with_matrix(batch_shape(self), n, m), self.options());

    at::Tensor U, S, Vh;
    if (hermitian) {
        at::Tensor values = at::empty({0}, self.options().dtype(c10::toRealValueType(self.scalar_type())));
        at::Tensor vectors = at::empty({0}, self.options());
        linalg_eigh_out_vulkan(self, "L", true, values, vectors);
        S = at::abs(values);
        U = at::mul(vectors, at::sign(values).unsqueeze(-2).to(vectors.scalar_type()));
        Vh = vectors.mH();
    } else {
        std::tie(U, S, Vh) = svd_impl(self, false, true);
    }
    at::Tensor largest = std::get<0>(at::max(S, -1, /*keepdim=*/true));
    bool has_atol = atol.has_value() && atol->defined();
    bool has_rtol = rtol.has_value() && rtol->defined();
    double eps = self.scalar_type() == at::kDouble ? std::numeric_limits<double>::epsilon() : std::numeric_limits<float>::epsilon();
    at::Tensor r = has_rtol ? rtol->unsqueeze(-1) : at::scalar_tensor((has_atol ? 0.0 : eps * static_cast<double>(std::max(m, n))), S.options());
    at::Tensor threshold = at::mul(largest, r);
    if (has_atol) threshold = at::maximum(threshold, atol->unsqueeze(-1));
    at::Tensor inverse = at::where(at::gt(S, threshold), at::reciprocal(S), at::zeros_like(S));
    // V S^+ U^H: conjugate transposes for complex input
    return at::matmul(at::mul(Vh.mH().resolve_conj(), inverse.unsqueeze(-2).to(Vh.scalar_type())), U.mH().resolve_conj());
}

at::Tensor& linalg_pinv_out_vulkan(const at::Tensor& self, const c10::optional<at::Tensor>& atol, const c10::optional<at::Tensor>& rtol, bool hermitian, at::Tensor& out)
{
    check_castable("linalg.pinv", "result", out, self.scalar_type());
    return fill_out(out, linalg_pinv_vulkan(self, atol, rtol, hermitian));
}

at::Tensor to_complex(const at::Tensor& real, const at::Tensor& imag, at::ScalarType dtype)
{
    return unview(at::view_as_complex(at::stack({real, imag}, -1).contiguous())).to(dtype);
}

// JAMA's orthes and hqr2; vectors normalized as LAPACK's geev: unit length, largest component real
std::tuple<at::Tensor, at::Tensor> eig_impl(const at::Tensor& A, bool compute_v);

// eigenvectors are defined up to a scalar: one convention on every device, unit norm with the largest entry real and positive
at::Tensor canonical_eigenvectors(const at::Tensor& v)
{
    if (v.size(-2) == 0) return v;
    at::Tensor unit = at::div(v, at::linalg_vector_norm(v, 2, {-2}, true));
    at::Tensor largest = at::gather(unit, -2, at::argmax(at::abs(unit), -2, /*keepdim=*/true));
    return at::mul(unit, at::div(at::abs(largest), largest));
}

// a complex eig from the real embedding's, which has every eigenvalue of A and its conjugate: A's are the ones whose
// eigenvector has the form [p; -i p], and p is A's eigenvector (an eigenvalue that is exactly real is ambiguous)
std::tuple<at::Tensor, at::Tensor> complex_eig(const at::Tensor& A, bool compute_v)
{
    int64_t n = A.size(-1);
    std::vector<int64_t> batch = batch_shape(A);
    at::ScalarType dtype = complex_compute_dtype(A.scalar_type());
    auto [w, vectors] = eig_impl(real_embedding(torchvulkan::complex_parts(A.to(dtype).reshape({batch_count(A), n, n}))), true);
    at::Tensor p = vectors.narrow(-2, 0, n), q = vectors.narrow(-2, n, n);
    at::Tensor score = at::linalg_vector_norm(at::add(q, at::mul(p, c10::complex<double>(0.0, 1.0))), 2, {-2}, false);
    at::Tensor chosen = std::get<0>(at::sort(std::get<1>(at::topk(score, n, -1, /*largest=*/false)), -1));
    std::vector<int64_t> value_shape = batch;
    value_shape.push_back(n);
    at::Tensor values = at::gather(w, -1, chosen).view(value_shape).to(A.scalar_type());
    if (!compute_v) return std::make_tuple(values, at::empty({0}, A.options()));
    at::Tensor v = at::gather(p, -1, chosen.unsqueeze(-2).expand({p.size(0), n, n}));
    return std::make_tuple(values, canonical_eigenvectors(v).view(with_matrix(batch, n, n)).to(A.scalar_type()));
}

std::tuple<at::Tensor, at::Tensor> eig_impl(const at::Tensor& A, bool compute_v)
{
    check_square(A, "linalg.eig");
    if (A.is_complex()) return complex_eig(A, compute_v);
    at::ScalarType dtype = linalg_dtype(A, "linalg.eig");
    at::ScalarType out_dtype = get_dtype_complex(A.scalar_type());
    int64_t n = A.size(-1);
    std::vector<int64_t> batch = batch_shape(A);
    at::Tensor h = batched_copy(A, dtype);
    at::Tensor v = at::empty_like(h);
    at::Tensor d = at::zeros({h.size(0), n}, h.options());
    at::Tensor e = at::zeros({h.size(0), n}, h.options());
    if (h.numel() > 0) {
        bool finite = at::isfinite(h).all().item<bool>();
        TORCH_CHECK(finite, "torch.linalg.eig: input tensor should not contain infs or NaNs.");
        dispatch_eig_shader(h, v, d, e);
    }
    std::vector<int64_t> value_shape = batch;
    value_shape.push_back(n);
    at::Tensor values = to_complex(d, e, out_dtype).view(value_shape);
    if (!compute_v) return std::make_tuple(values, at::empty({0}, A.options().dtype(out_dtype)));
    if (n == 0 || h.size(0) == 0) return std::make_tuple(values, at::empty(with_matrix(batch, n, n), A.options().dtype(out_dtype)));

    at::Tensor first = at::gt(e, 0).unsqueeze(1);
    at::Tensor second = at::lt(e, 0).unsqueeze(1);
    at::Tensor zero_column = at::zeros({h.size(0), n, 1}, v.options());
    at::Tensor next = at::cat({v.narrow(2, 1, std::max<int64_t>(n - 1, 0)), zero_column}, 2);
    at::Tensor previous = at::cat({zero_column, v.narrow(2, 0, std::max<int64_t>(n - 1, 0))}, 2);
    at::Tensor real = at::where(second, previous, v);
    at::Tensor imag = at::where(first, next, at::where(second, at::neg(v), at::zeros_like(v)));

    at::Tensor magnitude = at::add(at::mul(real, real), at::mul(imag, imag));
    at::Tensor norm = at::sqrt(at::sum(magnitude, 1, /*keepdim=*/true));
    real = at::div(real, norm);
    imag = at::div(imag, norm);
    at::Tensor largest = at::argmax(magnitude, 1, /*keepdim=*/true);
    at::Tensor pr = at::gather(real, 1, largest), pi = at::gather(imag, 1, largest);
    at::Tensor scale = at::sqrt(at::add(at::mul(pr, pr), at::mul(pi, pi)));
    at::Tensor cr = at::div(pr, scale), ci = at::neg(at::div(pi, scale));
    // eigenvectors are defined up to a scalar: one convention on every device, the largest entry real and positive
    at::Tensor rotated_real = at::sub(at::mul(real, cr), at::mul(imag, ci));
    at::Tensor rotated_imag = at::add(at::mul(real, ci), at::mul(imag, cr));
    at::Tensor vectors = to_complex(rotated_real, rotated_imag, out_dtype).view(with_matrix(batch, n, n));
    return std::make_tuple(values, vectors);
}

std::tuple<at::Tensor, at::Tensor> linalg_eig_vulkan(const at::Tensor& self)
{
    auto [values, vectors] = eig_impl(self, true);
    return std::make_tuple(values, column_major(vectors));
}

std::tuple<at::Tensor&, at::Tensor&> linalg_eig_out_vulkan(const at::Tensor& self, at::Tensor& eigenvalues, at::Tensor& eigenvectors)
{
    check_castable("torch.linalg.eig", "eigenvalues", eigenvalues, get_dtype_complex(self.scalar_type()));
    check_castable("torch.linalg.eig", "eigenvectors", eigenvectors, get_dtype_complex(self.scalar_type()));
    auto [values, vectors] = eig_impl(self, true);
    fill_out(eigenvalues, values);
    fill_out(eigenvectors, vectors);
    return std::forward_as_tuple(eigenvalues, eigenvectors);
}

at::Tensor linalg_eigvals_vulkan(const at::Tensor& self)
{
    return std::get<0>(eig_impl(self, false));
}

at::Tensor& linalg_eigvals_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    check_castable("torch.linalg.eigvals", "eigenvalues", out, get_dtype_complex(self.scalar_type()));
    return fill_out(out, linalg_eigvals_vulkan(self));
}

// the polar decomposition A = U H from the SVD A = W S V^H: U = W V^H, H = V S V^H
std::tuple<at::Tensor&, at::Tensor&> linalg_polar_out_vulkan(const at::Tensor& A, at::Tensor& U, at::Tensor& H)
{
    TORCH_CHECK(A.dim() >= 2, "linalg.polar: The input tensor A must have at least 2 dimensions.");
    TORCH_CHECK(A.size(-2) >= A.size(-1), "linalg.polar: input must have at least as many rows as columns, but got ",
        A.size(-2), " by ", A.size(-1), " matrices");
    auto [w, s, vh] = svd_impl(A, /*full_matrices=*/false, /*compute_uv=*/true);
    at::Tensor v = vh.mH();
    fill_out(U, at::matmul(w, vh));
    fill_out(H, at::matmul(at::mul(v, s.unsqueeze(-2).to(v.scalar_type())), vh).resolve_conj());
    return std::forward_as_tuple(U, H);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("linalg_matrix_sqrth", TORCH_FN(at::native::linalg_matrix_sqrth));
    m.impl("linalg_polar.out", &linalg_polar_out_vulkan);
    m.impl("linalg_eig", &linalg_eig_vulkan);
    m.impl("linalg_eig.out", &linalg_eig_out_vulkan);
    m.impl("_linalg_eigvals", &linalg_eigvals_vulkan);
    m.impl("linalg_eigvals.out", &linalg_eigvals_out_vulkan);
    m.impl("linalg_lu_factor_ex.out", &linalg_lu_factor_ex_out_vulkan);
    m.impl("linalg_lu.out", &linalg_lu_out_vulkan);
    m.impl("lu_unpack", &lu_unpack_vulkan);
    m.impl("lu_unpack.out", &lu_unpack_out_vulkan);
    m.impl("linalg_lu_solve.out", &linalg_lu_solve_out_vulkan);
    m.impl("linalg_lu_solve", &linalg_lu_solve_vulkan);
    m.impl("_linalg_solve_ex.result", &linalg_solve_ex_out_vulkan);
    m.impl("linalg_inv_ex.inverse", &linalg_inv_ex_out_vulkan);
    m.impl("_linalg_det.result", &linalg_det_out_vulkan);
    m.impl("_linalg_slogdet.sign", &linalg_slogdet_out_vulkan);
    m.impl("linalg_cholesky_ex.L", &linalg_cholesky_ex_out_vulkan);
    m.impl("cholesky", &cholesky_vulkan);
    m.impl("cholesky.out", &cholesky_out_vulkan);
    m.impl("_cholesky_solve_helper", &cholesky_solve_helper_vulkan);
    m.impl("cholesky_inverse", &cholesky_inverse_vulkan);
    m.impl("cholesky_inverse.out", &cholesky_inverse_out_vulkan);
    m.impl("linalg_solve_triangular", &linalg_solve_triangular_vulkan);
    m.impl("linalg_solve_triangular.out", &linalg_solve_triangular_out_vulkan);
    m.impl("triangular_solve.X", &triangular_solve_out_vulkan);
    m.impl("geqrf", &geqrf_vulkan);
    m.impl("geqrf.a", &geqrf_out_vulkan);
    m.impl("linalg_householder_product", &linalg_householder_product_vulkan);
    m.impl("linalg_householder_product.out", &linalg_householder_product_out_vulkan);
    m.impl("ormqr", &ormqr_vulkan);
    m.impl("ormqr.out", &ormqr_out_vulkan);
    m.impl("linalg_qr.out", &linalg_qr_out_vulkan);
    m.impl("_linalg_svd.U", &linalg_svd_out_vulkan);
    m.impl("_linalg_eigh.eigenvalues", &linalg_eigh_out_vulkan);
    m.impl("linalg_ldl_factor_ex.out", &linalg_ldl_factor_ex_out_vulkan);
    m.impl("linalg_ldl_solve.out", &linalg_ldl_solve_out_vulkan);
    m.impl("linalg_lstsq.out", &linalg_lstsq_out_vulkan);
    m.impl("linalg_matrix_exp", &linalg_matrix_exp_vulkan);
    m.impl("linalg_cross.out", &linalg_cross_out_vulkan);
    m.impl("linalg_pinv.atol_rtol_tensor", &linalg_pinv_vulkan);
    m.impl("linalg_pinv.atol_rtol_tensor_out", &linalg_pinv_out_vulkan);
}
