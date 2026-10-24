#include <torch/extension.h>
#include <ATen/TensorIterator.h>
#include <functional>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "api/ops/helpers.h"
#include "api/fallback.h"

namespace {

enum class SpecialOp {
    AIRY_AI = 0,
    BESSEL_J0 = 1,
    BESSEL_J1 = 2,
    BESSEL_Y0 = 3,
    BESSEL_Y1 = 4,
    MODIFIED_BESSEL_I0 = 5,
    MODIFIED_BESSEL_I1 = 6,
    MODIFIED_BESSEL_K0 = 7,
    MODIFIED_BESSEL_K1 = 8,
    SCALED_MODIFIED_BESSEL_K0 = 9,
    SCALED_MODIFIED_BESSEL_K1 = 10,
    SPHERICAL_BESSEL_J0 = 11,
    NDTRI = 12,
    IGAMMA = 13,
    IGAMMAC = 14,
    ZETA = 15,
    POLYGAMMA = 16,
    CHEBYSHEV_T = 17,
    CHEBYSHEV_U = 18,
    CHEBYSHEV_V = 19,
    CHEBYSHEV_W = 20,
    SHIFTED_CHEBYSHEV_T = 21,
    SHIFTED_CHEBYSHEV_U = 22,
    SHIFTED_CHEBYSHEV_V = 23,
    SHIFTED_CHEBYSHEV_W = 24,
    HERMITE_H = 25,
    HERMITE_HE = 26,
    LAGUERRE_L = 27,
    LEGENDRE_P = 28
};

bool is_cpu_scalar(const at::Tensor& t)
{
    return t.unsafeGetTensorImpl()->is_wrapped_number() || (t.dim() == 0 && t.device().is_cpu());
}

// float32 whatever the floating dtype, like the other transcendentals
bool dispatch_special_shader(
    const at::Tensor& self,
    const at::Tensor& other,
    double scalar,
    const at::Tensor& out,
    SpecialOp operation)
{
    DeviceContext* device = get_device_context(out);
    at::ScalarType dtype = out.scalar_type();
    if (!is_dtype_supported(device, dtype)) return false;

    bool use_scalar = !other.defined();
    at::Tensor a = self.to(out.device(), dtype);
    at::Tensor b = use_scalar ? at::Tensor() : other.to(out.device(), dtype);

    at::TensorIteratorConfig config;
    config.set_check_mem_overlap(true).resize_outputs(false).add_output(out).add_input(a);
    if (!use_scalar) config.add_input(b);
    at::TensorIterator iter = config.build();

    uint64_t numel = iter.numel();
    if (numel == 0) return true;
    int32_t out_dims = static_cast<int32_t>(iter.ndim());
    if (out_dims > MAX_DIMS || !fits_32bit_indexing(iter)) return false;

    uint32_t workgroupSizeX = get_dtype_workgroup_size(dtype, 1, device);
    uint32_t contiguous = iter.is_contiguous() ? 1 : 0;

    SpecializationBuilder spd{};
    spd.push(static_cast<uint32_t>(operation))
       .push(contiguous)
       .push(use_scalar ? 1u : 0u)
       .push(out_dims)
       .push(workgroupSizeX);

    MetadataBuilder metadataBuilder = get_iter_metadata(iter, {1, use_scalar ? -1 : 2, 0});
    Metadata metadata = metadataBuilder.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(a))
       .push(use_scalar ? uint64_t(0) : get_tensor_address(b))
       .push(get_tensor_address(out))
       .push_metadata(contiguous ? nullptr : &metadata)
       .push(numel)
       .push(static_cast<float>(scalar));

    std::vector<at::Tensor> reads = {a};
    if (!use_scalar) reads.push_back(b);

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::SPECIAL, dtype},
        spd.build(),
        pcs.build(),
        /* read = */ reads,
        /* write = */ {out},
        get_group_count(numel, workgroupSizeX), 1, 1
    );
    return true;
}

at::ScalarType special_result_type(const at::Tensor& self, const at::Tensor& other)
{
    at::ScalarType dtype = other.defined() ? at::result_type(self, other) : self.scalar_type();
    if (at::isIntegralType(dtype, /*includeBool=*/true)) dtype = c10::typeMetaToScalarType(at::get_default_dtype());
    return dtype;
}

at::Tensor& special_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out,
    SpecialOp operation,
    const std::function<at::Tensor(const at::Tensor&, const at::Tensor&)>& fallback)
{
    at::ScalarType dtype = special_result_type(self, other);
    TORCH_CHECK(at::canCast(dtype, out.scalar_type()), "result type ", dtype, " can't be cast to the desired output type ", out.scalar_type());

    bool scalar_other = other.defined() && is_cpu_scalar(other) && !is_cpu_scalar(self);
    at::Tensor lhs = is_cpu_scalar(self) && other.defined() ? self.to(other.device()) : self;
    std::vector<int64_t> shape = (!other.defined() || scalar_other) ? lhs.sizes().vec() : at::infer_size(lhs.sizes(), other.sizes());

    at::Tensor result = out.scalar_type() == dtype ? out : at::empty({0}, out.options().dtype(dtype));
    if (result.sizes() != shape) result.resize_(shape);

    bool ran = scalar_other
        ? dispatch_special_shader(lhs, at::Tensor(), other.item().toDouble(), result, operation)
        : dispatch_special_shader(lhs, other, 0.0, result, operation);
    if (!ran) {
        TORCHVULKAN_CPU_FALLBACK("A special function on ", dtype, " can't run on the Vulkan device. Falling back to CPU.");
        result.copy_(fallback(self.cpu(), other.defined() ? other.cpu() : other));
    }
    if (!result.is_same(out)) fill_out(out, result);
    return out;
}

at::Tensor& airy_ai_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return special_out(self, at::Tensor(), out, SpecialOp::AIRY_AI, [](const at::Tensor& a, const at::Tensor&) { return at::special_airy_ai(a); });
}

at::Tensor& bessel_j0_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return special_out(self, at::Tensor(), out, SpecialOp::BESSEL_J0, [](const at::Tensor& a, const at::Tensor&) { return at::special_bessel_j0(a); });
}

at::Tensor& bessel_j1_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return special_out(self, at::Tensor(), out, SpecialOp::BESSEL_J1, [](const at::Tensor& a, const at::Tensor&) { return at::special_bessel_j1(a); });
}

at::Tensor& bessel_y0_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return special_out(self, at::Tensor(), out, SpecialOp::BESSEL_Y0, [](const at::Tensor& a, const at::Tensor&) { return at::special_bessel_y0(a); });
}

at::Tensor& bessel_y1_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return special_out(self, at::Tensor(), out, SpecialOp::BESSEL_Y1, [](const at::Tensor& a, const at::Tensor&) { return at::special_bessel_y1(a); });
}

at::Tensor& modified_bessel_i0_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return special_out(self, at::Tensor(), out, SpecialOp::MODIFIED_BESSEL_I0, [](const at::Tensor& a, const at::Tensor&) { return at::special_modified_bessel_i0(a); });
}

at::Tensor& modified_bessel_i1_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return special_out(self, at::Tensor(), out, SpecialOp::MODIFIED_BESSEL_I1, [](const at::Tensor& a, const at::Tensor&) { return at::special_modified_bessel_i1(a); });
}

at::Tensor& modified_bessel_k0_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return special_out(self, at::Tensor(), out, SpecialOp::MODIFIED_BESSEL_K0, [](const at::Tensor& a, const at::Tensor&) { return at::special_modified_bessel_k0(a); });
}

at::Tensor& modified_bessel_k1_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return special_out(self, at::Tensor(), out, SpecialOp::MODIFIED_BESSEL_K1, [](const at::Tensor& a, const at::Tensor&) { return at::special_modified_bessel_k1(a); });
}

at::Tensor& scaled_modified_bessel_k0_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return special_out(self, at::Tensor(), out, SpecialOp::SCALED_MODIFIED_BESSEL_K0, [](const at::Tensor& a, const at::Tensor&) { return at::special_scaled_modified_bessel_k0(a); });
}

at::Tensor& scaled_modified_bessel_k1_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return special_out(self, at::Tensor(), out, SpecialOp::SCALED_MODIFIED_BESSEL_K1, [](const at::Tensor& a, const at::Tensor&) { return at::special_scaled_modified_bessel_k1(a); });
}

at::Tensor& spherical_bessel_j0_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return special_out(self, at::Tensor(), out, SpecialOp::SPHERICAL_BESSEL_J0, [](const at::Tensor& a, const at::Tensor&) { return at::special_spherical_bessel_j0(a); });
}

at::Tensor& ndtri_out_vulkan(const at::Tensor& self, at::Tensor& out)
{
    return special_out(self, at::Tensor(), out, SpecialOp::NDTRI, [](const at::Tensor& a, const at::Tensor&) { return at::special_ndtri(a); });
}

at::Tensor& igamma_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out)
{
    return special_out(self, other, out, SpecialOp::IGAMMA, [](const at::Tensor& a, const at::Tensor& b) { return at::igamma(a, b); });
}

at::Tensor& igammac_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out)
{
    return special_out(self, other, out, SpecialOp::IGAMMAC, [](const at::Tensor& a, const at::Tensor& b) { return at::igammac(a, b); });
}

at::Tensor& zeta_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out)
{
    return special_out(self, other, out, SpecialOp::ZETA, [](const at::Tensor& a, const at::Tensor& b) { return at::special_zeta(a, b); });
}

at::Tensor& polygamma_out_vulkan(int64_t n, const at::Tensor& self, at::Tensor& out)
{
    TORCH_CHECK(n >= 0, "polygamma(n, x) does not support negative n.");
    at::Tensor order = at::scalar_tensor(n, at::kLong);
    order.unsafeGetTensorImpl()->set_wrapped_number(true);
    return special_out(self, order, out, SpecialOp::POLYGAMMA, [n](const at::Tensor& a, const at::Tensor&) { return at::polygamma(n, a); });
}

at::Tensor& chebyshev_t_out_vulkan(const at::Tensor& x, const at::Tensor& n, at::Tensor& out)
{
    return special_out(x, n, out, SpecialOp::CHEBYSHEV_T, [](const at::Tensor& a, const at::Tensor& b) { return at::special_chebyshev_polynomial_t(a, b); });
}

at::Tensor& chebyshev_u_out_vulkan(const at::Tensor& x, const at::Tensor& n, at::Tensor& out)
{
    return special_out(x, n, out, SpecialOp::CHEBYSHEV_U, [](const at::Tensor& a, const at::Tensor& b) { return at::special_chebyshev_polynomial_u(a, b); });
}

at::Tensor& chebyshev_v_out_vulkan(const at::Tensor& x, const at::Tensor& n, at::Tensor& out)
{
    return special_out(x, n, out, SpecialOp::CHEBYSHEV_V, [](const at::Tensor& a, const at::Tensor& b) { return at::special_chebyshev_polynomial_v(a, b); });
}

at::Tensor& chebyshev_w_out_vulkan(const at::Tensor& x, const at::Tensor& n, at::Tensor& out)
{
    return special_out(x, n, out, SpecialOp::CHEBYSHEV_W, [](const at::Tensor& a, const at::Tensor& b) { return at::special_chebyshev_polynomial_w(a, b); });
}

at::Tensor& shifted_chebyshev_t_out_vulkan(const at::Tensor& x, const at::Tensor& n, at::Tensor& out)
{
    return special_out(x, n, out, SpecialOp::SHIFTED_CHEBYSHEV_T, [](const at::Tensor& a, const at::Tensor& b) { return at::special_shifted_chebyshev_polynomial_t(a, b); });
}

at::Tensor& shifted_chebyshev_u_out_vulkan(const at::Tensor& x, const at::Tensor& n, at::Tensor& out)
{
    return special_out(x, n, out, SpecialOp::SHIFTED_CHEBYSHEV_U, [](const at::Tensor& a, const at::Tensor& b) { return at::special_shifted_chebyshev_polynomial_u(a, b); });
}

at::Tensor& shifted_chebyshev_v_out_vulkan(const at::Tensor& x, const at::Tensor& n, at::Tensor& out)
{
    return special_out(x, n, out, SpecialOp::SHIFTED_CHEBYSHEV_V, [](const at::Tensor& a, const at::Tensor& b) { return at::special_shifted_chebyshev_polynomial_v(a, b); });
}

at::Tensor& shifted_chebyshev_w_out_vulkan(const at::Tensor& x, const at::Tensor& n, at::Tensor& out)
{
    return special_out(x, n, out, SpecialOp::SHIFTED_CHEBYSHEV_W, [](const at::Tensor& a, const at::Tensor& b) { return at::special_shifted_chebyshev_polynomial_w(a, b); });
}

at::Tensor& hermite_h_out_vulkan(const at::Tensor& x, const at::Tensor& n, at::Tensor& out)
{
    return special_out(x, n, out, SpecialOp::HERMITE_H, [](const at::Tensor& a, const at::Tensor& b) { return at::special_hermite_polynomial_h(a, b); });
}

at::Tensor& hermite_he_out_vulkan(const at::Tensor& x, const at::Tensor& n, at::Tensor& out)
{
    return special_out(x, n, out, SpecialOp::HERMITE_HE, [](const at::Tensor& a, const at::Tensor& b) { return at::special_hermite_polynomial_he(a, b); });
}

at::Tensor& laguerre_l_out_vulkan(const at::Tensor& x, const at::Tensor& n, at::Tensor& out)
{
    return special_out(x, n, out, SpecialOp::LAGUERRE_L, [](const at::Tensor& a, const at::Tensor& b) { return at::special_laguerre_polynomial_l(a, b); });
}

at::Tensor& legendre_p_out_vulkan(const at::Tensor& x, const at::Tensor& n, at::Tensor& out)
{
    return special_out(x, n, out, SpecialOp::LEGENDRE_P, [](const at::Tensor& a, const at::Tensor& b) { return at::special_legendre_polynomial_p(a, b); });
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("special_airy_ai.out", &airy_ai_out_vulkan);
    m.impl("special_bessel_j0.out", &bessel_j0_out_vulkan);
    m.impl("special_bessel_j1.out", &bessel_j1_out_vulkan);
    m.impl("special_bessel_y0.out", &bessel_y0_out_vulkan);
    m.impl("special_bessel_y1.out", &bessel_y1_out_vulkan);
    m.impl("special_modified_bessel_i0.out", &modified_bessel_i0_out_vulkan);
    m.impl("special_modified_bessel_i1.out", &modified_bessel_i1_out_vulkan);
    m.impl("special_modified_bessel_k0.out", &modified_bessel_k0_out_vulkan);
    m.impl("special_modified_bessel_k1.out", &modified_bessel_k1_out_vulkan);
    m.impl("special_scaled_modified_bessel_k0.out", &scaled_modified_bessel_k0_out_vulkan);
    m.impl("special_scaled_modified_bessel_k1.out", &scaled_modified_bessel_k1_out_vulkan);
    m.impl("special_spherical_bessel_j0.out", &spherical_bessel_j0_out_vulkan);
    m.impl("special_ndtri.out", &ndtri_out_vulkan);
    m.impl("igamma.out", &igamma_out_vulkan);
    m.impl("igammac.out", &igammac_out_vulkan);
    m.impl("special_zeta.out", &zeta_out_vulkan);
    m.impl("polygamma.out", &polygamma_out_vulkan);
    m.impl("special_chebyshev_polynomial_t.out", &chebyshev_t_out_vulkan);
    m.impl("special_chebyshev_polynomial_u.out", &chebyshev_u_out_vulkan);
    m.impl("special_chebyshev_polynomial_v.out", &chebyshev_v_out_vulkan);
    m.impl("special_chebyshev_polynomial_w.out", &chebyshev_w_out_vulkan);
    m.impl("special_shifted_chebyshev_polynomial_t.out", &shifted_chebyshev_t_out_vulkan);
    m.impl("special_shifted_chebyshev_polynomial_u.out", &shifted_chebyshev_u_out_vulkan);
    m.impl("special_shifted_chebyshev_polynomial_v.out", &shifted_chebyshev_v_out_vulkan);
    m.impl("special_shifted_chebyshev_polynomial_w.out", &shifted_chebyshev_w_out_vulkan);
    m.impl("special_hermite_polynomial_h.out", &hermite_h_out_vulkan);
    m.impl("special_hermite_polynomial_he.out", &hermite_he_out_vulkan);
    m.impl("special_laguerre_polynomial_l.out", &laguerre_l_out_vulkan);
    m.impl("special_legendre_polynomial_p.out", &legendre_p_out_vulkan);
}
