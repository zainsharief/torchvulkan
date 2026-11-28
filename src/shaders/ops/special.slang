module special;

__exported import core.special_functions;
import core.indexing;

[[vk::constant_id(0)]] public const uint OP_TYPE = 0;
[[vk::constant_id(1)]] public const uint IS_CONTIG = 0;
[[vk::constant_id(2)]] public const uint USE_SCALAR = 0;
[[vk::constant_id(3)]] public const uint NDIM = 0;
[[vk::constant_id(4)]] public const uint WG_x = 0;

public struct Metadata
{
    public uint64_t sizes;
    public uint64_t strides_a;
    public uint64_t strides_b;
    public uint64_t strides_out;
}

public struct SpecialPushConstants
{
    public uint64_t tensor_a;
    public uint64_t tensor_b;
    public uint64_t tensor_out;
    public uint64_t metadata;
    public uint64_t numel;
    public float scalar_value;
}

// must match SpecialOp in special.cpp
float evalSpecial(float a, float b)
{
    switch (OP_TYPE)
    {
    case 0:  return airy_ai_forward(a);
    case 1:  return bessel_j0_forward(a);
    case 2:  return bessel_j1_forward(a);
    case 3:  return bessel_y0_forward(a);
    case 4:  return bessel_y1_forward(a);
    case 5:  return modified_bessel_i0_forward(a);
    case 6:  return modified_bessel_i1_forward(a);
    case 7:  return modified_bessel_k0_forward(a);
    case 8:  return modified_bessel_k1_forward(a);
    case 9:  return scaled_modified_bessel_k0_forward(a);
    case 10: return scaled_modified_bessel_k1_forward(a);
    case 11: return spherical_bessel_j0_forward(a);
    case 12: return ndtri(a);
    case 13: return igamma(a, b);
    case 14: return igammac(a, b);
    case 15: return zeta(a, b);
    case 16: return polygamma(a, int(b));
    case 17: return chebyshev_polynomial_t_forward(a, b);
    case 18: return chebyshev_polynomial_u_forward(a, b);
    case 19: return chebyshev_polynomial_v_forward(a, b);
    case 20: return chebyshev_polynomial_w_forward(a, b);
    case 21: return shifted_chebyshev_polynomial_t_forward(a, b);
    case 22: return shifted_chebyshev_polynomial_u_forward(a, b);
    case 23: return shifted_chebyshev_polynomial_v_forward(a, b);
    case 24: return shifted_chebyshev_polynomial_w_forward(a, b);
    case 25: return hermite_polynomial_h_forward(a, b);
    case 26: return hermite_polynomial_he_forward(a, b);
    case 27: return laguerre_polynomial_l_forward(a, b);
    case 28: return legendre_polynomial_p_forward(a, b);
    default: return a;
    }
}

public void specialKernel<T : ITorchVulkan>(uint3 gi, SpecialPushConstants pcs)
{
    uint idx = flat_thread(gi, WG_x);
    if (idx >= pcs.numel) return;


    uint idx_a = idx;
    uint idx_b = idx;
    uint idx_out = idx;

    if (IS_CONTIG == 0)
    {
        Metadata md = *(Metadata*)pcs.metadata;
        uint* strides_a = (uint*)md.strides_a;
        uint* strides_b = (uint*)md.strides_b;
        uint* strides_out = (uint*)md.strides_out;

        idx_a = 0;
        idx_b = 0;
        idx_out = 0;

        var walker = IndexWalker(md.sizes, NDIM, idx);
        for (uint i = 0; i < NDIM; i++)
        {
            uint coord = walker.next(i);
            idx_a += coord * strides_a[i];
            idx_b += coord * strides_b[i];
            idx_out += coord * strides_out[i];
        }
    }

    float a = castValue<T, float>(T.load(pcs.tensor_a, idx_a));
    float b = pcs.scalar_value;
    if (USE_SCALAR == 0) b = castValue<T, float>(T.load(pcs.tensor_b, idx_b));
    T.store(pcs.tensor_out, idx_out, castValue<float, T>(evalSpecial(a, b)));
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void special_main<T : ITorchVulkan>(
    uint3 gi : SV_DispatchThreadID,
    uniform SpecialPushConstants pcs)
{
    specialKernel<T>(gi, pcs);
}
