module cast;

__exported import core.dtype;
import core.indexing;

[[vk::constant_id(0)]] public const uint NDIM = 0;
[[vk::constant_id(1)]] public const uint WG_x = 0;
[[vk::constant_id(2)]] public const uint IS_BOOL_CAST = 0;
[[vk::constant_id(3)]] public const uint IS_CONTIG = 0;

public struct Metadata
{
    public uint64_t sizes;
    public uint64_t strides_in;
    public uint64_t strides_out;
}

public struct CastPushConstants
{
    public uint64_t tensor_in;
    public uint64_t tensor_out;
    public uint64_t metadata;
    public uint64_t numel;
}

Dst convertElement<Src : ITorchVulkan, Dst : ITorchVulkan>(Src v)
{
    if (IS_BOOL_CAST == 1) return Dst.castFromI64(Src.ext_nonzero(v) ? 1 : 0);
    return castValue<Src, Dst>(v);
}

public void castKernel<Src : ITorchVulkan, Dst : ITorchVulkan, let VEC_SIZE : int>(
    uint3 gi,
    uint64_t address_in,
    uint64_t address_out,
    uint64_t address_metadata,
    uint64_t numel)
{
    uint idx = gi.x;

    if (IS_CONTIG != 0)
    {
        if (idx * VEC_SIZE >= numel) return;

        Src* tensor_in = (Src*)address_in;
        Dst* tensor_out = (Dst*)address_out;

        Src v[VEC_SIZE];
        [ForceUnroll] for (int k = 0; k < VEC_SIZE; k++) v[k] = tensor_in[idx * VEC_SIZE + k];

        Dst res[VEC_SIZE];
        [ForceUnroll] for (int k = 0; k < VEC_SIZE; k++)
        {
            res[k] = convertElement<Src, Dst>(v[k]);
        }

        uint64_t remainder = numel - (idx * VEC_SIZE);
        uint64_t count = (remainder >= VEC_SIZE) ? VEC_SIZE : remainder;
        for (uint i = 0; i < count; i++) tensor_out[idx * VEC_SIZE + i] = res[i];

        return;
    }

    if (idx >= numel) return;

    Src* tensor_in = (Src*)address_in;
    Dst* tensor_out = (Dst*)address_out;

    Metadata md = *(Metadata*)address_metadata;
    uint* strides_in = (uint*)md.strides_in;
    uint* strides_out = (uint*)md.strides_out;

    uint idx_in = 0;
    uint idx_out = 0;

    var walker = IndexWalker(md.sizes, NDIM, idx);
    for (uint i = 0; i < NDIM; i++)
    {
        uint coord = walker.next(i);
        idx_in += coord * strides_in[i];
        idx_out += coord * strides_out[i];
    }

    Src value = tensor_in[idx_in];
    tensor_out[idx_out] = convertElement<Src, Dst>(value);
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void cast_main<Src : ITorchVulkan, Dst : ITorchVulkan, let VEC_SIZE : int>(
    uint3 gi : SV_DispatchThreadID,
    uniform CastPushConstants pcs)
{
    castKernel<Src, Dst, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel);
}
