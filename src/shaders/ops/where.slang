module where;

__exported import core.dtype;
import core.indexing;

[[vk::constant_id(0)]] public const uint IS_CONTIG = 0;
[[vk::constant_id(1)]] public const uint NDIM = 0;
[[vk::constant_id(2)]] public const uint WG_x = 0;

public struct Metadata
{
    public uint64_t sizes;
    public uint64_t strides_cond;
    public uint64_t strides_a;
    public uint64_t strides_b;
}

public struct WherePushConstants
{
    public uint64_t tensor_cond;
    public uint64_t tensor_a;
    public uint64_t tensor_b;
    public uint64_t tensor_out;
    public uint64_t metadata;
    public uint64_t numel;
}

// cond is a bool tensor, loaded as B
public void whereKernel<T : ITorchVulkan, B : ITorchVulkan>(
    uint3 gi,
    uint64_t address_cond,
    uint64_t address_a,
    uint64_t address_b,
    uint64_t address_out,
    uint64_t address_metadata,
    uint64_t numel)
{
    uint idx = flat_thread(gi, WG_x);
    if (idx >= numel) return;


    uint idx_c = idx;
    uint idx_a = idx;
    uint idx_b = idx;

    if (IS_CONTIG == 0)
    {
        Metadata md = *(Metadata*)address_metadata;
        uint* strides_cond = (uint*)md.strides_cond;
        uint* strides_a = (uint*)md.strides_a;
        uint* strides_b = (uint*)md.strides_b;

        idx_c = 0;
        idx_a = 0;
        idx_b = 0;

        var walker = IndexWalker(md.sizes, NDIM, idx);
        for (uint i = 0; i < NDIM; i++)
        {
            uint coord = walker.next(i);
            idx_c += coord * strides_cond[i];
            idx_a += coord * strides_a[i];
            idx_b += coord * strides_b[i];
        }
    }

    T.store(address_out, idx, B.ext_nonzero(B.load(address_cond, idx_c)) ? T.load(address_a, idx_a) : T.load(address_b, idx_b));
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void where_main<T : ITorchVulkan, B : ITorchVulkan>(
    uint3 gi : SV_DispatchThreadID,
    uniform WherePushConstants pcs)
{
    whereKernel<T, B>(gi, pcs.tensor_cond, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel);
}
