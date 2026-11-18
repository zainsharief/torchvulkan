module addc;

__exported import core.dtype;
import core.indexing;

[[vk::constant_id(0)]] public const uint IS_CONTIG = 0;
[[vk::constant_id(1)]] public const uint NDIM = 0;
[[vk::constant_id(2)]] public const uint WG_x = 0;
[[vk::constant_id(3)]] public const uint DIVIDE = 0;
[[vk::constant_id(4)]] public const uint HAS_VALUE = 0;

public struct Metadata
{
    public uint64_t sizes;
    public uint64_t strides_self;
    public uint64_t strides_tensor1;
    public uint64_t strides_tensor2;
}

public struct AddcPushConstants<T : ITorchVulkan>
{
    public uint64_t tensor_self;
    public uint64_t tensor1;
    public uint64_t tensor2;
    public uint64_t tensor_out;
    public uint64_t metadata;
    public uint64_t numel;
    public T.Accum value;
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void addc_main<T : ITorchVulkan>(
    uint3 gi : SV_DispatchThreadID,
    uniform AddcPushConstants<T> pcs)
{
    uint idx = flat_thread(gi, WG_x);
    if (idx >= pcs.numel) return;

    uint idx_self = idx;
    uint idx_1 = idx;
    uint idx_2 = idx;
    if (IS_CONTIG == 0)
    {
        Metadata md = *(Metadata*)pcs.metadata;
        uint* strides_self = (uint*)md.strides_self;
        uint* strides_1 = (uint*)md.strides_tensor1;
        uint* strides_2 = (uint*)md.strides_tensor2;
        idx_self = 0;
        idx_1 = 0;
        idx_2 = 0;
        var walker = IndexWalker(md.sizes, NDIM, idx);
        for (uint i = 0; i < NDIM; i++)
        {
            uint coord = walker.next(i);
            idx_self += coord * strides_self[i];
            idx_1 += coord * strides_1[i];
            idx_2 += coord * strides_2[i];
        }
    }

    T.Accum self = ((T*)pcs.tensor_self)[idx_self].toAccum();
    T.Accum a = ((T*)pcs.tensor1)[idx_1].toAccum();
    T.Accum b = ((T*)pcs.tensor2)[idx_2].toAccum();
    T.Accum combined;
    if (DIVIDE != 0) combined = HAS_VALUE != 0 ? pcs.value * T.Accum.ext_div(a, b) : T.Accum.ext_div(a, b);
    else combined = HAS_VALUE != 0 ? (pcs.value * a) * b : a * b;
    ((T*)pcs.tensor_out)[idx] = T.fromAccum(self + combined);
}
