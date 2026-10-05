module fill;

__exported import core.dtype;
import core.indexing;

[[vk::constant_id(0)]] public const uint NDIM = 0;
[[vk::constant_id(1)]] public const uint IS_CONTIG = 0;
[[vk::constant_id(2)]] public const uint WG_x = 0;

public struct Metadata
{
    public uint64_t sizes;
    public uint64_t strides;
}

public struct FillPushConstants<T>
{
    public uint64_t tensor;
    public uint64_t metadata;
    public uint64_t numel;
    public T fill_value;
}

public void fillKernel<T, let VEC_SIZE : int>(
    uint3 gi,
    uint64_t address,
    uint64_t address_metadata,
    uint64_t numel,
    T fill_value)
{
    uint idx = gi.x;
    T* tensor = (T*)address;

    if (IS_CONTIG == 1)
    {
        if (idx * VEC_SIZE >= numel) return;

        uint64_t remainder = numel - (idx * VEC_SIZE);
        uint64_t count = (remainder >= VEC_SIZE) ? VEC_SIZE : remainder;
        for (uint i = 0; i < count; i++) tensor[idx * VEC_SIZE + i] = fill_value;

        return;
    }

    if (idx >= numel) return;

    Metadata md = *(Metadata*)address_metadata;
    uint* strides = (uint*)md.strides;

    uint idx_in = 0;

    var walker = IndexWalker(md.sizes, NDIM, idx);
    for (uint i = 0; i < NDIM; i++) idx_in += walker.next(i) * strides[i];

    tensor[idx_in] = fill_value;
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void fill_main<T, let VEC_SIZE : int>(
    uint3 gi : SV_DispatchThreadID,
    uniform FillPushConstants<T> pcs)
{
    fillKernel<T, VEC_SIZE>(gi, pcs.tensor, pcs.metadata, pcs.numel, pcs.fill_value);
}
