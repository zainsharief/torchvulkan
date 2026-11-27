module copy;

import core.indexing;
import core.dtype;

[[vk::constant_id(0)]] public const uint NDIM = 0;
[[vk::constant_id(1)]] public const uint WG_x = 0;

public struct Metadata
{
    public uint64_t sizes;
    public uint64_t strides_in;
    public uint64_t strides_out;
}

public struct CopyPushConstants
{
    public uint64_t tensor_in;
    public uint64_t tensor_out;
    public uint64_t metadata;
    public uint64_t numel;
}

public void copyKernel<M : IElementMover>(
    uint3 gi,
    uint64_t address_in,
    uint64_t address_out,
    uint64_t address_metadata,
    uint64_t numel)
{
    uint idx = flat_thread(gi, WG_x);
    if (idx >= numel) return;

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

    M.move(address_out, idx_out, address_in, idx_in);
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void copy_main<T>(
    uint3 gi : SV_DispatchThreadID,
    uniform CopyPushConstants pcs)
{
    copyKernel<PointerMover<T>>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel);
}

// 1- and 2-byte elements on devices without 8/16-bit storage, as their emulated types
[shader("compute")]
[numthreads(WG_x, 1, 1)]
void copy_emulated_main<T : ITorchVulkan>(
    uint3 gi : SV_DispatchThreadID,
    uniform CopyPushConstants pcs)
{
    copyKernel<ElementMover<T>>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel);
}
