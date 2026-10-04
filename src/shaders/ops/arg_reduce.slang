module arg_reduce;

__exported import core.dtype;
import core.indexing;

[[vk::constant_id(0)]] public const uint OP_TYPE = 0;
[[vk::constant_id(1)]] public const uint NDIM = 0;
[[vk::constant_id(2)]] public const uint WG_x = 0;

public struct Metadata
{
    public uint64_t sizes;
    public uint64_t strides_in;
}

public struct ArgReducePushConstants
{
    public uint64_t tensor_in;
    public uint64_t tensor_out;
    public uint64_t metadata;
    public uint64_t num_lines;
    public uint reduce_stride;
    public uint reduce_size;
}

public void argReduceKernel<T : ITorchVulkan>(
    uint3 gi,
    uint64_t address_in,
    uint64_t address_out,
    uint64_t address_metadata,
    uint64_t num_lines,
    uint reduce_stride,
    uint reduce_size)
{
    uint idx = gi.x;
    if (idx >= num_lines || reduce_size == 0) return;

    T* tensor_in = (T*)address_in;
    int64_t* tensor_out = (int64_t*)address_out;

    Metadata md = *(Metadata*)address_metadata;
    uint* strides_in = (uint*)md.strides_in;

    uint idx_in = 0;

    var walker = IndexWalker(md.sizes, NDIM, idx);
    for (uint i = 0; i < NDIM; i++) idx_in += walker.next(i) * strides_in[i];

    T best = tensor_in[idx_in];
    uint best_idx = 0;

    for (uint j = 1; j < reduce_size; j++)
    {
        if (T.ext_isnan(best)) break;

        T v = tensor_in[idx_in + j * reduce_stride];
        bool take;

        if (T.ext_isnan(v)) take = true;
        else take = (OP_TYPE == 0) ? T.ext_gt(v, best) : T.ext_lt(v, best);

        if (take) { best = v; best_idx = j; }
    }

    tensor_out[idx] = int64_t(best_idx);
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void arg_reduce_main<T : ITorchVulkan>(
    uint3 gi : SV_DispatchThreadID,
    uniform ArgReducePushConstants pcs)
{
    argReduceKernel<T>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.num_lines, pcs.reduce_stride, pcs.reduce_size);
}
