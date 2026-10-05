module reduce_subgroup;

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

public struct SubgroupReducePushConstants
{
    public uint64_t tensor_in;
    public uint64_t tensor_out;
    public uint64_t metadata;
    public uint64_t numel_out;
    public uint reduce_stride;
    public uint reduce_size;
}

public interface IWaveReduceOp<T>
{
    static T identity();
    static T combine(T a, T b);
    static T wave_reduce(T a);
}

public struct SumWaveOp<T : __BuiltinFloatingPointType> : IWaveReduceOp<T>
{
    public static T identity() { return T.ext_zero(); }
    public static T combine(T a, T b) { return T.ext_add(a, b); }
    public static T wave_reduce(T a) { return WaveActiveSum(a); }
}

public struct AmaxWaveOp<T : __BuiltinFloatingPointType> : IWaveReduceOp<T>
{
    public static T identity() { return T.ext_lowest(); }
    public static T combine(T a, T b) { return T.ext_max(a, b); }
    public static T wave_reduce(T a) { return WaveActiveMax(a); }
}

public struct AminWaveOp<T : __BuiltinFloatingPointType> : IWaveReduceOp<T>
{
    public static T identity() { return T.ext_highest(); }
    public static T combine(T a, T b) { return T.ext_min(a, b); }
    public static T wave_reduce(T a) { return WaveActiveMin(a); }
}

public void baseSubgroupReduceKernel<T : __BuiltinFloatingPointType, Op : IWaveReduceOp<T>>(
    uint out_idx,
    uint lane,
    uint64_t address_in,
    uint64_t address_out,
    uint64_t address_metadata,
    uint64_t numel_out,
    uint reduce_stride,
    uint reduce_size)
{
    if (out_idx >= numel_out) return;

    T* tensor_in = (T*)address_in;
    T* tensor_out = (T*)address_out;

    Metadata md = *(Metadata*)address_metadata;
    uint* strides_in = (uint*)md.strides_in;

    uint idx_in = 0;

    var walker = IndexWalker(md.sizes, NDIM, out_idx);
    for (uint i = 0; i < NDIM; i++) idx_in += walker.next(i) * strides_in[i];

    T acc = Op.identity();
    for (uint r = lane; r < reduce_size; r += WG_x) acc = Op.combine(acc, tensor_in[idx_in + r * reduce_stride]);
    acc = Op.wave_reduce(acc);

    if (lane == 0) tensor_out[out_idx] = acc;
}

public void dispatchSubgroupReduce<T : __BuiltinFloatingPointType>(
    uint out_idx, uint lane, SubgroupReducePushConstants pcs)
{
    switch (OP_TYPE)
    {
    case 0: baseSubgroupReduceKernel<T, SumWaveOp<T>>(out_idx, lane, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel_out, pcs.reduce_stride, pcs.reduce_size); break;
    case 1: baseSubgroupReduceKernel<T, AmaxWaveOp<T>>(out_idx, lane, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel_out, pcs.reduce_stride, pcs.reduce_size); break;
    case 2: baseSubgroupReduceKernel<T, AminWaveOp<T>>(out_idx, lane, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel_out, pcs.reduce_stride, pcs.reduce_size); break;
    default: break;
    }
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void reduce_subgroup_main<T : __BuiltinFloatingPointType>(
    uint3 group_id : SV_GroupID,
    uint3 group_thread_id : SV_GroupThreadID,
    uniform SubgroupReducePushConstants pcs)
{
    dispatchSubgroupReduce<T>(group_id.x, group_thread_id.x, pcs);
}
