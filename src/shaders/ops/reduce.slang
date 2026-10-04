module reduce;

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

public struct ReducePushConstants
{
    public uint64_t tensor_in;
    public uint64_t tensor_out;
    public uint64_t metadata;
    public uint64_t numel_out;
    public uint reduce_stride;
    public uint reduce_size;
}

public interface IReduceOp<T>
{
    static T combine(T a, T b);
}

public struct SumReduceOp<T : ITorchVulkan> : IReduceOp<T> { public static T combine(T a, T b) { return T.ext_add(a, b); } }
public struct AmaxReduceOp<T : ITorchVulkan> : IReduceOp<T> { public static T combine(T a, T b) { return T.ext_max(a, b); } }
public struct AminReduceOp<T : ITorchVulkan> : IReduceOp<T> { public static T combine(T a, T b) { return T.ext_min(a, b); } }
public struct ProdReduceOp<T : ITorchVulkan> : IReduceOp<T> { public static T combine(T a, T b) { return T.ext_mul(a, b); } }

public void baseReduceKernel<T : ITorchVulkan, Op : IReduceOp<T>>(
    uint3 gi,
    uint64_t address_in,
    uint64_t address_out,
    uint64_t address_metadata,
    uint64_t numel_out,
    uint reduce_stride,
    uint reduce_size)
{
    uint idx = gi.x;
    if (idx >= numel_out) return;

    T* tensor_in = (T*)address_in;
    T* tensor_out = (T*)address_out;

    Metadata md = *(Metadata*)address_metadata;
    uint* strides_in = (uint*)md.strides_in;

    uint idx_in = 0;

    var walker = IndexWalker(md.sizes, NDIM, idx);
    for (uint i = 0; i < NDIM; i++) idx_in += walker.next(i) * strides_in[i];

    T acc = tensor_in[idx_in];
    for (uint r = 1; r < reduce_size; r++) acc = Op.combine(acc, tensor_in[idx_in + r * reduce_stride]);

    tensor_out[idx] = acc;
}

public void dispatchReduce<T : ITorchVulkan>(uint3 gi, ReducePushConstants pcs)
{
    switch (OP_TYPE)
    {
    case 0: baseReduceKernel<T, SumReduceOp<T>>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel_out, pcs.reduce_stride, pcs.reduce_size); break;
    case 1: baseReduceKernel<T, AmaxReduceOp<T>>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel_out, pcs.reduce_stride, pcs.reduce_size); break;
    case 2: baseReduceKernel<T, AminReduceOp<T>>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel_out, pcs.reduce_stride, pcs.reduce_size); break;
    case 3: baseReduceKernel<T, ProdReduceOp<T>>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel_out, pcs.reduce_stride, pcs.reduce_size); break;
    default: break;
    }
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void reduce_main<T : ITorchVulkan>(
    uint3 gi : SV_DispatchThreadID,
    uniform ReducePushConstants pcs)
{
    dispatchReduce<T>(gi, pcs);
}
