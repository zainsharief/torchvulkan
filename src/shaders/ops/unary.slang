module unary;

__exported import ops.unary_ops;
import core.indexing;

[[vk::constant_id(0)]] public const uint OP_TYPE = 0;
[[vk::constant_id(1)]] public const uint IS_CONTIG = 0;
[[vk::constant_id(2)]] public const uint NDIM = 0;
[[vk::constant_id(3)]] public const uint WG_x = 0;

public struct Metadata
{
    public uint64_t sizes;
    public uint64_t strides_in;
    public uint64_t strides_out;
}

public struct UnaryPushConstants
{
    public uint64_t tensor_in;
    public uint64_t tensor_out;
    public uint64_t metadata;
    public uint64_t numel;
}

public void baseUnaryKernel<T : ITorchVulkan, Op : IUnaryOp<T>, let VEC_SIZE : int>(
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

        T* tensor_in = (T*)address_in;
        T* tensor_out = (T*)address_out;

        T v[VEC_SIZE];
        [ForceUnroll] for (int k = 0; k < VEC_SIZE; k++) v[k] = tensor_in[idx * VEC_SIZE + k];

        T res[VEC_SIZE];
        [ForceUnroll] for (int k = 0; k < VEC_SIZE; k++) res[k] = Op.eval(v[k]);

        uint64_t remainder = numel - (idx * VEC_SIZE);
        uint64_t count = (remainder >= VEC_SIZE) ? VEC_SIZE : remainder;
        for (uint i = 0; i < count; i++) tensor_out[idx * VEC_SIZE + i] = res[i];

        return;
    }

    if (idx >= numel) return;

    T* tensor_in = (T*)address_in;
    T* tensor_out = (T*)address_out;

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

    tensor_out[idx_out] = Op.eval(tensor_in[idx_in]);
}

public void dispatchUnary<T : ITorchVulkan, let VEC_SIZE : int>(uint3 gi, UnaryPushConstants pcs)
{
    switch (OP_TYPE)
    {
    case 0:  baseUnaryKernel<T, ReluOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 1:  baseUnaryKernel<T, ExpOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 2:  baseUnaryKernel<T, LogOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 3:  baseUnaryKernel<T, SqrtOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 4:  baseUnaryKernel<T, NegOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 5:  baseUnaryKernel<T, ReciprocalOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 6:  baseUnaryKernel<T, SinOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 7:  baseUnaryKernel<T, CosOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 8:  baseUnaryKernel<T, TanOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 9:  baseUnaryKernel<T, AsinOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 10: baseUnaryKernel<T, AcosOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 11: baseUnaryKernel<T, AtanOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 12: baseUnaryKernel<T, SinhOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 13: baseUnaryKernel<T, CoshOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 14: baseUnaryKernel<T, TanhOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 15: baseUnaryKernel<T, Exp2Op<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 16: baseUnaryKernel<T, Log2Op<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 17: baseUnaryKernel<T, Log10Op<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 18: baseUnaryKernel<T, Expm1Op<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 19: baseUnaryKernel<T, Log1pOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 20: baseUnaryKernel<T, RsqrtOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 21: baseUnaryKernel<T, SigmoidOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 22: baseUnaryKernel<T, AsinhOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 23: baseUnaryKernel<T, AcoshOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 24: baseUnaryKernel<T, AtanhOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 25: baseUnaryKernel<T, Deg2radOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 26: baseUnaryKernel<T, Rad2degOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 27: baseUnaryKernel<T, FloorOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 28: baseUnaryKernel<T, CeilOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 29: baseUnaryKernel<T, TruncOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 30: baseUnaryKernel<T, RoundOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 31: baseUnaryKernel<T, FracOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 32: baseUnaryKernel<T, AbsOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 33: baseUnaryKernel<T, SignOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 34: baseUnaryKernel<T, ErfOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 35: baseUnaryKernel<T, ErfinvOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 36: baseUnaryKernel<T, SincOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 37: baseUnaryKernel<T, EntrOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 38: baseUnaryKernel<T, LgammaOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 39: baseUnaryKernel<T, DigammaOp<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 40: baseUnaryKernel<T, I0Op<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    case 41: baseUnaryKernel<T, I1Op<T>, VEC_SIZE>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel); break;
    default: break;
    }
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void unary_main<T : ITorchVulkan, let VEC_SIZE : int>(
    uint3 gi : SV_DispatchThreadID,
    uniform UnaryPushConstants pcs)
{
    dispatchUnary<T, VEC_SIZE>(gi, pcs);
}
