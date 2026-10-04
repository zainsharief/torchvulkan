module compare;

__exported import core.dtype;
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

public struct ComparePushConstants<T>
{
    public uint64_t tensor_a;
    public uint64_t tensor_b;
    public uint64_t tensor_out;
    public uint64_t metadata;
    public uint64_t numel;
    public T scalar_value;
}

public interface ICompareOp<T>
{
    static bool eval(T a, T b);
}

public struct EqOp<T : ITorchVulkan> : ICompareOp<T> { public static bool eval(T a, T b) { return T.ext_eq(a, b); } }
public struct NeOp<T : ITorchVulkan> : ICompareOp<T> { public static bool eval(T a, T b) { return T.ext_ne(a, b); } }
public struct LtOp<T : ITorchVulkan> : ICompareOp<T> { public static bool eval(T a, T b) { return T.ext_lt(a, b); } }
public struct LeOp<T : ITorchVulkan> : ICompareOp<T> { public static bool eval(T a, T b) { return T.ext_le(a, b); } }
public struct GtOp<T : ITorchVulkan> : ICompareOp<T> { public static bool eval(T a, T b) { return T.ext_gt(a, b); } }
public struct GeOp<T : ITorchVulkan> : ICompareOp<T> { public static bool eval(T a, T b) { return T.ext_ge(a, b); } }

public struct LogicalAndOp<T : ITorchVulkan> : ICompareOp<T> { public static bool eval(T a, T b) { return T.ext_nonzero(a) && T.ext_nonzero(b); } }
public struct LogicalOrOp<T : ITorchVulkan> : ICompareOp<T> { public static bool eval(T a, T b) { return T.ext_nonzero(a) || T.ext_nonzero(b); } }
public struct LogicalXorOp<T : ITorchVulkan> : ICompareOp<T> { public static bool eval(T a, T b) { return T.ext_nonzero(a) != T.ext_nonzero(b); } }

public void baseCompareKernel<T : ITorchVulkan, Op : ICompareOp<T>, let VEC_SIZE : int>(
    uint3 gi,
    uint64_t address_a,
    uint64_t address_b,
    uint64_t address_out,
    uint64_t address_metadata,
    uint64_t numel,
    T scalar_value)
{
    uint idx = gi.x;

    if (IS_CONTIG != 0)
    {
        if (idx * VEC_SIZE >= numel) return;

        T* tensor_a = (T*)address_a;

        T va[VEC_SIZE];
        [ForceUnroll] for (int k = 0; k < VEC_SIZE; k++) va[k] = tensor_a[idx * VEC_SIZE + k];

        T vb[VEC_SIZE];
        [ForceUnroll] for (int k = 0; k < VEC_SIZE; k++) vb[k] = scalar_value;

        // the second operand has no address when it is a scalar, so it is only read when it exists
        if (USE_SCALAR == 0)
        {
            T* tensor_b = (T*)address_b;
            [ForceUnroll] for (int k = 0; k < VEC_SIZE; k++) vb[k] = tensor_b[idx * VEC_SIZE + k];
        }

        uint8_t res[VEC_SIZE];
        [ForceUnroll] for (int k = 0; k < VEC_SIZE; k++) res[k] = Op.eval(va[k], vb[k]) ? uint8_t(1) : uint8_t(0);

        uint64_t remainder = numel - (idx * VEC_SIZE);
        uint64_t count = (remainder >= VEC_SIZE) ? VEC_SIZE : remainder;
        uint8_t* tensor_out = (uint8_t*)address_out;
        for (uint i = 0; i < count; i++) tensor_out[idx * VEC_SIZE + i] = res[i];

        return;
    }

    if (idx >= numel) return;

    T* tensor_a = (T*)address_a;
    uint8_t* tensor_out = (uint8_t*)address_out;

    Metadata md = *(Metadata*)address_metadata;
    uint* strides_a = (uint*)md.strides_a;
    uint* strides_b = (uint*)md.strides_b;
    uint* strides_out = (uint*)md.strides_out;

    uint idx_a = 0;
    uint idx_b = 0;
    uint idx_out = 0;

    var walker = IndexWalker(md.sizes, NDIM, idx);
    for (uint i = 0; i < NDIM; i++)
    {
        uint coord = walker.next(i);
        idx_a += coord * strides_a[i];
        idx_b += coord * strides_b[i];
        idx_out += coord * strides_out[i];
    }

    T va = tensor_a[idx_a];

    // the second operand has no address when it is a scalar, so it is only read when it exists
    T vb = scalar_value;
    if (USE_SCALAR == 0)
    {
        T* tensor_b = (T*)address_b;
        vb = tensor_b[idx_b];
    }

    tensor_out[idx_out] = Op.eval(va, vb) ? uint8_t(1) : uint8_t(0);
}

public void dispatchCompare<T : ITorchVulkan, let VEC_SIZE : int>(uint3 gi, ComparePushConstants<T> pcs)
{
    switch (OP_TYPE)
    {
    case 0: baseCompareKernel<T, EqOp<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.scalar_value); break;
    case 1: baseCompareKernel<T, NeOp<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.scalar_value); break;
    case 2: baseCompareKernel<T, LtOp<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.scalar_value); break;
    case 3: baseCompareKernel<T, LeOp<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.scalar_value); break;
    case 4: baseCompareKernel<T, GtOp<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.scalar_value); break;
    case 5: baseCompareKernel<T, GeOp<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.scalar_value); break;
    case 6: baseCompareKernel<T, LogicalAndOp<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.scalar_value); break;
    case 7: baseCompareKernel<T, LogicalOrOp<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.scalar_value); break;
    case 8: baseCompareKernel<T, LogicalXorOp<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.scalar_value); break;
    default: break;
    }
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void compare_main<T : ITorchVulkan, let VEC_SIZE : int>(
    uint3 gi : SV_DispatchThreadID,
    uniform ComparePushConstants<T> pcs)
{
    dispatchCompare<T, VEC_SIZE>(gi, pcs);
}
