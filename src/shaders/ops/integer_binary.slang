module integer_binary;

import core.indexing;
import core.dtype;

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

public struct IntegerBinaryPushConstants<T : ITorchVulkan>
{
    public uint64_t tensor_a;
    public uint64_t tensor_b;
    public uint64_t tensor_out;
    public uint64_t metadata;
    public uint64_t numel;
    public T.PushScalar scalar_value;
}

bool isSigned<T : __BuiltinIntegerType>() { return T(-1) < T(0); }

T absInteger<T : __BuiltinIntegerType>(T a) { return (a < T(0)) ? -a : a; }

T leftShift<T : __BuiltinIntegerType>(T a, T b)
{
    T bits = T(sizeof(T) * 8);
    if (b < T(0) || b >= bits) return T(0);
    return a << b;
}

T rightShift<T : __BuiltinIntegerType>(T a, T b)
{
    T bits = T(sizeof(T) * 8);
    if (b < T(0) || b >= bits)
    {
        if (isSigned<T>()) return a >> (bits - T(1));
        return T(0);
    }
    return a >> b;
}

T gcd<T : __BuiltinIntegerType>(T a_in, T b_in)
{
    T a = absInteger(a_in);
    T b = absInteger(b_in);
    while (a != T(0))
    {
        T c = a;
        a = b % a;
        b = c;
    }
    return b;
}

T lcm<T : __BuiltinIntegerType>(T a, T b)
{
    T g = gcd(a, b);
    return (g == T(0)) ? T(0) : absInteger(a / g * b);
}

// must match IntegerBinaryOp in bitwise.cpp
T evalIntegerBinary<T : __BuiltinIntegerType>(T a, T b)
{
    switch (OP_TYPE)
    {
    case 0: return a & b;
    case 1: return a | b;
    case 2: return a ^ b;
    case 3: return leftShift(a, b);
    case 4: return rightShift(a, b);
    case 5: return gcd(a, b);
    case 6: return lcm(a, b);
    default: return a;
    }
}

// in T.Accum: an 8- or 16-bit integer without native support computes in int32 and wraps on the store, as C++ promotes
public void integerBinaryKernel<T : ITorchVulkan>(uint3 gi, IntegerBinaryPushConstants<T> pcs) where T.Accum : __BuiltinIntegerType
{
    uint idx = flat_thread(gi, WG_x);
    if (idx >= pcs.numel) return;


    uint idx_a = idx;
    uint idx_b = idx;
    uint idx_out = idx;

    if (IS_CONTIG == 0)
    {
        Metadata md = *(Metadata*)pcs.metadata;
        uint* strides_a = (uint*)md.strides_a;
        uint* strides_b = (uint*)md.strides_b;
        uint* strides_out = (uint*)md.strides_out;

        idx_a = 0;
        idx_b = 0;
        idx_out = 0;

        var walker = IndexWalker(md.sizes, NDIM, idx);
        for (uint i = 0; i < NDIM; i++)
        {
            uint coord = walker.next(i);
            idx_a += coord * strides_a[i];
            idx_b += coord * strides_b[i];
            idx_out += coord * strides_out[i];
        }
    }

    T b = T.fromPushScalar(pcs.scalar_value);
    if (USE_SCALAR == 0) b = T.load(pcs.tensor_b, idx_b);
    T.store(pcs.tensor_out, idx_out, T.fromAccum(evalIntegerBinary<T.Accum>(T.load(pcs.tensor_a, idx_a).toAccum(), b.toAccum())));
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void integer_binary_main<T : ITorchVulkan>(
    uint3 gi : SV_DispatchThreadID,
    uniform IntegerBinaryPushConstants<T> pcs) where T.Accum : __BuiltinIntegerType
{
    integerBinaryKernel<T>(gi, pcs);
}
