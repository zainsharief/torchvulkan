module attention;

__exported import core.dtype;

public struct AttentionPushConstants
{
    public uint64_t q;
    public uint64_t k;
    public uint64_t v;
    public uint64_t bias;
    public uint64_t out;
    public uint64_t lse;
    public uint q_batch;
    public uint q_head;
    public uint q_row;
    public uint k_batch;
    public uint k_head;
    public uint k_row;
    public uint v_batch;
    public uint v_head;
    public uint v_row;
    public uint out_batch;
    public uint out_head;
    public uint out_row;
    public uint bias_batch;
    public uint bias_head;
    public uint bias_row;
    public uint bias_col;
    public uint L;
    public uint S;
    public uint heads;
    public float scale;
}

public struct AttentionBackwardPushConstants
{
    public uint64_t q;
    public uint64_t k;
    public uint64_t v;
    public uint64_t bias;
    public uint64_t grad_out;
    public uint64_t lse;
    public uint64_t delta;
    public uint64_t grad_a;
    public uint64_t grad_b;
    public uint bias_batch;
    public uint bias_head;
    public uint bias_row;
    public uint bias_col;
    public uint L;
    public uint S;
    public uint heads;
    public float scale;
}

// the lowest finite value rather than -inf, which fast math may not keep
public A maskedScore<A : ITorchVulkan>() { return A.ext_lowest(); }

public A positiveInfinity<A : ITorchVulkan>() { return A.castFromF32(asfloat(0x7f800000u)); }

public T.Accum biasAt<T : ITorchVulkan>(uint64_t address, uint b, uint h, uint row, uint key, uint L, uint S, uint4 strides)
{
    T* bias = (T*)address;
    uint r = min(row, L - 1);
    uint c = min(key, S - 1);
    return bias[b * strides.x + h * strides.y + r * strides.z + c * strides.w].toAccum();
}

public T.Accum score<T : ITorchVulkan>(T.Accum dot, T.Accum scale, bool has_bias, T.Accum bias)
{
    typealias A = T.Accum;
    A s = dot * scale;
    if (has_bias) s = s + bias;
    return A.ext_max(s, maskedScore<A>());
}
