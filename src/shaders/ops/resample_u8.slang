module resample_u8;

__exported import core.dtype;
import core.indexing;

[[vk::constant_id(0)]] public const uint WG_x = 0;

public struct ResampleU8PushConstants
{
    public uint64_t input;
    public uint64_t output;
    public uint64_t index_min;
    public uint64_t index_size;
    public uint64_t weights;
    public uint64_t count;
    public uint in_length;
    public uint out_length;
    public uint inner;
    public uint taps;
    public uint precision;
}

// as ATen's uint8 separable loops: a rounding bias, the integer products, then shift and clamp
// T holds the uint8 image, W the int16 weights
void resampleU8Kernel<T : ITorchVulkan, W : ITorchVulkan>(uint64_t idx, ResampleU8PushConstants pcs)
{
    if (idx >= pcs.count) return;
    uint64_t inner = uint64_t(pcs.inner);
    uint64_t outer = idx / (uint64_t(pcs.out_length) * inner);
    uint o = uint((idx / inner) % uint64_t(pcs.out_length));
    uint64_t k = idx % inner;

    int64_t first = ((int64_t*)pcs.index_min)[o];
    int64_t size = ((int64_t*)pcs.index_size)[o];
    uint64_t weights = uint64_t(o) * pcs.taps;
    uint64_t base = outer * uint64_t(pcs.in_length) * inner + k;

    int acc = 1 << (pcs.precision - 1);
    for (int64_t j = 0; j < size; j++)
    {
        int value = castValue<T, int32_t>(T.load(pcs.input, base + uint64_t(first + j) * inner));
        acc += value * castValue<W, int32_t>(W.load(pcs.weights, weights + uint64_t(j)));
    }
    T.store(pcs.output, idx, castValue<int32_t, T>(clamp(acc >> int(pcs.precision), 0, 255)));
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void resample_u8_main<T : ITorchVulkan, W : ITorchVulkan>(
    uint3 gi : SV_DispatchThreadID,
    uniform ResampleU8PushConstants pcs)
{
    resampleU8Kernel<T, W>(uint64_t(flat_thread(gi, WG_x)), pcs);
}
