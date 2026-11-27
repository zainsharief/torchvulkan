module quantize_rows;

__exported import core.dtype;
import core.indexing;

[[vk::constant_id(0)]] public const uint WG_x = 256;

// torchvulkan::quantize_per_row: per row of a [rows, cols] input, scale = max(amax(|x|), 1e-10) / qmax and
// q = clamp(round_half_even(x * (qmax / max(amax(|x|), 1e-10))), -qmax, qmax) as int8: the float32 arithmetic of
// the elementwise torch expression in one pass over the row instead of six. Vulkan divides within 2.5 ulp, so a scale
// can differ from torch's in its last bit, and a code whose x * qmax / amax lands on a .5 boundary by one
struct QuantizeRowsPushConstants
{
    uint64_t x;
    uint64_t q;
    uint64_t scale;
    uint rows;
    uint cols;
    uint x_row_stride;
    float qmax;
}

static groupshared float smem[WG_x];


[shader("compute")]
[numthreads(WG_x, 1, 1)]
void quantize_rows_main<T : ITorchVulkan>(
    uint3 group_id : SV_GroupID,
    uint3 group_thread_id : SV_GroupThreadID,
    uniform QuantizeRowsPushConstants p)
{
    uint row = flat_group(group_id);
    uint lane = group_thread_id.x;
    // a whole workgroup takes one row, so it leaves together
    if (row >= p.rows) return;

    T* x = ((T*)p.x) + uint64_t(row) * p.x_row_stride;
    int8_t* q = ((int8_t*)p.q) + uint64_t(row) * p.cols;

    float m = 0.0f;
    for (uint j = lane; j < p.cols; j += WG_x) m = max(m, abs(castValue<T, float>(x[j])));
    smem[lane] = m;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = WG_x / 2; stride > 0; stride /= 2)
    {
        if (lane < stride) smem[lane] = max(smem[lane], smem[lane + stride]);
        GroupMemoryBarrierWithGroupSync();
    }
    float amax = max(smem[0], 1e-10f);
    // torch evaluates qmax / amax as reciprocal(amax) * qmax
    float inv = (1.0f / amax) * p.qmax;
    if (lane == 0) ((float*)p.scale)[row] = amax / p.qmax;

    for (uint j = lane; j < p.cols; j += WG_x)
    {
        float v = clamp(float.ext_round(castValue<T, float>(x[j]) * inv), -p.qmax, p.qmax);
        q[j] = int8_t(int(v));
    }
}
