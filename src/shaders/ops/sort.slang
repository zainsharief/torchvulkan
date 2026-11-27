module sort;

__exported import core.dtype;
import core.indexing;

[[vk::constant_id(0)]] public const uint MODE = 0;
[[vk::constant_id(1)]] public const uint DESCENDING = 0;
[[vk::constant_id(2)]] public const uint WG_x = 0;

public struct SortPushConstants
{
    public uint64_t keys_in;
    public uint64_t positions_in;
    public uint64_t keys_out;
    public uint64_t positions_out;
    public uint64_t input;
    public uint64_t threads;
    public uint n;
    public uint padded;
    public uint block;
    public uint distance;
}

bool sortsBefore<T : ITorchVulkan>(T x, int64_t px, T y, int64_t py, uint n)
{
    bool pad_x = px >= int64_t(n);
    bool pad_y = py >= int64_t(n);
    if (pad_x || pad_y) return pad_x == pad_y ? px < py : pad_y;

    bool nan_x = T.ext_isnan(x);
    bool nan_y = T.ext_isnan(y);
    if (nan_x || nan_y)
    {
        if (nan_x && nan_y) return px < py;
        return DESCENDING == 1 ? nan_x : nan_y;
    }

    if (DESCENDING == 1)
    {
        if (T.ext_gt(x, y)) return true;
        if (T.ext_gt(y, x)) return false;
    }
    else
    {
        if (T.ext_lt(x, y)) return true;
        if (T.ext_lt(y, x)) return false;
    }
    return px < py;
}

public void sortKernel<T : ITorchVulkan>(uint3 gi, SortPushConstants pcs)
{
    uint idx = flat_thread(gi, WG_x);
    if (uint64_t(idx) >= pcs.threads) return;

    int64_t* positions_out = (int64_t*)pcs.positions_out;

    if (MODE == 0)
    {
        uint line = idx / pcs.padded;
        uint p = idx % pcs.padded;
        if (p < pcs.n) T.store(pcs.keys_out, idx, T.load(pcs.input, uint64_t(line) * pcs.n + p));
        positions_out[idx] = int64_t(p);
        return;
    }

    int64_t* positions_in = (int64_t*)pcs.positions_in;

    uint half_padded = pcs.padded / 2;
    uint line = idx / half_padded;
    uint q = idx % half_padded;
    uint i = 2 * pcs.distance * (q / pcs.distance) + (q % pcs.distance);
    uint64_t a = uint64_t(line) * pcs.padded + i;
    uint64_t b = a + pcs.distance;

    T key_a = T.load(pcs.keys_in, a);
    T key_b = T.load(pcs.keys_in, b);
    int64_t pos_a = positions_in[a];
    int64_t pos_b = positions_in[b];

    // the first half of each block sorts forwards, the second backwards, which makes it bitonic
    bool forwards = (i & pcs.block) == 0;
    bool swap = forwards ? sortsBefore<T>(key_b, pos_b, key_a, pos_a, pcs.n) : sortsBefore<T>(key_a, pos_a, key_b, pos_b, pcs.n);

    T.store(pcs.keys_out, a, swap ? key_b : key_a);
    T.store(pcs.keys_out, b, swap ? key_a : key_b);
    positions_out[a] = swap ? pos_b : pos_a;
    positions_out[b] = swap ? pos_a : pos_b;
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void sort_main<T : ITorchVulkan>(
    uint3 gi : SV_DispatchThreadID,
    uniform SortPushConstants pcs)
{
    sortKernel<T>(gi, pcs);
}
