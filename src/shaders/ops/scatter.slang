module scatter;

__exported import core.dtype;
import core.indexing;

[[vk::constant_id(0)]] public const uint MODE = 0;
[[vk::constant_id(1)]] public const uint INCLUDE_SELF = 0;
[[vk::constant_id(2)]] public const uint INTEGRAL = 0;
[[vk::constant_id(3)]] public const uint WG_x = 0;

public struct ScatterPushConstants
{
    public uint64_t out;
    public uint64_t self;
    public uint64_t src;
    public uint64_t keys;
    public uint64_t order;
    public uint64_t count;
}

T combine<T : ITorchVulkan>(T acc, T v)
{
    switch (MODE)
    {
    case 1: return T.ext_add(acc, v);
    case 2: return T.ext_mul(acc, v);
    case 3: return T.ext_add(acc, v);
    case 4: return T.ext_max(acc, v);
    case 5: return T.ext_min(acc, v);
    default: return v;
    }
}

T identity<T : ITorchVulkan>()
{
    switch (MODE)
    {
    case 2: return T.ext_one();
    case 4: return T.ext_lowest();
    case 5: return T.ext_highest();
    default: return T.ext_zero();
    }
}

public void scatterKernel<T : ITorchVulkan>(uint3 gi, ScatterPushConstants pcs)
{
    uint64_t i = uint64_t(flat_thread(gi, WG_x));
    if (i >= pcs.count) return;

    int64_t* keys = (int64_t*)pcs.keys;
    int64_t* order = (int64_t*)pcs.order;

    int64_t key = keys[i];
    if (MODE == 0)
    {
        if (key >= 0) T.store(pcs.out, key, T.load(pcs.src, i));
        return;
    }

    if (i > 0 && keys[i - 1] == key) return;

    T acc = identity<T>();
    uint64_t n = 0;
    uint64_t j = i;
    if (INCLUDE_SELF != 0)
    {
        acc = T.load(pcs.self, key);
        n = 1;
    }
    else if (MODE != 6)
    {
        acc = T.load(pcs.src, order[j]);
        n = 1;
        j++;
    }
    for (; j < pcs.count && keys[j] == key; j++)
    {
        acc = (combine<T>(acc, T.load(pcs.src, order[j])));
        n++;
    }

    if (MODE == 3)
    {
        T divisor = castValue<uint64_t, T>(n);
        acc = ((INTEGRAL != 0) ? T.ext_floor_divide(acc, divisor) : T.ext_div(acc, divisor));
    }
    T.store(pcs.out, key, acc);
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void scatter_main<T : ITorchVulkan>(
    uint3 gi : SV_DispatchThreadID,
    uniform ScatterPushConstants pcs)
{
    scatterKernel<T>(gi, pcs);
}
