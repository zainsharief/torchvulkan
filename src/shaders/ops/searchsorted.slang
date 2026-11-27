module searchsorted;

__exported import core.dtype;
import core.indexing;

[[vk::constant_id(0)]] public const uint RIGHT = 0;
[[vk::constant_id(1)]] public const uint HAS_SORTER = 0;
[[vk::constant_id(2)]] public const uint OUT_INT32 = 0;
[[vk::constant_id(3)]] public const uint WG_x = 0;

public struct SearchsortedPushConstants
{
    public uint64_t sorted;
    public uint64_t sorter;
    public uint64_t values;
    public uint64_t out;
    public uint64_t count;
    public uint boundaries;
    public uint values_per_row;
    public uint shared_boundaries;
}

// torch's lower and upper bound, so NaNs land past the end
public void searchsortedKernel<T : ITorchVulkan>(uint3 gi, SearchsortedPushConstants pcs)
{
    uint64_t idx = uint64_t(flat_thread(gi, WG_x));
    if (idx >= pcs.count) return;

    int64_t* sorter = (int64_t*)pcs.sorter;

    uint64_t row = (pcs.shared_boundaries != 0) ? 0 : idx / uint64_t(pcs.values_per_row);
    uint64_t base = row * uint64_t(pcs.boundaries);
    T value = T.load(pcs.values, idx);

    uint start = 0;
    uint end = pcs.boundaries;
    while (start < end)
    {
        uint mid = start + ((end - start) >> 1);
        uint64_t position = base + uint64_t(mid);
        if (HAS_SORTER != 0) position = base + uint64_t(sorter[position]);
        T mid_value = T.load(pcs.sorted, position);
        bool before = (RIGHT != 0) ? !T.ext_gt(mid_value, value) : !T.ext_ge(mid_value, value);
        if (before) start = mid + 1;
        else end = mid;
    }

    if (OUT_INT32 != 0) ((int32_t*)pcs.out)[idx] = int32_t(start);
    else ((int64_t*)pcs.out)[idx] = int64_t(start);
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void searchsorted_main<T : ITorchVulkan>(
    uint3 gi : SV_DispatchThreadID,
    uniform SearchsortedPushConstants pcs)
{
    searchsortedKernel<T>(gi, pcs);
}
