module embedding_backward;

__exported import core.dtype;
import core.indexing;

[[vk::constant_id(0)]] public const uint WG_x = 0;

public struct EmbeddingBackwardPushConstants
{
    public uint64_t grad;
    public uint64_t sorted_indices;
    public uint64_t permutation;
    public uint64_t grad_weight;
    public int64_t padding_idx;
    public uint64_t numel_out;
    public uint n;
    public uint d;
    public uint scale_by_frequency;
}

public void embeddingBackwardKernel<T : ITorchVulkan>(uint3 gi, EmbeddingBackwardPushConstants pcs)
{
    uint idx = flat_thread(gi, WG_x);
    if (idx >= pcs.numel_out) return;

    int64_t* sorted_indices = (int64_t*)pcs.sorted_indices;
    int64_t* permutation = (int64_t*)pcs.permutation;

    int64_t row = int64_t(idx / pcs.d);
    uint col = idx % pcs.d;

    T.Accum acc = T.ext_zero().toAccum();
    if (row != pcs.padding_idx)
    {
        uint lo = 0;
        uint hi = pcs.n;
        while (lo < hi)
        {
            uint mid = (lo + hi) / 2;
            if (sorted_indices[mid] < row) lo = mid + 1;
            else hi = mid;
        }

        uint i = lo;
        for (; i < pcs.n && sorted_indices[i] == row; i++)
        {
            uint64_t source = uint64_t(permutation[i]) * pcs.d + col;
            acc = T.Accum.ext_add(acc, T.load(pcs.grad, source).toAccum());
        }

        if (pcs.scale_by_frequency == 1 && i > lo) acc = T.Accum.ext_div(acc, T.Accum.castFromI64(int64_t(i - lo)));
    }

    T.store(pcs.grad_weight, idx, T.fromAccum(acc));
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void embedding_backward_main<T : ITorchVulkan>(
    uint3 gi : SV_DispatchThreadID,
    uniform EmbeddingBackwardPushConstants pcs)
{
    embeddingBackwardKernel<T>(gi, pcs);
}
