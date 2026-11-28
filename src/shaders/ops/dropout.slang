module dropout;

__exported import core.philox;
import core.indexing;

[[vk::constant_id(0)]] public const uint WG_x = 0;

public struct DropoutPushConstants<A : ITorchVulkan>
{
    public uint64_t input;
    public uint64_t output;
    public uint64_t mask;
    public uint64_t numel;
    public uint64_t seed;
    public uint64_t offset;
    public uint64_t keep_prob; // a T.Real's bits
    public A.PushScalar scale; // scaled in T.Accum
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
// the mask is a bool tensor, stored as B
void dropout_main<T : IRandomDtype, B : ITorchVulkan, let VEC_SIZE : int>(
    uint3 gi : SV_DispatchThreadID,
    uniform DropoutPushConstants<T.Accum> pcs)
{
    typealias R = T.Real;
    typealias A = T.Accum;

    uint stream_id = flat_thread(gi, WG_x);
    uint first = stream_id * VEC_SIZE;
    if (first >= pcs.numel) return;

    var stream = RandomStream<R>(pcs.seed, pcs.offset, stream_id);
    R keepProb = R.fromBits(pcs.keep_prob);
    A scale = A.fromPushScalar(pcs.scale);

    for (int k = 0; k < VEC_SIZE; k++)
    {
        uint element = first + k;
        if (element >= pcs.numel) return;

        bool kept = stream.uniform() < keepProb;
        // multiplying by zero rather than writing it keeps torch's nan for a dropped inf
        A value = A.ext_mul(T.load(pcs.input, element).toAccum(), kept ? scale : A.ext_zero());
        T.store(pcs.output, element, T.fromAccum(value));
        B.store(pcs.mask, element, B.castFromI64(kept ? 1 : 0));
    }
}
