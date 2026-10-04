module nllloss;

__exported import core.dtype;

[[vk::constant_id(0)]] public const uint MODE = 0;
[[vk::constant_id(1)]] public const uint WG_x = 0;

public struct NllLossPushConstants
{
    public uint64_t tensor_in;
    public uint64_t target;
    public uint64_t tensor_out;
    public uint N;
    public uint C;
    public int ignore_index;
}

public interface INllLossMode
{
    static void apply<T : ITorchVulkan>(
        uint tid, T* tensor_in, int64_t* target, T* tensor_out, uint N, uint C, int ignore_index);
}

public struct ForwardGatherOp : INllLossMode
{
    public static void apply<T : ITorchVulkan>(
        uint tid, T* tensor_in, int64_t* target, T* tensor_out, uint N, uint C, int ignore_index)
    {
        if (tid >= N) return;

        int t = int(target[tid]);
        if (t == ignore_index) { tensor_out[tid] = T.ext_zero(); return; }
        tensor_out[tid] = T.ext_neg(tensor_in[tid * C + uint(t)]);
    }
}

public struct BackwardScatterOp : INllLossMode
{
    public static void apply<T : ITorchVulkan>(
        uint tid, T* tensor_in, int64_t* target, T* tensor_out, uint N, uint C, int ignore_index)
    {
        if (tid >= N * C) return;

        uint i = tid / C;
        uint c = tid % C;
        int t = int(target[i]);
        tensor_out[tid] = (t != ignore_index && int(c) == t) ? T.ext_neg(tensor_in[i]) : T.ext_zero();
    }
}

public struct WeightGatherOp : INllLossMode
{
    public static void apply<T : ITorchVulkan>(
        uint tid, T* tensor_in, int64_t* target, T* tensor_out, uint N, uint C, int ignore_index)
    {
        if (tid >= N) return;

        int t = int(target[tid]);
        if (t == ignore_index) { tensor_out[tid] = T.ext_zero(); return; }
        tensor_out[tid] = tensor_in[uint(t)];
    }
}

public void dispatchNllLoss<T : ITorchVulkan>(uint3 gi, NllLossPushConstants pcs)
{
    T* tensor_in = (T*)pcs.tensor_in;
    int64_t* target = (int64_t*)pcs.target;
    T* tensor_out = (T*)pcs.tensor_out;

    switch (MODE)
    {
    case 0: ForwardGatherOp.apply<T>(gi.x, tensor_in, target, tensor_out, pcs.N, pcs.C, pcs.ignore_index); break;
    case 1: BackwardScatterOp.apply<T>(gi.x, tensor_in, target, tensor_out, pcs.N, pcs.C, pcs.ignore_index); break;
    case 2: WeightGatherOp.apply<T>(gi.x, tensor_in, target, tensor_out, pcs.N, pcs.C, pcs.ignore_index); break;
    default: break;
    }
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void nllloss_main<T : ITorchVulkan>(
    uint3 gi : SV_DispatchThreadID,
    uniform NllLossPushConstants pcs)
{
    dispatchNllLoss<T>(gi, pcs);
}
