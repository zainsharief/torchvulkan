module norm;

__exported import core.dtype;
import core.indexing;

[[vk::constant_id(0)]] public const uint MODE = 0;
[[vk::constant_id(1)]] public const uint WG_x = 0;
[[vk::constant_id(2)]] public const uint HAS_WEIGHT = 0;
[[vk::constant_id(3)]] public const uint HAS_BIAS = 0;
[[vk::constant_id(4)]] public const uint TRAINING = 0;
[[vk::constant_id(5)]] public const uint UPDATE_RUNNING = 0;
[[vk::constant_id(6)]] public const uint RMS = 0;
[[vk::constant_id(7)]] public const uint STATS_ACCUM = 0;
[[vk::constant_id(8)]] public const uint OUTPUTS = 0;

public static const int NORM_SMEM_SIZE = 256;
public static const uint PARAM_COLUMNS = 32;

public struct NormPushConstants
{
    public uint64_t x;
    public uint64_t dy;
    public uint64_t y;
    public uint64_t weight;
    public uint64_t bias;
    public uint64_t mean;
    public uint64_t rstd;
    public uint64_t running_mean;
    public uint64_t running_var;
    public uint64_t grad_weight;
    public uint64_t grad_bias;
    public uint outer;
    public uint channels;
    public uint inner;
    public float eps;
    public float momentum;
}

T.Accum loadStat<T : ITorchVulkan>(uint64_t address, uint i)
{
    if (STATS_ACCUM == 1) return ((T.Accum*)address)[i];
    return ((T*)address)[i].toAccum();
}

void storeStat<T : ITorchVulkan>(uint64_t address, uint i, T.Accum v)
{
    if (STATS_ACCUM == 1) ((T.Accum*)address)[i] = v;
    else ((T*)address)[i] = T.fromAccum(v);
}

T.Accum blockSum<T : ITorchVulkan>(T.Accum v, uint lane, __ref T.Accum smem[NORM_SMEM_SIZE])
{
    GroupMemoryBarrierWithGroupSync();
    smem[lane] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = WG_x / 2; stride > 0; stride /= 2)
    {
        if (lane < stride) smem[lane] = smem[lane] + smem[lane + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    return smem[0];
}

uint batchIndex(uint i, uint c, NormPushConstants pcs)
{
    uint count = pcs.outer * pcs.inner;
    i = min(i, count - 1);
    uint n = i / pcs.inner;
    return (n * pcs.channels + c) * pcs.inner + (i - n * pcs.inner);
}

void batchForward<T : ITorchVulkan>(uint c, uint lane, __ref T.Accum smem[NORM_SMEM_SIZE], NormPushConstants pcs)
{
    typealias A = T.Accum;
    if (c >= pcs.channels) return;

    T* x = (T*)pcs.x;
    T* y = (T*)pcs.y;
    uint count = pcs.outer * pcs.inner;
    A eps = A.castFromF32(pcs.eps);

    A mean, invstd;
    if (TRAINING == 1)
    {
        A n = A.castFromU64(uint64_t(count));
        A sum = A.ext_zero();
        for (uint i = lane; i < count; i += WG_x) sum = sum + x[batchIndex(i, c, pcs)].toAccum();
        mean = blockSum<T>(sum, lane, smem) / n;

        A sq = A.ext_zero();
        for (uint i = lane; i < count; i += WG_x)
        {
            A d = x[batchIndex(i, c, pcs)].toAccum() - mean;
            sq = sq + d * d;
        }
        A var = blockSum<T>(sq, lane, smem) / n;
        invstd = A.ext_rsqrt(var + eps);

        if (lane == 0)
        {
            storeStat<T>(pcs.mean, c, mean);
            storeStat<T>(pcs.rstd, c, invstd);
            if (UPDATE_RUNNING == 1)
            {
                T* running_mean = (T*)pcs.running_mean;
                T* running_var = (T*)pcs.running_var;
                A m = A.castFromF32(pcs.momentum);
                A keep = A.ext_one() - m;
                A unbiased = var * n / (n - A.ext_one());
                running_mean[c] = T.fromAccum(keep * running_mean[c].toAccum() + m * mean);
                running_var[c] = T.fromAccum(keep * running_var[c].toAccum() + m * unbiased);
            }
        }
    }
    else
    {
        mean = ((T*)pcs.running_mean)[c].toAccum();
        invstd = A.ext_rsqrt(((T*)pcs.running_var)[c].toAccum() + eps);
    }

    A scale = invstd;
    if (HAS_WEIGHT == 1) scale = scale * ((T*)pcs.weight)[c].toAccum();
    A shift = A.ext_zero();
    if (HAS_BIAS == 1) shift = ((T*)pcs.bias)[c].toAccum();

    for (uint i = lane; i < count; i += WG_x)
    {
        uint idx = batchIndex(i, c, pcs);
        y[idx] = T.fromAccum((x[idx].toAccum() - mean) * scale + shift);
    }
}

void batchBackward<T : ITorchVulkan>(uint c, uint lane, __ref T.Accum smem[NORM_SMEM_SIZE], NormPushConstants pcs)
{
    typealias A = T.Accum;
    if (c >= pcs.channels) return;

    T* x = (T*)pcs.x;
    T* dy = (T*)pcs.dy;
    uint count = pcs.outer * pcs.inner;
    A n = A.castFromU64(uint64_t(count));

    A mean, invstd;
    if (TRAINING == 1)
    {
        mean = loadStat<T>(pcs.mean, c);
        invstd = loadStat<T>(pcs.rstd, c);
    }
    else
    {
        mean = ((T*)pcs.running_mean)[c].toAccum();
        invstd = A.ext_rsqrt(((T*)pcs.running_var)[c].toAccum() + A.castFromF32(pcs.eps));
    }

    A sum_dy = A.ext_zero();
    A sum_dy_xmu = A.ext_zero();
    if (TRAINING == 1 || (OUTPUTS & 6) != 0)
    {
        for (uint i = lane; i < count; i += WG_x)
        {
            uint idx = batchIndex(i, c, pcs);
            A g = dy[idx].toAccum();
            sum_dy = sum_dy + g;
            sum_dy_xmu = sum_dy_xmu + g * (x[idx].toAccum() - mean);
        }
        sum_dy = blockSum<T>(sum_dy, lane, smem);
        sum_dy_xmu = blockSum<T>(sum_dy_xmu, lane, smem);
    }

    if (lane == 0)
    {
        if ((OUTPUTS & 2) != 0) ((T*)pcs.grad_weight)[c] = T.fromAccum(sum_dy_xmu * invstd);
        if ((OUTPUTS & 4) != 0) ((T*)pcs.grad_bias)[c] = T.fromAccum(sum_dy);
    }

    if ((OUTPUTS & 1) == 0) return;

    T* grad_input = (T*)pcs.y;
    A scale = invstd;
    if (HAS_WEIGHT == 1) scale = scale * ((T*)pcs.weight)[c].toAccum();
    A mean_dy = sum_dy / n;
    A proj = sum_dy_xmu / n * invstd * invstd;

    for (uint i = lane; i < count; i += WG_x)
    {
        uint idx = batchIndex(i, c, pcs);
        A g = dy[idx].toAccum();
        if (TRAINING == 1) g = g - mean_dy - (x[idx].toAccum() - mean) * proj;
        grad_input[idx] = T.fromAccum(g * scale);
    }
}

void rowForward<T : ITorchVulkan>(uint row, uint lane, __ref T.Accum smem[NORM_SMEM_SIZE], NormPushConstants pcs)
{
    typealias A = T.Accum;
    if (row >= pcs.outer) return;

    uint size = pcs.channels;
    T* x = ((T*)pcs.x) + uint64_t(row) * size;
    T* y = ((T*)pcs.y) + uint64_t(row) * size;
    A n = A.castFromU64(uint64_t(size));

    A mean = A.ext_zero();
    if (RMS == 0)
    {
        A sum = A.ext_zero();
        for (uint j = lane; j < size; j += WG_x) sum = sum + x[min(j, size - 1)].toAccum();
        mean = blockSum<T>(sum, lane, smem) / n;
    }

    A sq = A.ext_zero();
    for (uint j = lane; j < size; j += WG_x)
    {
        A d = x[min(j, size - 1)].toAccum() - mean;
        sq = sq + d * d;
    }
    A rstd = A.ext_rsqrt(blockSum<T>(sq, lane, smem) / n + A.castFromF32(pcs.eps));

    if (lane == 0)
    {
        if (RMS == 0) storeStat<T>(pcs.mean, row, mean);
        storeStat<T>(pcs.rstd, row, rstd);
    }

    for (uint j = lane; j < size; j += WG_x)
    {
        uint k = min(j, size - 1);
        A v = (x[k].toAccum() - mean) * rstd;
        if (HAS_WEIGHT == 1) v = v * ((T*)pcs.weight)[k].toAccum();
        if (HAS_BIAS == 1) v = v + ((T*)pcs.bias)[k].toAccum();
        y[k] = T.fromAccum(v);
    }
}

void rowBackwardInput<T : ITorchVulkan>(uint row, uint lane, __ref T.Accum smem[NORM_SMEM_SIZE], NormPushConstants pcs)
{
    typealias A = T.Accum;
    if (row >= pcs.outer) return;

    uint size = pcs.channels;
    T* x = ((T*)pcs.x) + uint64_t(row) * size;
    T* dy = ((T*)pcs.dy) + uint64_t(row) * size;
    T* grad_input = ((T*)pcs.y) + uint64_t(row) * size;
    A n = A.castFromU64(uint64_t(size));

    A mean = A.ext_zero();
    if (RMS == 0) mean = loadStat<T>(pcs.mean, row);
    A rstd = loadStat<T>(pcs.rstd, row);

    A sum_g = A.ext_zero();
    A sum_g_xhat = A.ext_zero();
    for (uint j = lane; j < size; j += WG_x)
    {
        uint k = min(j, size - 1);
        A g = dy[k].toAccum();
        if (HAS_WEIGHT == 1) g = g * ((T*)pcs.weight)[k].toAccum();
        sum_g = sum_g + g;
        sum_g_xhat = sum_g_xhat + g * (x[k].toAccum() - mean) * rstd;
    }
    A mean_g = A.ext_zero();
    if (RMS == 0) mean_g = blockSum<T>(sum_g, lane, smem) / n;
    A mean_g_xhat = blockSum<T>(sum_g_xhat, lane, smem) / n;

    for (uint j = lane; j < size; j += WG_x)
    {
        uint k = min(j, size - 1);
        A g = dy[k].toAccum();
        if (HAS_WEIGHT == 1) g = g * ((T*)pcs.weight)[k].toAccum();
        A xhat = (x[k].toAccum() - mean) * rstd;
        grad_input[k] = T.fromAccum(rstd * (g - mean_g - xhat * mean_g_xhat));
    }
}

void rowBackwardParams<T : ITorchVulkan>(uint group, uint lane, __ref T.Accum smem[NORM_SMEM_SIZE], NormPushConstants pcs)
{
    typealias A = T.Accum;
    uint size = pcs.channels;
    uint rows = pcs.outer;
    uint col = group * PARAM_COLUMNS + lane % PARAM_COLUMNS;
    uint k = min(col, size - 1);
    uint row_lane = lane / PARAM_COLUMNS;
    uint row_lanes = WG_x / PARAM_COLUMNS;

    T* x = (T*)pcs.x;
    T* dy = (T*)pcs.dy;

    A sum_dy = A.ext_zero();
    A sum_dy_xhat = A.ext_zero();
    for (uint r = row_lane; r < rows; r += row_lanes)
    {
        uint rr = min(r, rows - 1);
        A mean = A.ext_zero();
        if (RMS == 0) mean = loadStat<T>(pcs.mean, rr);
        A rstd = loadStat<T>(pcs.rstd, rr);
        uint idx = rr * size + k;
        A g = dy[idx].toAccum();
        sum_dy = sum_dy + g;
        sum_dy_xhat = sum_dy_xhat + g * (x[idx].toAccum() - mean) * rstd;
    }

    for (uint pass = 0; pass < 2; pass++)
    {
        GroupMemoryBarrierWithGroupSync();
        smem[lane] = pass == 0 ? sum_dy_xhat : sum_dy;
        GroupMemoryBarrierWithGroupSync();
        for (uint stride = row_lanes / 2; stride > 0; stride /= 2)
        {
            if (row_lane < stride) smem[lane] = smem[lane] + smem[lane + stride * PARAM_COLUMNS];
            GroupMemoryBarrierWithGroupSync();
        }
        if (row_lane == 0 && col < size)
        {
            if (pass == 0 && (OUTPUTS & 2) != 0) ((T*)pcs.grad_weight)[col] = T.fromAccum(smem[lane]);
            if (pass == 1 && (OUTPUTS & 4) != 0) ((T*)pcs.grad_bias)[col] = T.fromAccum(smem[lane]);
        }
    }
}

public void dispatchNorm<T : ITorchVulkan>(uint group, uint lane, __ref T.Accum smem[NORM_SMEM_SIZE], NormPushConstants pcs)
{
    switch (MODE)
    {
    case 0: batchForward<T>(group, lane, smem, pcs); break;
    case 1: batchBackward<T>(group, lane, smem, pcs); break;
    case 2: rowForward<T>(group, lane, smem, pcs); break;
    case 3: rowBackwardInput<T>(group, lane, smem, pcs); break;
    case 4: rowBackwardParams<T>(group, lane, smem, pcs); break;
    default: break;
    }
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void norm_main<T : ITorchVulkan>(
    uint3 group_id : SV_GroupID,
    uint3 group_thread_id : SV_GroupThreadID,
    uniform NormPushConstants pcs)
{
    static groupshared T.Accum smem[NORM_SMEM_SIZE];
    dispatchNorm<T>(flat_group(group_id), group_thread_id.x, smem, pcs);
}
