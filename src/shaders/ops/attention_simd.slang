module attention_simd;

__exported import core.dtype;
import helpers.attention;

[[vk::constant_id(0)]] public const uint MODE = 0;
[[vk::constant_id(1)]] public const uint WG = 128;
[[vk::constant_id(2)]] public const uint BR = 16;
[[vk::constant_id(3)]] public const uint BC = 32;
[[vk::constant_id(4)]] public const uint D = 64;
[[vk::constant_id(5)]] public const uint DV = 64;
[[vk::constant_id(6)]] public const uint GROUP = 1;
[[vk::constant_id(7)]] public const uint HAS_BIAS = 0;
[[vk::constant_id(8)]] public const uint CAUSAL = 0;
[[vk::constant_id(9)]] public const uint SMEM_SIZE = 1;
[[vk::constant_id(10)]] public const uint ACC_A = 1;
[[vk::constant_id(11)]] public const uint ACC_B = 1;
[[vk::constant_id(12)]] public const uint SPLITS = 1;
[[vk::constant_id(13)]] public const uint SPLIT_KEYS = 0;

public struct Shared<A>
{
    public static groupshared A smem[SMEM_SIZE];
}

// padded by one element so consecutive threads reading a column hit different banks
uint kvWidth() { return D > DV ? D : DV; }

bool isMasked(uint row, uint key, uint L, uint S)
{
    return key >= S || row >= L || (CAUSAL == 1 && key > row);
}

void attentionForward<T : ITorchVulkan>(uint3 wg, uint t, AttentionPushConstants pcs)
{
    typealias A = T.Accum;
    T* q = (T*)pcs.q;
    T* k = (T*)pcs.k;
    T* v = (T*)pcs.v;

    uint row0 = wg.x * BR;
    uint h = wg.y;
    uint z = wg.z;
    uint b = z / SPLITS;
    uint kvh = h / GROUP;
    uint q_base = b * pcs.q_batch + h * pcs.q_head;
    uint k_base = b * pcs.k_batch + kvh * pcs.k_head;
    uint v_base = b * pcs.v_batch + kvh * pcs.v_head;
    uint4 bias_strides = uint4(pcs.bias_batch, pcs.bias_head, pcs.bias_row, pcs.bias_col);

    // shared: q[BR][D + 1], kv[BC][max(D, DV) + 1], p[BR][BC + 1], stat[BR]
    const uint KVW = kvWidth() + 1;
    uint q_off = 0;
    uint kv_off = BR * (D + 1);
    uint p_off = kv_off + BC * KVW;
    uint stat_off = p_off + BR * (BC + 1);

    for (uint i = t; i < BR * D; i += WG)
    {
        uint r = i / D;
        uint d = i % D;
        uint row = min(row0 + r, pcs.L - 1);
        Shared<A>.smem[q_off + r * (D + 1) + d] = q[q_base + row * pcs.q_row + d].toAccum();
    }

    A o[ACC_A];
    [unroll] for (uint j = 0; j < ACC_A; j++) o[j] = A.ext_zero();

    A m = maskedScore<A>();
    A l = A.ext_zero();
    A scale = A.castFromF32(pcs.scale);
    A lowest = maskedScore<A>();

    uint sc = t % BC;
    uint row_step = WG / BC;

    uint kv_begin = 0;
    uint kv_end = pcs.S;
    if (SPLITS > 1)
    {
        kv_begin = (z % SPLITS) * SPLIT_KEYS;
        kv_end = min(pcs.S, kv_begin + SPLIT_KEYS);
    }
    if (CAUSAL == 1) kv_end = min(kv_end, row0 + BR);

    for (uint c0 = kv_begin; c0 < kv_end; c0 += BC)
    {
        GroupMemoryBarrierWithGroupSync();
        for (uint i = t; i < BC * D; i += WG)
        {
            uint c = i / D;
            uint d = i % D;
            uint key = min(c0 + c, pcs.S - 1);
            Shared<A>.smem[kv_off + c * KVW + d] = k[k_base + key * pcs.k_row + d].toAccum();
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint r = t / BC; r < BR; r += row_step)
        {
            A dot = A.ext_zero();
            for (uint d = 0; d < D; d++) dot = dot + Shared<A>.smem[q_off + r * (D + 1) + d] * Shared<A>.smem[kv_off + sc * KVW + d];
            uint row = row0 + r;
            uint key = c0 + sc;
            A bias = A.ext_zero();
            if (HAS_BIAS == 1) bias = biasAt<T>(pcs.bias, b, h, row, key, pcs.L, pcs.S, bias_strides);
            A s = score<T>(dot, scale, HAS_BIAS == 1, bias);
            Shared<A>.smem[p_off + r * (BC + 1) + sc] = isMasked(row, key, pcs.L, pcs.S) ? lowest : s;
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint i = t; i < BC * DV; i += WG)
        {
            uint c = i / DV;
            uint d = i % DV;
            uint key = min(c0 + c, pcs.S - 1);
            A value = v[v_base + key * pcs.v_row + d].toAccum();
            Shared<A>.smem[kv_off + c * KVW + d] = c0 + c < pcs.S ? value : A.ext_zero();
        }

        if (t < BR)
        {
            uint pr = p_off + t * (BC + 1);
            A block_max = lowest;
            for (uint c = 0; c < BC; c++) block_max = A.ext_max(block_max, Shared<A>.smem[pr + c]);
            A m_new = A.ext_max(m, block_max);
            bool empty = A.ext_eq(m_new, lowest);
            A alpha = empty ? A.ext_one() : A.ext_exp(m - m_new);
            A sum = A.ext_zero();
            for (uint c = 0; c < BC; c++)
            {
                A s = Shared<A>.smem[pr + c];
                A p = (empty || A.ext_eq(s, lowest)) ? A.ext_zero() : A.ext_exp(s - m_new);
                Shared<A>.smem[pr + c] = p;
                sum = sum + p;
            }
            l = l * alpha + sum;
            m = m_new;
            Shared<A>.smem[stat_off + t] = alpha;
        }
        GroupMemoryBarrierWithGroupSync();

        [unroll] for (uint j = 0; j < ACC_A; j++)
        {
            uint e = min(t + j * WG, BR * DV - 1);
            uint r = e / DV;
            uint d = e % DV;
            A acc = o[j] * Shared<A>.smem[stat_off + r];
            for (uint c = 0; c < BC; c++) acc = acc + Shared<A>.smem[p_off + r * (BC + 1) + c] * Shared<A>.smem[kv_off + c * KVW + d];
            o[j] = acc;
        }
    }

    GroupMemoryBarrierWithGroupSync();
    if (t < BR)
    {
        Shared<A>.smem[stat_off + t] = l;
        uint row = row0 + t;
        A zero = A.ext_zero();
        // a row that attends to nothing gets a logsumexp of +inf
        A lse = A.ext_gt(l, zero) ? m + A.ext_log(l) : positiveInfinity<A>();
        if (row < pcs.L) ((A*)pcs.lse)[(z * pcs.heads + h) * pcs.L + row] = lse;
    }
    GroupMemoryBarrierWithGroupSync();

    uint out_base = z * pcs.out_batch + h * pcs.out_head;
    [unroll] for (uint j = 0; j < ACC_A; j++)
    {
        uint e = t + j * WG;
        uint r = min(e, BR * DV - 1) / DV;
        uint d = e % DV;
        uint row = row0 + r;
        A sum = Shared<A>.smem[stat_off + r];
        A value = A.ext_gt(sum, A.ext_zero()) ? o[j] / sum : A.ext_zero();
        if (e < BR * DV && row < pcs.L)
        {
            uint i = out_base + row * pcs.out_row + d;
            if (SPLITS > 1) ((A*)pcs.out)[i] = value;
            else ((T*)pcs.out)[i] = T.fromAccum(value);
        }
    }
}

void loadRows<T : ITorchVulkan>(uint t, T* src, uint base, uint r0, uint rows, uint width, uint off, uint count)
{
    typealias A = T.Accum;
    for (uint i = t; i < count * width; i += WG)
    {
        uint r = i / width;
        uint d = i % width;
        uint row = min(r0 + r, rows - 1);
        A value = src[base + row * width + d].toAccum();
        Shared<A>.smem[off + r * (width + 1) + d] = r0 + r < rows ? value : A.ext_zero();
    }
}

void probabilityGradients<T : ITorchVulkan>(
    uint t, uint b, uint h, uint r0, uint c0, AttentionBackwardPushConstants pcs,
    uint q_off, uint do_off, uint k_off, uint v_off, uint lse_off, uint delta_off, uint p_off, uint ds_off, bool keep_p)
{
    typealias A = T.Accum;
    uint sc = t % BC;
    A scale = A.castFromF32(pcs.scale);
    A lowest = maskedScore<A>();
    uint4 bias_strides = uint4(pcs.bias_batch, pcs.bias_head, pcs.bias_row, pcs.bias_col);

    for (uint r = t / BC; r < BR; r += WG / BC)
    {
        A dot = A.ext_zero();
        for (uint d = 0; d < D; d++) dot = dot + Shared<A>.smem[q_off + r * (D + 1) + d] * Shared<A>.smem[k_off + sc * (D + 1) + d];
        A dp = A.ext_zero();
        for (uint d = 0; d < DV; d++) dp = dp + Shared<A>.smem[do_off + r * (DV + 1) + d] * Shared<A>.smem[v_off + sc * (DV + 1) + d];

        uint row = r0 + r;
        uint key = c0 + sc;
        A bias = A.ext_zero();
        if (HAS_BIAS == 1) bias = biasAt<T>(pcs.bias, b, h, row, key, pcs.L, pcs.S, bias_strides);
        A s = score<T>(dot, scale, HAS_BIAS == 1, bias);
        A lse = Shared<A>.smem[lse_off + r];
        // a row that attends to nothing has every score masked
        bool masked = isMasked(row, key, pcs.L, pcs.S) || A.ext_eq(s, lowest) || A.ext_eq(lse, positiveInfinity<A>());
        A p = masked ? A.ext_zero() : A.ext_exp(s - lse);
        if (keep_p) Shared<A>.smem[p_off + r * (BC + 1) + sc] = p;
        Shared<A>.smem[ds_off + r * (BC + 1) + sc] = p * (dp - Shared<A>.smem[delta_off + r]);
    }
}

void loadRowStats<T : ITorchVulkan>(uint t, uint base, uint r0, AttentionBackwardPushConstants pcs, uint lse_off, uint delta_off)
{
    typealias A = T.Accum;
    if (t < BR)
    {
        uint row = min(r0 + t, pcs.L - 1);
        Shared<A>.smem[lse_off + t] = ((A*)pcs.lse)[base + row];
        Shared<A>.smem[delta_off + t] = ((A*)pcs.delta)[base + row];
    }
}

void attentionBackwardKV<T : ITorchVulkan>(uint3 wg, uint t, AttentionBackwardPushConstants pcs)
{
    typealias A = T.Accum;
    uint c0 = wg.x * BC;
    uint kvh = wg.y;
    uint b = wg.z;
    uint kv_heads = pcs.heads / GROUP;

    // shared: k[BC][D + 1], v[BC][DV + 1], q[BR][D + 1], dO[BR][DV + 1], p[BR][BC + 1], ds[BR][BC + 1], lse[BR], delta[BR]
    uint k_off = 0;
    uint v_off = k_off + BC * (D + 1);
    uint q_off = v_off + BC * (DV + 1);
    uint do_off = q_off + BR * (D + 1);
    uint p_off = do_off + BR * (DV + 1);
    uint ds_off = p_off + BR * (BC + 1);
    uint lse_off = ds_off + BR * (BC + 1);
    uint delta_off = lse_off + BR;

    uint kv_row = b * kv_heads + kvh;
    loadRows<T>(t, (T*)pcs.k, kv_row * pcs.S * D, c0, pcs.S, D, k_off, BC);
    loadRows<T>(t, (T*)pcs.v, kv_row * pcs.S * DV, c0, pcs.S, DV, v_off, BC);

    A dk[ACC_A];
    A dv[ACC_B];
    [unroll] for (uint j = 0; j < ACC_A; j++) dk[j] = A.ext_zero();
    [unroll] for (uint j = 0; j < ACC_B; j++) dv[j] = A.ext_zero();

    uint r_start = CAUSAL == 1 ? (c0 / BR) * BR : 0;

    for (uint g = 0; g < GROUP; g++)
    {
        uint h = kvh * GROUP + g;
        uint q_row = (b * pcs.heads + h) * pcs.L;
        for (uint r0 = r_start; r0 < pcs.L; r0 += BR)
        {
            GroupMemoryBarrierWithGroupSync();
            loadRows<T>(t, (T*)pcs.q, q_row * D, r0, pcs.L, D, q_off, BR);
            loadRows<T>(t, (T*)pcs.grad_out, q_row * DV, r0, pcs.L, DV, do_off, BR);
            loadRowStats<T>(t, q_row, r0, pcs, lse_off, delta_off);
            GroupMemoryBarrierWithGroupSync();

            probabilityGradients<T>(t, b, h, r0, c0, pcs, q_off, do_off, k_off, v_off, lse_off, delta_off, p_off, ds_off, true);
            GroupMemoryBarrierWithGroupSync();

            [unroll] for (uint j = 0; j < ACC_B; j++)
            {
                uint e = min(t + j * WG, BC * DV - 1);
                uint c = e / DV;
                uint d = e % DV;
                A acc = dv[j];
                for (uint r = 0; r < BR; r++) acc = acc + Shared<A>.smem[p_off + r * (BC + 1) + c] * Shared<A>.smem[do_off + r * (DV + 1) + d];
                dv[j] = acc;
            }
            [unroll] for (uint j = 0; j < ACC_A; j++)
            {
                uint e = min(t + j * WG, BC * D - 1);
                uint c = e / D;
                uint d = e % D;
                A acc = dk[j];
                for (uint r = 0; r < BR; r++) acc = acc + Shared<A>.smem[ds_off + r * (BC + 1) + c] * Shared<A>.smem[q_off + r * (D + 1) + d];
                dk[j] = acc;
            }
        }
    }

    A scale = A.castFromF32(pcs.scale);
    T* grad_k = (T*)pcs.grad_a;
    T* grad_v = (T*)pcs.grad_b;
    [unroll] for (uint j = 0; j < ACC_A; j++)
    {
        uint e = t + j * WG;
        uint key = c0 + min(e, BC * D - 1) / D;
        if (e < BC * D && key < pcs.S) grad_k[(kv_row * pcs.S + key) * D + e % D] = T.fromAccum(dk[j] * scale);
    }
    [unroll] for (uint j = 0; j < ACC_B; j++)
    {
        uint e = t + j * WG;
        uint key = c0 + min(e, BC * DV - 1) / DV;
        if (e < BC * DV && key < pcs.S) grad_v[(kv_row * pcs.S + key) * DV + e % DV] = T.fromAccum(dv[j]);
    }
}

void attentionBackwardQ<T : ITorchVulkan>(uint3 wg, uint t, AttentionBackwardPushConstants pcs)
{
    typealias A = T.Accum;
    uint r0 = wg.x * BR;
    uint h = wg.y;
    uint b = wg.z;
    uint kv_heads = pcs.heads / GROUP;
    uint kv_row = b * kv_heads + h / GROUP;
    uint q_row = (b * pcs.heads + h) * pcs.L;

    // shared: q[BR][D + 1], dO[BR][DV + 1], k[BC][D + 1], v[BC][DV + 1], ds[BR][BC + 1], lse[BR], delta[BR]
    uint q_off = 0;
    uint do_off = q_off + BR * (D + 1);
    uint k_off = do_off + BR * (DV + 1);
    uint v_off = k_off + BC * (D + 1);
    uint ds_off = v_off + BC * (DV + 1);
    uint lse_off = ds_off + BR * (BC + 1);
    uint delta_off = lse_off + BR;

    loadRows<T>(t, (T*)pcs.q, q_row * D, r0, pcs.L, D, q_off, BR);
    loadRows<T>(t, (T*)pcs.grad_out, q_row * DV, r0, pcs.L, DV, do_off, BR);
    loadRowStats<T>(t, q_row, r0, pcs, lse_off, delta_off);

    A dq[ACC_A];
    [unroll] for (uint j = 0; j < ACC_A; j++) dq[j] = A.ext_zero();

    uint kv_end = pcs.S;
    if (CAUSAL == 1) kv_end = min(pcs.S, r0 + BR);

    for (uint c0 = 0; c0 < kv_end; c0 += BC)
    {
        GroupMemoryBarrierWithGroupSync();
        loadRows<T>(t, (T*)pcs.k, kv_row * pcs.S * D, c0, pcs.S, D, k_off, BC);
        loadRows<T>(t, (T*)pcs.v, kv_row * pcs.S * DV, c0, pcs.S, DV, v_off, BC);
        GroupMemoryBarrierWithGroupSync();

        probabilityGradients<T>(t, b, h, r0, c0, pcs, q_off, do_off, k_off, v_off, lse_off, delta_off, ds_off, ds_off, false);
        GroupMemoryBarrierWithGroupSync();

        [unroll] for (uint j = 0; j < ACC_A; j++)
        {
            uint e = min(t + j * WG, BR * D - 1);
            uint r = e / D;
            uint d = e % D;
            A acc = dq[j];
            for (uint c = 0; c < BC; c++) acc = acc + Shared<A>.smem[ds_off + r * (BC + 1) + c] * Shared<A>.smem[k_off + c * (D + 1) + d];
            dq[j] = acc;
        }
    }

    A scale = A.castFromF32(pcs.scale);
    T* grad_q = (T*)pcs.grad_a;
    [unroll] for (uint j = 0; j < ACC_A; j++)
    {
        uint e = t + j * WG;
        uint row = r0 + min(e, BR * D - 1) / D;
        if (e < BR * D && row < pcs.L) grad_q[(q_row + row) * D + e % D] = T.fromAccum(dq[j] * scale);
    }
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void attention_simd_main<T : ITorchVulkan>(
    uint3 wg : SV_GroupID,
    uint t : SV_GroupIndex,
    uniform AttentionPushConstants pcs)
{
    attentionForward<T>(wg, t, pcs);
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void attention_simd_backward_main<T : ITorchVulkan>(
    uint3 wg : SV_GroupID,
    uint t : SV_GroupIndex,
    uniform AttentionBackwardPushConstants pcs)
{
    if (MODE == 1) attentionBackwardKV<T>(wg, t, pcs);
    else attentionBackwardQ<T>(wg, t, pcs);
}
