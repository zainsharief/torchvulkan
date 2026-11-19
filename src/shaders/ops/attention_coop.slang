module attention_coop;

__exported import core.dtype;
import helpers.attention;

[[vk::constant_id(0)]] public const uint WG = 128;
[[vk::constant_id(1)]] public const uint SUBGROUP_SIZE = 32;
[[vk::constant_id(2)]] public const uint BR = 16;
[[vk::constant_id(3)]] public const uint BC = 32;
[[vk::constant_id(4)]] public const uint D = 64;
[[vk::constant_id(5)]] public const uint DV = 64;
[[vk::constant_id(6)]] public const uint DP = 64;
[[vk::constant_id(7)]] public const uint DVP = 64;
[[vk::constant_id(8)]] public const uint GROUP = 1;
[[vk::constant_id(9)]] public const uint HAS_BIAS = 0;
[[vk::constant_id(10)]] public const uint CAUSAL = 0;
[[vk::constant_id(11)]] public const uint BLOCK_SIZE = 16;
[[vk::constant_id(12)]] public const uint SMEM_T = 1;
[[vk::constant_id(13)]] public const uint SMEM_A = 1;
[[vk::constant_id(14)]] public const uint S_COLS = 1;
[[vk::constant_id(15)]] public const uint O_COLS = 1;
[[vk::constant_id(16)]] public const uint SPLITS = 1;
[[vk::constant_id(17)]] public const uint SPLIT_KEYS = 0;
[[vk::constant_id(18)]] public const uint MODE = 1;
[[vk::constant_id(19)]] public const uint FRAGS = 1;

// Slang can't size a CoopMat by a spec constant: scripts/splice_coop_dims.py swaps this placeholder for BLOCK_SIZE,
// so it must only appear in the typealiases below
static const uint COOP_DIM = 1009;

// keeps shared-memory rows 16-byte aligned for cooperative-matrix loads
public static const uint PAD = 8;

public typealias Accumulator<U : __BuiltinArithmeticType> =
    linalg::CoopMat<U, MemoryScope.Subgroup, COOP_DIM, COOP_DIM, linalg::CoopMatMatrixUse.MatrixAccumulator>;
public typealias FragmentA<T : ICoopElement> =
    linalg::CoopMat<T, MemoryScope.Subgroup, COOP_DIM, COOP_DIM, linalg::CoopMatMatrixUse.MatrixA>;
public typealias FragmentB<T : ICoopElement> =
    linalg::CoopMat<T, MemoryScope.Subgroup, COOP_DIM, COOP_DIM, linalg::CoopMatMatrixUse.MatrixB>;

// T: q[BR][DP + PAD], kv[BC][max(DP, DVP) + PAD], p[BR][BC + PAD]
public struct SharedT<T>
{
    public static groupshared T smem[SMEM_T];
}

// Accum: the block's scores, then its P V, as [BR][max(BC, DVP) + PAD]; then WG reduction slots
public struct SharedA<A>
{
    public static groupshared A smem[SMEM_A];
}

bool isMasked(uint row, uint key, uint L, uint S)
{
    return key >= S || row >= L || (CAUSAL == 1 && key > row);
}

void loadBlock<T : ITorchVulkan & ICoopElement>(
    uint t, T* src, uint base, uint row_stride, uint r0, uint rows, uint count, uint width, uint padded, uint off, uint stride)
{
    for (uint i = t; i < count * padded; i += WG)
    {
        uint r = i / padded;
        uint d = i % padded;
        uint row = min(r0 + r, rows - 1);
        T value = src[base + row * row_stride + min(d, width - 1)];
        SharedT<T>.smem[off + r * stride + d] = (d < width && r0 + r < rows) ? value : T.ext_zero();
    }
}

void attentionCoop<T : ITorchVulkan & ICoopElement>(uint3 wg, uint t, AttentionPushConstants pcs)
{
    typealias A = T.Accum;

    uint row0 = wg.x * BR;
    uint h = wg.y;
    uint z = wg.z;
    uint b = z / SPLITS;
    uint kvh = h / GROUP;
    uint q_base = b * pcs.q_batch + h * pcs.q_head;
    uint k_base = b * pcs.k_batch + kvh * pcs.k_head;
    uint v_base = b * pcs.v_batch + kvh * pcs.v_head;
    uint4 bias_strides = uint4(pcs.bias_batch, pcs.bias_head, pcs.bias_row, pcs.bias_col);

    const uint QW = DP + PAD;
    const uint KVW = (DP > DVP ? DP : DVP) + PAD;
    const uint PW = BC + PAD;
    const uint SW = (BC > DVP ? BC : DVP) + PAD;
    uint kv_off = BR * QW;
    uint p_off = kv_off + BC * KVW;
    uint red_off = BR * SW;

    loadBlock<T>(t, (T*)pcs.q, q_base, pcs.q_row, row0, pcs.L, BR, D, DP, 0, QW);

    const uint TPR = WG / BR;
    uint sr = t / TPR;
    uint sj = t % TPR;
    uint warp = t / SUBGROUP_SIZE;
    uint warps = WG / SUBGROUP_SIZE;
    uint row = row0 + sr;

    A o[O_COLS];
    [unroll] for (uint i = 0; i < O_COLS; i++) o[i] = A.ext_zero();
    A lowest = maskedScore<A>();
    A scale = A.castFromF32(pcs.scale);
    A m = lowest;
    A l = A.ext_zero();

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
        loadBlock<T>(t, (T*)pcs.k, k_base, pcs.k_row, c0, pcs.S, BC, D, DP, kv_off, KVW);
        GroupMemoryBarrierWithGroupSync();

        uint blocks_c = BC / BLOCK_SIZE;
        for (uint f = warp; f < (BR / BLOCK_SIZE) * blocks_c; f += warps)
        {
            uint rb = f / blocks_c;
            uint cb = f % blocks_c;
            Accumulator<A> acc = Accumulator<A>(A.ext_zero());
            for (uint ks = 0; ks < DP; ks += BLOCK_SIZE)
            {
                FragmentA<T> fa;
                fa = fa.Load<linalg::CoopMatMatrixLayout.RowMajor>(SharedT<T>.smem, rb * BLOCK_SIZE * QW + ks, QW);
                FragmentB<T> fb;
                fb = fb.Load<linalg::CoopMatMatrixLayout.ColumnMajor>(SharedT<T>.smem, kv_off + cb * BLOCK_SIZE * KVW + ks, KVW);
                acc = linalg::coopMatMulAdd<A, false>(fa, fb, acc);
            }
            acc.Store<linalg::CoopMatMatrixLayout.RowMajor>(SharedA<A>.smem, rb * BLOCK_SIZE * SW + cb * BLOCK_SIZE, SW);
        }
        GroupMemoryBarrierWithGroupSync();

        loadBlock<T>(t, (T*)pcs.v, v_base, pcs.v_row, c0, pcs.S, BC, DV, DVP, kv_off, KVW);

        A s[S_COLS];
        A local_max = lowest;
        [unroll] for (uint i = 0; i < S_COLS; i++)
        {
            uint c = sj + i * TPR;
            uint key = c0 + c;
            A bias = A.ext_zero();
            if (HAS_BIAS == 1) bias = biasAt<T>(pcs.bias, b, h, row, key, pcs.L, pcs.S, bias_strides);
            A value = score<T>(SharedA<A>.smem[sr * SW + c], scale, HAS_BIAS == 1, bias);
            s[i] = isMasked(row, key, pcs.L, pcs.S) ? lowest : value;
            local_max = A.ext_max(local_max, s[i]);
        }
        SharedA<A>.smem[red_off + t] = local_max;
        GroupMemoryBarrierWithGroupSync();

        A block_max = lowest;
        for (uint j = 0; j < TPR; j++) block_max = A.ext_max(block_max, SharedA<A>.smem[red_off + sr * TPR + j]);
        A m_new = A.ext_max(m, block_max);
        bool empty = A.ext_eq(m_new, lowest);
        A alpha = empty ? A.ext_one() : A.ext_exp(m - m_new);
        A sum = A.ext_zero();
        [unroll] for (uint i = 0; i < S_COLS; i++)
        {
            A p = (empty || A.ext_eq(s[i], lowest)) ? A.ext_zero() : A.ext_exp(s[i] - m_new);
            // the sum takes P as the product reads it, rounded to T
            T rounded = T.fromAccum(p);
            SharedT<T>.smem[p_off + sr * PW + sj + i * TPR] = rounded;
            sum = sum + rounded.toAccum();
        }
        l = l * alpha + sum;
        m = m_new;
        [unroll] for (uint i = 0; i < O_COLS; i++) o[i] = o[i] * alpha;
        GroupMemoryBarrierWithGroupSync();

        uint blocks_v = DVP / BLOCK_SIZE;
        for (uint f = warp; f < (BR / BLOCK_SIZE) * blocks_v; f += warps)
        {
            uint rb = f / blocks_v;
            uint vb = f % blocks_v;
            Accumulator<A> acc = Accumulator<A>(A.ext_zero());
            for (uint ks = 0; ks < BC; ks += BLOCK_SIZE)
            {
                FragmentA<T> fa;
                fa = fa.Load<linalg::CoopMatMatrixLayout.RowMajor>(SharedT<T>.smem, p_off + rb * BLOCK_SIZE * PW + ks, PW);
                FragmentB<T> fb;
                fb = fb.Load<linalg::CoopMatMatrixLayout.RowMajor>(SharedT<T>.smem, kv_off + ks * KVW + vb * BLOCK_SIZE, KVW);
                acc = linalg::coopMatMulAdd<A, false>(fa, fb, acc);
            }
            acc.Store<linalg::CoopMatMatrixLayout.RowMajor>(SharedA<A>.smem, rb * BLOCK_SIZE * SW + vb * BLOCK_SIZE, SW);
        }
        GroupMemoryBarrierWithGroupSync();

        [unroll] for (uint i = 0; i < O_COLS; i++) o[i] = o[i] + SharedA<A>.smem[sr * SW + sj + i * TPR];
    }

    GroupMemoryBarrierWithGroupSync();
    SharedA<A>.smem[red_off + t] = l;
    GroupMemoryBarrierWithGroupSync();
    A total = A.ext_zero();
    for (uint j = 0; j < TPR; j++) total = total + SharedA<A>.smem[red_off + sr * TPR + j];
    bool attended = A.ext_gt(total, A.ext_zero());

    // a row that attends to nothing gets a logsumexp of +inf
    if (sj == 0 && row < pcs.L)
    {
        ((A*)pcs.lse)[(z * pcs.heads + h) * pcs.L + row] = attended ? m + A.ext_log(total) : positiveInfinity<A>();
    }

    uint out_base = z * pcs.out_batch + h * pcs.out_head;
    [unroll] for (uint i = 0; i < O_COLS; i++)
    {
        uint d = sj + i * TPR;
        A value = attended ? o[i] / total : A.ext_zero();
        if (row < pcs.L && d < DV)
        {
            uint index = out_base + row * pcs.out_row + d;
            if (SPLITS > 1) ((A*)pcs.out)[index] = value;
            else ((T*)pcs.out)[index] = T.fromAccum(value);
        }
    }
}

// Backward. Shared T: k[BC][DP + PAD], v[BC][DVP + PAD], q[BR][DP + PAD], dO[BR][DVP + PAD], p[BR][BC + PAD],
// ds[BR][BC + PAD]. Shared Accum: s[BR][BC + PAD], dp[BR][BC + PAD], lse[BR], delta[BR], then a BLOCK_SIZE square
// per subgroup for writing the gradients out
struct BackwardLayout
{
    uint KW, VW, PW;
    uint k_off, v_off, q_off, do_off, p_off, ds_off;
    uint dp_off, lse_off, delta_off, scratch_off;
}

BackwardLayout backwardLayout()
{
    BackwardLayout m;
    m.KW = DP + PAD;
    m.VW = DVP + PAD;
    m.PW = BC + PAD;
    m.k_off = 0;
    m.v_off = BC * m.KW;
    m.q_off = m.v_off + BC * m.VW;
    m.do_off = m.q_off + BR * m.KW;
    m.p_off = m.do_off + BR * m.VW;
    m.ds_off = m.p_off + BR * m.PW;
    m.dp_off = BR * m.PW;
    m.lse_off = 2 * BR * m.PW;
    m.delta_off = m.lse_off + BR;
    m.scratch_off = m.delta_off + BR;
    return m;
}

void scoreProducts<T : ITorchVulkan & ICoopElement>(uint warp, BackwardLayout m)
{
    typealias A = T.Accum;
    uint blocks_c = BC / BLOCK_SIZE;
    uint per = (BR / BLOCK_SIZE) * blocks_c;
    for (uint f = warp; f < 2 * per; f += WG / SUBGROUP_SIZE)
    {
        bool grad = f >= per;
        uint g = grad ? f - per : f;
        uint rb = g / blocks_c;
        uint cb = g % blocks_c;
        uint width = grad ? DVP : DP;
        uint stride = grad ? m.VW : m.KW;
        uint a_off = (grad ? m.do_off : m.q_off) + rb * BLOCK_SIZE * stride;
        uint b_off = (grad ? m.v_off : m.k_off) + cb * BLOCK_SIZE * stride;
        Accumulator<A> acc = Accumulator<A>(A.ext_zero());
        for (uint ks = 0; ks < width; ks += BLOCK_SIZE)
        {
            FragmentA<T> fa;
            fa = fa.Load<linalg::CoopMatMatrixLayout.RowMajor>(SharedT<T>.smem, a_off + ks, stride);
            FragmentB<T> fb;
            fb = fb.Load<linalg::CoopMatMatrixLayout.ColumnMajor>(SharedT<T>.smem, b_off + ks, stride);
            acc = linalg::coopMatMulAdd<A, false>(fa, fb, acc);
        }
        acc.Store<linalg::CoopMatMatrixLayout.RowMajor>(SharedA<A>.smem, (grad ? m.dp_off : 0) + rb * BLOCK_SIZE * m.PW + cb * BLOCK_SIZE, m.PW);
    }
}

void scoreGradients<T : ITorchVulkan & ICoopElement>(uint t, uint b, uint h, uint r0, uint c0, AttentionBackwardPushConstants pcs, BackwardLayout m)
{
    typealias A = T.Accum;
    A scale = A.castFromF32(pcs.scale);
    A lowest = maskedScore<A>();
    uint4 bias_strides = uint4(pcs.bias_batch, pcs.bias_head, pcs.bias_row, pcs.bias_col);
    for (uint i = t; i < BR * BC; i += WG)
    {
        uint r = i / BC;
        uint c = i % BC;
        uint row = r0 + r;
        uint key = c0 + c;
        A bias = A.ext_zero();
        if (HAS_BIAS == 1) bias = biasAt<T>(pcs.bias, b, h, row, key, pcs.L, pcs.S, bias_strides);
        A s = score<T>(SharedA<A>.smem[r * m.PW + c], scale, HAS_BIAS == 1, bias);
        A lse = SharedA<A>.smem[m.lse_off + r];
        // a row that attends to nothing has every score masked
        bool masked = isMasked(row, key, pcs.L, pcs.S) || A.ext_eq(s, lowest) || A.ext_eq(lse, positiveInfinity<A>());
        A p = masked ? A.ext_zero() : A.ext_exp(s - lse);
        A ds = p * (SharedA<A>.smem[m.dp_off + r * m.PW + c] - SharedA<A>.smem[m.delta_off + r]);
        SharedT<T>.smem[m.p_off + r * m.PW + c] = T.fromAccum(p);
        SharedT<T>.smem[m.ds_off + r * m.PW + c] = T.fromAccum(ds);
    }
}

void loadRowStats<T : ITorchVulkan & ICoopElement>(uint t, uint base, uint r0, AttentionBackwardPushConstants pcs, BackwardLayout m)
{
    typealias A = T.Accum;
    for (uint r = t; r < BR; r += WG)
    {
        uint row = min(r0 + r, pcs.L - 1);
        SharedA<A>.smem[m.lse_off + r] = ((A*)pcs.lse)[base + row];
        SharedA<A>.smem[m.delta_off + r] = ((A*)pcs.delta)[base + row];
    }
}

void storeGradient<T : ITorchVulkan & ICoopElement>(
    bool live, uint lane, uint warp, Accumulator<T.Accum> acc, T* dst, uint base, uint row0, uint rows, uint col0, uint width, T.Accum scale,
    BackwardLayout m)
{
    typealias A = T.Accum;
    uint scratch = m.scratch_off + warp * BLOCK_SIZE * BLOCK_SIZE;
    acc.Store<linalg::CoopMatMatrixLayout.RowMajor>(SharedA<A>.smem, scratch, BLOCK_SIZE);
    GroupMemoryBarrierWithWaveSync();
    for (uint i = lane; i < BLOCK_SIZE * BLOCK_SIZE; i += SUBGROUP_SIZE)
    {
        uint r = row0 + i / BLOCK_SIZE;
        uint c = col0 + i % BLOCK_SIZE;
        if (live && r < rows && c < width) dst[base + r * width + c] = T.fromAccum(SharedA<A>.smem[scratch + i] * scale);
    }
    GroupMemoryBarrierWithWaveSync();
}

void attentionCoopBackwardKV<T : ITorchVulkan & ICoopElement>(uint3 wg, uint t, AttentionBackwardPushConstants pcs)
{
    typealias A = T.Accum;
    BackwardLayout m = backwardLayout();
    uint c0 = wg.x * BC;
    uint kvh = wg.y;
    uint b = wg.z;
    uint kv_row = b * (pcs.heads / GROUP) + kvh;
    uint warp = t / SUBGROUP_SIZE;
    uint warps = WG / SUBGROUP_SIZE;

    loadBlock<T>(t, (T*)pcs.k, kv_row * pcs.S * D, D, c0, pcs.S, BC, D, DP, m.k_off, m.KW);
    loadBlock<T>(t, (T*)pcs.v, kv_row * pcs.S * DV, DV, c0, pcs.S, BC, DV, DVP, m.v_off, m.VW);

    // blocks [0, nk) are dK's, [nk, nk + nv) dV's; subgroup w owns w, w + warps, ...
    uint dk_cols = DP / BLOCK_SIZE;
    uint dv_cols = DVP / BLOCK_SIZE;
    uint nk = (BC / BLOCK_SIZE) * dk_cols;
    uint total = nk + (BC / BLOCK_SIZE) * dv_cols;
    Accumulator<A> acc[FRAGS];
    [unroll] for (uint i = 0; i < FRAGS; i++) acc[i] = Accumulator<A>(A.ext_zero());

    uint r_start = CAUSAL == 1 ? (c0 / BR) * BR : 0;

    for (uint g = 0; g < GROUP; g++)
    {
        uint h = kvh * GROUP + g;
        uint q_row = (b * pcs.heads + h) * pcs.L;
        for (uint r0 = r_start; r0 < pcs.L; r0 += BR)
        {
            GroupMemoryBarrierWithGroupSync();
            loadBlock<T>(t, (T*)pcs.q, q_row * D, D, r0, pcs.L, BR, D, DP, m.q_off, m.KW);
            loadBlock<T>(t, (T*)pcs.grad_out, q_row * DV, DV, r0, pcs.L, BR, DV, DVP, m.do_off, m.VW);
            loadRowStats<T>(t, q_row, r0, pcs, m);
            GroupMemoryBarrierWithGroupSync();

            scoreProducts<T>(warp, m);
            GroupMemoryBarrierWithGroupSync();
            scoreGradients<T>(t, b, h, r0, c0, pcs, m);
            GroupMemoryBarrierWithGroupSync();

            [unroll] for (uint i = 0; i < FRAGS; i++)
            {
                uint f = warp + i * warps;
                if (f < total)
                {
                    bool value = f >= nk;
                    uint g2 = value ? f - nk : f;
                    uint cols = value ? dv_cols : dk_cols;
                    uint kb = g2 / cols;
                    uint xb = g2 % cols;
                    // P^T and dS^T are P and dS read column-major
                    uint a_off = (value ? m.p_off : m.ds_off) + kb * BLOCK_SIZE;
                    uint b_off = (value ? m.do_off : m.q_off) + xb * BLOCK_SIZE;
                    uint b_stride = value ? m.VW : m.KW;
                    for (uint ks = 0; ks < BR; ks += BLOCK_SIZE)
                    {
                        FragmentA<T> fa;
                        fa = fa.Load<linalg::CoopMatMatrixLayout.ColumnMajor>(SharedT<T>.smem, a_off + ks * m.PW, m.PW);
                        FragmentB<T> fb;
                        fb = fb.Load<linalg::CoopMatMatrixLayout.RowMajor>(SharedT<T>.smem, b_off + ks * b_stride, b_stride);
                        acc[i] = linalg::coopMatMulAdd<A, false>(fa, fb, acc[i]);
                    }
                }
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();
    A scale = A.castFromF32(pcs.scale);
    uint lane = t % SUBGROUP_SIZE;
    [unroll] for (uint i = 0; i < FRAGS; i++)
    {
        uint f = warp + i * warps;
        bool value = f >= nk;
        uint g2 = value ? f - nk : f;
        uint cols = value ? dv_cols : dk_cols;
        uint row0 = c0 + (g2 / cols) * BLOCK_SIZE;
        uint col0 = (g2 % cols) * BLOCK_SIZE;
        uint width = value ? DV : D;
        T* dst = (T*)(value ? pcs.grad_b : pcs.grad_a);
        storeGradient<T>(f < total, lane, warp, acc[i], dst, kv_row * pcs.S * width, row0, pcs.S, col0, width, value ? A.ext_one() : scale, m);
    }
}

void attentionCoopBackwardQ<T : ITorchVulkan & ICoopElement>(uint3 wg, uint t, AttentionBackwardPushConstants pcs)
{
    typealias A = T.Accum;
    BackwardLayout m = backwardLayout();
    uint r0 = wg.x * BR;
    uint h = wg.y;
    uint b = wg.z;
    uint kv_row = b * (pcs.heads / GROUP) + h / GROUP;
    uint q_row = (b * pcs.heads + h) * pcs.L;
    uint warp = t / SUBGROUP_SIZE;
    uint warps = WG / SUBGROUP_SIZE;

    loadBlock<T>(t, (T*)pcs.q, q_row * D, D, r0, pcs.L, BR, D, DP, m.q_off, m.KW);
    loadBlock<T>(t, (T*)pcs.grad_out, q_row * DV, DV, r0, pcs.L, BR, DV, DVP, m.do_off, m.VW);
    loadRowStats<T>(t, q_row, r0, pcs, m);

    uint dq_cols = DP / BLOCK_SIZE;
    uint total = (BR / BLOCK_SIZE) * dq_cols;
    Accumulator<A> acc[FRAGS];
    [unroll] for (uint i = 0; i < FRAGS; i++) acc[i] = Accumulator<A>(A.ext_zero());

    uint kv_end = pcs.S;
    if (CAUSAL == 1) kv_end = min(pcs.S, r0 + BR);

    for (uint c0 = 0; c0 < kv_end; c0 += BC)
    {
        GroupMemoryBarrierWithGroupSync();
        loadBlock<T>(t, (T*)pcs.k, kv_row * pcs.S * D, D, c0, pcs.S, BC, D, DP, m.k_off, m.KW);
        loadBlock<T>(t, (T*)pcs.v, kv_row * pcs.S * DV, DV, c0, pcs.S, BC, DV, DVP, m.v_off, m.VW);
        GroupMemoryBarrierWithGroupSync();

        scoreProducts<T>(warp, m);
        GroupMemoryBarrierWithGroupSync();
        scoreGradients<T>(t, b, h, r0, c0, pcs, m);
        GroupMemoryBarrierWithGroupSync();

        [unroll] for (uint i = 0; i < FRAGS; i++)
        {
            uint f = warp + i * warps;
            if (f < total)
            {
                uint rb = f / dq_cols;
                uint db = f % dq_cols;
                for (uint ks = 0; ks < BC; ks += BLOCK_SIZE)
                {
                    FragmentA<T> fa;
                    fa = fa.Load<linalg::CoopMatMatrixLayout.RowMajor>(SharedT<T>.smem, m.ds_off + rb * BLOCK_SIZE * m.PW + ks, m.PW);
                    FragmentB<T> fb;
                    fb = fb.Load<linalg::CoopMatMatrixLayout.RowMajor>(SharedT<T>.smem, m.k_off + ks * m.KW + db * BLOCK_SIZE, m.KW);
                    acc[i] = linalg::coopMatMulAdd<A, false>(fa, fb, acc[i]);
                }
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();
    A scale = A.castFromF32(pcs.scale);
    uint lane = t % SUBGROUP_SIZE;
    [unroll] for (uint i = 0; i < FRAGS; i++)
    {
        uint f = warp + i * warps;
        uint row0 = r0 + (f / dq_cols) * BLOCK_SIZE;
        storeGradient<T>(f < total, lane, warp, acc[i], (T*)pcs.grad_a, q_row * D, row0, pcs.L, (f % dq_cols) * BLOCK_SIZE, D, scale, m);
    }
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void attention_coop_backward_main<T : ITorchVulkan & ICoopElement>(
    uint3 wg : SV_GroupID,
    uint t : SV_GroupIndex,
    uniform AttentionBackwardPushConstants pcs)
{
    if (MODE == 1) attentionCoopBackwardKV<T>(wg, t, pcs);
    else attentionCoopBackwardQ<T>(wg, t, pcs);
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void attention_coop_main<T : ITorchVulkan & ICoopElement>(
    uint3 wg : SV_GroupID,
    uint t : SV_GroupIndex,
    uniform AttentionPushConstants pcs)
{
    attentionCoop<T>(wg, t, pcs);
}
