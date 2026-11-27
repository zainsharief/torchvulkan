module attention_warp;

__exported import core.dtype;
import helpers.attention;

// Forward attention where each subgroup owns BLOCK_SIZE query rows from start to end: its Q fragments and its output
// accumulators stay in registers for the whole key loop, and only the scores go through (subgroup-private) shared
// memory for the softmax. The workgroup shares the key and value blocks. Cooperative matrices don't say which row an
// accumulator element is in, so a matrix holding each element's row index, loaded once, tells the rescale which
// row's factor to use (matrices of one type share their element layout).
[[vk::constant_id(0)]] public const uint WG = 128;
[[vk::constant_id(1)]] public const uint SUBGROUP_SIZE = 32;
[[vk::constant_id(2)]] public const uint BC = 32;
[[vk::constant_id(3)]] public const uint D = 64;
[[vk::constant_id(4)]] public const uint DV = 64;
[[vk::constant_id(5)]] public const uint DP = 64;
[[vk::constant_id(6)]] public const uint DVP = 64;
[[vk::constant_id(7)]] public const uint GROUP = 1;
[[vk::constant_id(8)]] public const uint HAS_BIAS = 0;
[[vk::constant_id(9)]] public const uint CAUSAL = 0;
[[vk::constant_id(10)]] public const uint BLOCK_SIZE = 16;
[[vk::constant_id(11)]] public const uint SPLITS = 1;
[[vk::constant_id(12)]] public const uint SPLIT_KEYS = 0;
// array sizes, from the host: Slang can't size an array by an expression of spec constants
[[vk::constant_id(13)]] public const uint SMEM_T = 1;
[[vk::constant_id(14)]] public const uint SMEM_A = 1;
[[vk::constant_id(15)]] public const uint Q_FRAGS = 1;   // query columns / BLOCK_SIZE
[[vk::constant_id(16)]] public const uint O_FRAGS = 1;   // value columns / BLOCK_SIZE

// scripts/splice_coop_dims.py swaps this placeholder for BLOCK_SIZE, so it must only appear in the typealiases below
static const uint COOP_DIM = 1009;

// keeps shared-memory rows 16-byte aligned for cooperative-matrix loads
static const uint PAD = 8;

typealias Accumulator<U : __BuiltinArithmeticType> =
    linalg::CoopMat<U, MemoryScope.Subgroup, COOP_DIM, COOP_DIM, linalg::CoopMatMatrixUse.MatrixAccumulator>;
typealias FragmentA<T : ICoopElement> =
    linalg::CoopMat<T, MemoryScope.Subgroup, COOP_DIM, COOP_DIM, linalg::CoopMatMatrixUse.MatrixA>;
typealias FragmentB<T : ICoopElement> =
    linalg::CoopMat<T, MemoryScope.Subgroup, COOP_DIM, COOP_DIM, linalg::CoopMatMatrixUse.MatrixB>;

// T: keys [BC][DP + PAD], values [BC][DVP + PAD], then per subgroup a [BLOCK_SIZE][BC + PAD] block for its
// probabilities (and, before the key loop, its query columns)
struct SharedT<T>
{
    static groupshared T smem[SMEM_T];
}

// Accum: per subgroup a [BLOCK_SIZE][BC + PAD] block of scores, then the row-index block, then per subgroup the rows'
// rescale factors [BLOCK_SIZE], a spare [BLOCK_SIZE] and the lanes' partial reductions [SUBGROUP_SIZE]
struct SharedA<A>
{
    static groupshared A smem[SMEM_A];
}

bool isMasked(uint row, uint key, uint L, uint S)
{
    return key >= S || row >= L || (CAUSAL == 1 && key > row);
}

// rows [r0, r0 + count) x width columns of a [rows][row_stride] source into shared memory with zeros past its edges
void loadRows<T : ITorchVulkan & ICoopElement>(uint t, T* src, uint base, uint row_stride, uint r0, uint rows, uint count, uint width, uint padded, uint off, uint stride)
{
    for (uint i = t; i < count * padded; i += WG)
    {
        uint r = i / padded;
        uint d = i - r * padded;
        bool inside = d < width && r0 + r < rows;
        T value = src[base + min(r0 + r, rows - 1) * row_stride + min(d, width - 1)];
        SharedT<T>.smem[off + r * stride + d] = inside ? value : T.ext_zero();
    }
}

void attentionWarp<T : ITorchVulkan & ICoopElement>(uint3 wg, uint t, AttentionPushConstants pcs) where T.Accum : __BuiltinArithmeticType
{
    typealias A = T.Accum;

    const uint WARPS = WG / SUBGROUP_SIZE;
    const uint KW = DP + PAD;
    const uint VW = DVP + PAD;
    const uint PW = BC + PAD;
    const uint LPR = SUBGROUP_SIZE / BLOCK_SIZE;  // lanes per query row in the softmax
    uint warp = t / SUBGROUP_SIZE;
    uint lane = t % SUBGROUP_SIZE;
    uint row0 = (wg.x * WARPS + warp) * BLOCK_SIZE;
    uint h = wg.y;
    uint z = wg.z;
    uint b = z / SPLITS;
    uint kvh = h / GROUP;
    uint q_base = b * pcs.q_batch + h * pcs.q_head;
    uint k_base = b * pcs.k_batch + kvh * pcs.k_head;
    uint v_base = b * pcs.v_batch + kvh * pcs.v_head;
    uint4 bias_strides = uint4(pcs.bias_batch, pcs.bias_head, pcs.bias_row, pcs.bias_col);

    uint v_off = BC * KW;
    uint p_off = v_off + BC * VW + warp * BLOCK_SIZE * PW;   // this subgroup's T block
    uint s_off = warp * BLOCK_SIZE * PW;                      // this subgroup's Accum block
    uint idx_off = WARPS * BLOCK_SIZE * PW;
    uint stat_off = idx_off + BLOCK_SIZE * BLOCK_SIZE + warp * (2 * BLOCK_SIZE + SUBGROUP_SIZE);
    uint red_off = stat_off + 2 * BLOCK_SIZE;

    // each element's row, as an accumulator laid out like the output's
    for (uint i = t; i < BLOCK_SIZE * BLOCK_SIZE; i += WG) SharedA<A>.smem[idx_off + i] = A.castFromU64(uint64_t(i / BLOCK_SIZE));
    GroupMemoryBarrierWithGroupSync();
    Accumulator<A> rowOf = Accumulator<A>.Load<linalg::CoopMatMatrixLayout.RowMajor>(SharedA<A>.smem, idx_off, BLOCK_SIZE);

    // the subgroup's query rows as A fragments, BLOCK_SIZE columns at a time through its own T block
    FragmentA<T> fq[Q_FRAGS];
    [unroll] for (uint ks = 0; ks < Q_FRAGS; ks++)
    {
        for (uint i = lane; i < BLOCK_SIZE * BLOCK_SIZE; i += SUBGROUP_SIZE)
        {
            uint r = i / BLOCK_SIZE;
            uint d = ks * BLOCK_SIZE + i % BLOCK_SIZE;
            bool inside = d < D && row0 + r < pcs.L;
            T value = ((T*)pcs.q)[q_base + min(row0 + r, pcs.L - 1) * pcs.q_row + min(d, D - 1)];
            SharedT<T>.smem[p_off + r * PW + i % BLOCK_SIZE] = inside ? value : T.ext_zero();
        }
        GroupMemoryBarrierWithWaveSync();
        fq[ks] = FragmentA<T>.Load<linalg::CoopMatMatrixLayout.RowMajor>(SharedT<T>.smem, p_off, PW);
        GroupMemoryBarrierWithWaveSync();
    }

    Accumulator<A> o[O_FRAGS];
    [unroll] for (uint i = 0; i < O_FRAGS; i++) o[i] = Accumulator<A>(A.ext_zero());

    // the softmax lanes: LPR lanes share a row and take every LPR-th key of the block
    uint sr = lane / LPR;
    uint sj = lane % LPR;
    uint row = row0 + sr;
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
    // the workgroup's last row bounds the keys a causal block sees
    if (CAUSAL == 1) kv_end = min(kv_end, wg.x * WARPS * BLOCK_SIZE + WARPS * BLOCK_SIZE);

    for (uint c0 = kv_begin; c0 < kv_end; c0 += BC)
    {
        GroupMemoryBarrierWithGroupSync();
        loadRows<T>(t, (T*)pcs.k, k_base, pcs.k_row, c0, pcs.S, BC, D, DP, 0, KW);
        loadRows<T>(t, (T*)pcs.v, v_base, pcs.v_row, c0, pcs.S, BC, DV, DVP, v_off, VW);
        GroupMemoryBarrierWithGroupSync();

        // S = Q K^T for the subgroup's rows, into its score block
        [unroll] for (uint cb = 0; cb < BC / BLOCK_SIZE; cb++)
        {
            Accumulator<A> acc = Accumulator<A>(A.ext_zero());
            [unroll] for (uint ks = 0; ks < Q_FRAGS; ks++)
            {
                FragmentB<T> fk = FragmentB<T>.Load<linalg::CoopMatMatrixLayout.ColumnMajor>(SharedT<T>.smem, cb * BLOCK_SIZE * KW + ks * BLOCK_SIZE, KW);
                acc = linalg::coopMatMulAdd<A, false>(fq[ks], fk, acc);
            }
            acc.Store<linalg::CoopMatMatrixLayout.RowMajor>(SharedA<A>.smem, s_off + cb * BLOCK_SIZE, PW);
        }
        GroupMemoryBarrierWithWaveSync();

        // online softmax: the block's row maximum, then P = exp(S - m) rounded to T, and the row sum of what the product reads
        A local_max = lowest;
        for (uint c = sj; c < BC; c += LPR)
        {
            uint key = c0 + c;
            A bias = A.ext_zero();
            if (HAS_BIAS == 1) bias = biasAt<T>(pcs.bias, b, h, row, key, pcs.L, pcs.S, bias_strides);
            A value = score<T>(SharedA<A>.smem[s_off + sr * PW + c], scale, HAS_BIAS == 1, bias);
            value = (isMasked(row, key, pcs.L, pcs.S) || key >= kv_end) ? lowest : value;
            SharedA<A>.smem[s_off + sr * PW + c] = value;
            local_max = A.ext_max(local_max, value);
        }
        SharedA<A>.smem[red_off + lane] = local_max;
        GroupMemoryBarrierWithWaveSync();
        A block_max = lowest;
        for (uint j = 0; j < LPR; j++) block_max = A.ext_max(block_max, SharedA<A>.smem[red_off + sr * LPR + j]);
        A m_new = A.ext_max(m, block_max);
        bool empty = A.ext_eq(m_new, lowest);
        A alpha = empty ? A.ext_one() : A.ext_exp(m - m_new);

        A sum = A.ext_zero();
        for (uint c = sj; c < BC; c += LPR)
        {
            A s = SharedA<A>.smem[s_off + sr * PW + c];
            A p = (empty || A.ext_eq(s, lowest)) ? A.ext_zero() : A.ext_exp(s - m_new);
            T rounded = T.fromAccum(p);
            SharedT<T>.smem[p_off + sr * PW + c] = rounded;
            sum = sum + rounded.toAccum();
        }
        GroupMemoryBarrierWithWaveSync();
        SharedA<A>.smem[red_off + lane] = sum;
        GroupMemoryBarrierWithWaveSync();
        A block_sum = A.ext_zero();
        for (uint j = 0; j < LPR; j++) block_sum = block_sum + SharedA<A>.smem[red_off + sr * LPR + j];
        l = l * alpha + block_sum;
        m = m_new;
        if (sj == 0) SharedA<A>.smem[stat_off + sr] = alpha;
        GroupMemoryBarrierWithWaveSync();

        // O = O * alpha[row] + P V
        [unroll] for (uint vb = 0; vb < O_FRAGS; vb++)
        {
            for (uint i = 0; i < o[vb].GetLength(); i++)
            {
                uint r = uint(castValue<A, float>(rowOf[i]) + 0.5f);
                o[vb][i] = o[vb][i] * SharedA<A>.smem[stat_off + r];
            }
        }
        [unroll] for (uint ks = 0; ks < BC / BLOCK_SIZE; ks++)
        {
            FragmentA<T> fp = FragmentA<T>.Load<linalg::CoopMatMatrixLayout.RowMajor>(SharedT<T>.smem, p_off + ks * BLOCK_SIZE, PW);
            [unroll] for (uint vb = 0; vb < O_FRAGS; vb++)
            {
                FragmentB<T> fv = FragmentB<T>.Load<linalg::CoopMatMatrixLayout.RowMajor>(SharedT<T>.smem, v_off + ks * BLOCK_SIZE * VW + vb * BLOCK_SIZE, VW);
                o[vb] = linalg::coopMatMulAdd<A, false>(fp, fv, o[vb]);
            }
        }
        GroupMemoryBarrierWithWaveSync();
    }

    // a row that attends to nothing gets a logsumexp of +inf and zeros
    bool attended = A.ext_gt(l, A.ext_zero());
    if (sj == 0)
    {
        SharedA<A>.smem[stat_off + sr] = attended ? A.ext_one() / l : A.ext_zero();
        if (row < pcs.L) ((A*)pcs.lse)[(z * pcs.heads + h) * pcs.L + row] = attended ? m + A.ext_log(l) : positiveInfinity<A>();
    }
    GroupMemoryBarrierWithWaveSync();

    uint out_base = z * pcs.out_batch + h * pcs.out_head;
    [unroll] for (uint vb = 0; vb < O_FRAGS; vb++)
    {
        Accumulator<A> value = o[vb];
        for (uint i = 0; i < value.GetLength(); i++)
        {
            uint r = uint(castValue<A, float>(rowOf[i]) + 0.5f);
            value[i] = value[i] * SharedA<A>.smem[stat_off + r];
        }
        // through the subgroup's score block, so rows past L and columns past DV stay unwritten
        value.Store<linalg::CoopMatMatrixLayout.RowMajor>(SharedA<A>.smem, s_off, PW);
        GroupMemoryBarrierWithWaveSync();
        for (uint i = lane; i < BLOCK_SIZE * BLOCK_SIZE; i += SUBGROUP_SIZE)
        {
            uint r = row0 + i / BLOCK_SIZE;
            uint d = vb * BLOCK_SIZE + i % BLOCK_SIZE;
            if (r < pcs.L && d < DV)
            {
                A v = SharedA<A>.smem[s_off + (i / BLOCK_SIZE) * PW + i % BLOCK_SIZE];
                uint index = out_base + r * pcs.out_row + d;
                if (SPLITS > 1) ((A*)pcs.out)[index] = v;
                else ((T*)pcs.out)[index] = T.fromAccum(v);
            }
        }
        GroupMemoryBarrierWithWaveSync();
    }
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void attention_warp_main<T : ITorchVulkan & ICoopElement>(
    uint3 wg : SV_GroupID,
    uint t : SV_GroupIndex,
    uniform AttentionPushConstants pcs) where T.Accum : __BuiltinArithmeticType
{
    attentionWarp<T>(wg, t, pcs);
}
