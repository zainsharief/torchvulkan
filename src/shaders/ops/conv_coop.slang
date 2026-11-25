module conv_coop;

import helpers.gemm_coop;
import helpers.conv;

static const uint GATHER_COLUMNS = 4;

struct ConvTile
{
    uint batch;
    uint group;
    uint row0;
    uint col0;
    uint warp_row;
    uint warp_col;
    uint M;
    uint N;
    uint K;
}

// clamped indices: lavapipe performs loads in untaken branches
void gatherSlice<T : ITorchVulkan & ICoopElement, A : IMatmulLoader<T.Accum>, B : IColumnGather<T>>(
    uint ti, ConvTile tile, uint k, A a, B b, B.Column columns[GATHER_COLUMNS], inout T regA[LOADS_A], inout T regB[LOADS_B * GATHER_COLUMNS])
{
    [unroll] for (uint j = 0; j < LOADS_A; ++j)
    {
        uint i = ti + j * WORKGROUP_SIZE;
        uint row = tile.row0 + i / BK;
        uint kk = k + i % BK;
        T loaded = T.fromAccumExact(a.load(tile.batch, min(row, tile.M - 1), min(kk, tile.K - 1)));
        T v = T.ext_zero();
        if (row < tile.M && kk < tile.K) v = loaded;
        regA[j] = v;
    }
    [unroll] for (uint j = 0; j < LOADS_B; ++j)
    {
        uint i = ti + j * WORKGROUP_SIZE;
        uint kk = k + i / (SMEM_COLS_B / GATHER_COLUMNS);
        uint col = tile.col0 + (i % (SMEM_COLS_B / GATHER_COLUMNS)) * GATHER_COLUMNS;
        B.Row row = b.row(min(kk, tile.K - 1));
        [unroll] for (uint q = 0; q < GATHER_COLUMNS; ++q)
        {
            T loaded = b.load(columns[q], row);
            T v = T.ext_zero();
            if (kk < tile.K && col + q < tile.N) v = loaded;
            regB[j * GATHER_COLUMNS + q] = v;
        }
    }
}

void storeSlice<T : ITorchVulkan & ICoopElement>(uint ti, uint stage, T regA[LOADS_A], T regB[LOADS_B * GATHER_COLUMNS])
{
    [unroll] for (uint j = 0; j < LOADS_A; ++j)
    {
        uint i = ti + j * WORKGROUP_SIZE;
        if (i < SMEM_ROWS_A * BK) Shared<T>.smemA[stage * stageA() + (i / BK) * (BK + PAD) + i % BK] = regA[j];
    }
    [unroll] for (uint j = 0; j < LOADS_B; ++j)
    {
        uint i = ti + j * WORKGROUP_SIZE;
        uint groups = SMEM_COLS_B / GATHER_COLUMNS;
        if (i < BK * groups)
        {
            uint at = stage * stageB() + (i / groups) * (SMEM_COLS_B + PAD) + (i % groups) * GATHER_COLUMNS;
            [unroll] for (uint q = 0; q < GATHER_COLUMNS; ++q) Shared<T>.smemB[at + q] = regB[j * GATHER_COLUMNS + q];
        }
    }
}

void multiplySlice<T : ITorchVulkan & ICoopElement>(ConvTile tile, uint stage, inout Accumulator<T.Accum> acc[WARP_FRAGS_M * WARP_FRAGS_N])
{
    uint base_a = stage * stageA();
    uint base_b = stage * stageB();
    for (uint step = 0; step < BK; step += BLOCK_SIZE)
    {
        FragmentA<T> matA[WARP_FRAGS_M];
        [unroll] for (uint mi = 0; mi < WARP_FRAGS_M; ++mi)
        {
            uint row = tile.warp_row + mi * BLOCK_SIZE;
            matA[mi] = matA[mi].Load<linalg::CoopMatMatrixLayout.RowMajor>(Shared<T>.smemA, base_a + row * (BK + PAD) + step, BK + PAD);
        }

        FragmentB<T> matB[WARP_FRAGS_N];
        [unroll] for (uint nj = 0; nj < WARP_FRAGS_N; ++nj)
        {
            uint col = tile.warp_col + nj * BLOCK_SIZE;
            if (B_TRANSPOSED == 1) matB[nj] = matB[nj].Load<linalg::CoopMatMatrixLayout.ColumnMajor>(Shared<T>.smemB, base_b + col * (BK + PAD) + step, BK + PAD);
            else matB[nj] = matB[nj].Load<linalg::CoopMatMatrixLayout.RowMajor>(Shared<T>.smemB, base_b + step * (SMEM_COLS_B + PAD) + col, SMEM_COLS_B + PAD);
        }

        [unroll] for (uint mi = 0; mi < WARP_FRAGS_M; ++mi)
        {
            [unroll] for (uint nj = 0; nj < WARP_FRAGS_N; ++nj)
            {
                uint flat_idx = mi * WARP_FRAGS_N + nj;
                acc[flat_idx] = linalg::coopMatMulAdd<T.Accum, false>(matA[mi], matB[nj], acc[flat_idx]);
            }
        }
    }
}

void finishTile<T : ITorchVulkan & ICoopElement, S : IMatmulStore<T.Accum>>(
    uint ti, ConvTile tile, inout Accumulator<T.Accum> acc[WARP_FRAGS_M * WARP_FRAGS_N], S out, uint64_t bias)
{
    typealias A = T.Accum;
    uint lane = ti % SUBGROUP_SIZE;
    uint scratch = (ti / SUBGROUP_SIZE) * BLOCK_SIZE * BLOCK_SIZE;

    // structured control flow only, as in matmul_coop's epilogue; blocks past the edge run the barriers too, as every
    // subgroup must there
    [unroll] for (uint mi = 0; mi < WARP_FRAGS_M; ++mi)
    {
        [unroll] for (uint nj = 0; nj < WARP_FRAGS_N; ++nj)
        {
            uint row = tile.row0 + tile.warp_row + mi * BLOCK_SIZE;
            uint col = tile.col0 + tile.warp_col + nj * BLOCK_SIZE;
            acc[mi * WARP_FRAGS_N + nj].Store<linalg::CoopMatMatrixLayout.RowMajor>(EpilogueShared<A>.smem, scratch, BLOCK_SIZE);
            GroupMemoryBarrierWithWaveSync();
            for (uint i = lane; i < BLOCK_SIZE * BLOCK_SIZE; i += SUBGROUP_SIZE)
            {
                uint r = row + i / BLOCK_SIZE;
                uint c = col + i % BLOCK_SIZE;
                if (r < tile.M && c < tile.N)
                {
                    A v = EpilogueShared<A>.smem[scratch + i];
                    if (HAS_BIAS == 1) v = v + ((T*)bias)[tile.group * tile.M + r].toAccum();
                    out.store(tile.batch, r, c, v);
                }
            }
            GroupMemoryBarrierWithWaveSync();
        }
    }
}

void convGemm<T : ITorchVulkan & ICoopElement, A : IMatmulLoader<T.Accum>, B : IColumnGather<T>, S : IMatmulStore<T.Accum>>(
    uint3 wg, uint ti, A a, B b, S out, uint64_t bias, uint group, uint M, uint N, uint K)
{
    // a phase with no taps still writes zeros and the bias
    if (N == 0) return;
    ConvTile tile;
    tile.batch = wg.z;
    tile.group = group;
    tile.row0 = wg.y * SMEM_ROWS_A;
    tile.col0 = wg.x * SMEM_COLS_B;
    uint warp_id = ti / SUBGROUP_SIZE;
    tile.warp_row = (warp_id / WARPS_N) * WARP_FRAGS_M * BLOCK_SIZE;
    tile.warp_col = (warp_id % WARPS_N) * WARP_FRAGS_N * BLOCK_SIZE;
    tile.M = M;
    tile.N = N;
    tile.K = K;

    Accumulator<T.Accum> acc[WARP_FRAGS_M * WARP_FRAGS_N];
    [unroll] for (uint i = 0; i < WARP_FRAGS_M * WARP_FRAGS_N; ++i) acc[i] = Accumulator<T.Accum>(T.Accum.ext_zero());

    B.Column columns[GATHER_COLUMNS];
    uint col = tile.col0 + (ti % (SMEM_COLS_B / GATHER_COLUMNS)) * GATHER_COLUMNS;
    [unroll] for (uint q = 0; q < GATHER_COLUMNS; ++q) columns[q] = b.column(min(col + q, N - 1));

    if (K > 0)
    {
        T regA[LOADS_A];
        T regB[LOADS_B * GATHER_COLUMNS];
        gatherSlice<T, A, B>(ti, tile, 0, a, b, columns, regA, regB);
        storeSlice<T>(ti, 0, regA, regB);
        GroupMemoryBarrierWithGroupSync();

        uint stage = 0;
        for (uint k = 0; k < K; k += BK)
        {
            bool has_next = k + BK < K;
            if (has_next) gatherSlice<T, A, B>(ti, tile, k + BK, a, b, columns, regA, regB);
            multiplySlice<T>(tile, stage, acc);
            if (has_next) storeSlice<T>(ti, stage ^ 1, regA, regB);
            GroupMemoryBarrierWithGroupSync();
            stage ^= 1;
        }
    }

    finishTile<T, S>(ti, tile, acc, out, bias);
}

void gatherSliceK<T : ITorchVulkan & ICoopElement, A : IColumnGather<T>, B : IColumnGather<T>>(
    uint ti, ConvTile tile, uint k, A a, B b, A.Column rows[LOADS_A], B.Column cols[LOADS_B], inout T regA[LOADS_A], inout T regB[LOADS_B])
{
    uint kk = k + ti % BK;
    A.Row ak = a.row(min(kk, tile.K - 1));
    B.Row bk = b.row(min(kk, tile.K - 1));
    [unroll] for (uint j = 0; j < LOADS_A; ++j)
    {
        T loaded = a.load(rows[j], ak);
        regA[j] = tile.row0 + (ti + j * WORKGROUP_SIZE) / BK < tile.M && kk < tile.K ? loaded : T.ext_zero();
    }
    [unroll] for (uint j = 0; j < LOADS_B; ++j)
    {
        T loaded = b.load(cols[j], bk);
        regB[j] = tile.col0 + (ti + j * WORKGROUP_SIZE) / BK < tile.N && kk < tile.K ? loaded : T.ext_zero();
    }
}

void storeSliceK<T : ITorchVulkan & ICoopElement>(uint ti, uint stage, T regA[LOADS_A], T regB[LOADS_B])
{
    [unroll] for (uint j = 0; j < LOADS_A; ++j)
    {
        uint i = ti + j * WORKGROUP_SIZE;
        if (i < SMEM_ROWS_A * BK) Shared<T>.smemA[stage * stageA() + (i / BK) * (BK + PAD) + i % BK] = regA[j];
    }
    [unroll] for (uint j = 0; j < LOADS_B; ++j)
    {
        uint i = ti + j * WORKGROUP_SIZE;
        if (i < SMEM_COLS_B * BK) Shared<T>.smemB[stage * stageB() + (i / BK) * (BK + PAD) + i % BK] = regB[j];
    }
}

void convGemmK<T : ITorchVulkan & ICoopElement, A : IColumnGather<T>, B : IColumnGather<T>, S : IMatmulStore<T.Accum>>(
    uint3 wg, uint ti, A a, B b, S out, uint M, uint N, uint K)
{
    ConvTile tile;
    tile.batch = wg.z;
    tile.group = 0;
    tile.row0 = wg.y * SMEM_ROWS_A;
    tile.col0 = wg.x * SMEM_COLS_B;
    uint warp_id = ti / SUBGROUP_SIZE;
    tile.warp_row = (warp_id / WARPS_N) * WARP_FRAGS_M * BLOCK_SIZE;
    tile.warp_col = (warp_id % WARPS_N) * WARP_FRAGS_N * BLOCK_SIZE;
    tile.M = M;
    tile.N = N;
    tile.K = K;

    Accumulator<T.Accum> acc[WARP_FRAGS_M * WARP_FRAGS_N];
    [unroll] for (uint i = 0; i < WARP_FRAGS_M * WARP_FRAGS_N; ++i) acc[i] = Accumulator<T.Accum>(T.Accum.ext_zero());

    A.Column rows[LOADS_A];
    [unroll] for (uint j = 0; j < LOADS_A; ++j) rows[j] = a.column(min(tile.row0 + (ti + j * WORKGROUP_SIZE) / BK, M - 1));
    B.Column cols[LOADS_B];
    [unroll] for (uint j = 0; j < LOADS_B; ++j) cols[j] = b.column(min(tile.col0 + (ti + j * WORKGROUP_SIZE) / BK, N - 1));

    T regA[LOADS_A];
    T regB[LOADS_B];
    gatherSliceK<T, A, B>(ti, tile, 0, a, b, rows, cols, regA, regB);
    storeSliceK<T>(ti, 0, regA, regB);
    GroupMemoryBarrierWithGroupSync();

    uint stage = 0;
    for (uint k = 0; k < K; k += BK)
    {
        bool has_next = k + BK < K;
        if (has_next) gatherSliceK<T, A, B>(ti, tile, k + BK, a, b, rows, cols, regA, regB);
        multiplySlice<T>(tile, stage, acc);
        if (has_next) storeSliceK<T>(ti, stage ^ 1, regA, regB);
        GroupMemoryBarrierWithGroupSync();
        stage ^= 1;
    }

    finishTile<T, S>(ti, tile, acc, out, 0);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void conv_coop_main<T : ITorchVulkan & ICoopElement, let VEC : int>(
    uint3 li : SV_GroupThreadID,
    uint3 wg : SV_GroupID,
    uniform ConvPushConstants pcs)
{
    uint ti = li.x;
    uint threads = WORKGROUP_SIZE;

    uint* source = (uint*)(*(Metadata*)pcs.metadata).params;
    for (uint i = ti; i < PARAMS; i += threads) params[i] = source[i];
    GroupMemoryBarrierWithGroupSync();
    Geometry g = Geometry();

    // z is split * groups + group for grad weight, phase * groups + group for a sub-pixel grad input
    uint outer = MODE == 2 || (MODE == 1 && SUBPIXEL == 1) ? g.groupsDiv.div(wg.z) : 0;
    uint group = wg.z - outer * g.groups;
    uint count = pcs.n * g.outputs.d;

    if (MODE == 0)
    {
        WeightRows<T> a = { pcs.weight, g };
        ForwardColumns<T> b = { pcs.x, g, group };
        GroupedStore<T> out = { pcs.y, g.groups, g.cy, g.outputs };
        uint M = g.cy, N = count, K = g.cx * TAPS;
        convGemm<T, WeightRows<T>, ForwardColumns<T>, GroupedStore<T>>(wg, ti, a, b, out, pcs.bias, group, M, N, K);
    }
    else if (MODE == 1 && SUBPIXEL == 1)
    {
        Phase phase = loadPhase(outer, g, ti);
        WeightPhaseColumns<T> a = { pcs.weight, g, group };
        PhaseColumns<T> b = { pcs.y, g, phase, group };
        PhaseStore<T> out = { pcs.x, g, phase, group };
        uint M = g.cx, N = pcs.n * phase.positions, K = g.cy * phase.taps;
        convGemm<T, WeightPhaseColumns<T>, PhaseColumns<T>, PhaseStore<T>>(wg, ti, a, b, out, pcs.bias, group, M, N, K);
    }
    else if (MODE == 1)
    {
        WeightColumns<T> a = { pcs.weight, g };
        GradInputColumns<T> b = { pcs.y, g, group };
        GroupedStore<T> out = { pcs.x, g.groups, g.cx, g.inputs };
        uint M = g.cx, N = pcs.n * g.inputs.d, K = g.cy * TAPS;
        convGemm<T, WeightColumns<T>, GradInputColumns<T>, GroupedStore<T>>(wg, ti, a, b, out, pcs.bias, group, M, N, K);
    }
    else
    {
        uint first = outer * pcs.chunk;
        OutputRows<T> a = { pcs.y, g, group, first, count };
        GradWeightColumns<T> b = { pcs.x, g, group, first, count };
        BlockStore<T> out = { pcs.weight, g.cy, g.cx * TAPS };
        uint M = g.cy, N = g.cx * TAPS, K = pcs.chunk;
        convGemmK<T, OutputRows<T>, GradWeightColumns<T>, BlockStore<T>>(wg, ti, a, b, out, M, N, K);
    }
}
