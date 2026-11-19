module matmul_coop;

import helpers.gemm_coop;

struct CoopPushConstants<T>
{
    uint64_t tensor_a;
    uint64_t tensor_b;
    uint64_t tensor_out;
    uint64_t tensor_bias;
    TensorStrides strides_a;
    TensorStrides strides_b;
    TensorStrides strides_out;
    TensorStrides strides_bias;
    uint m_dim;
    uint n_dim;
    uint k_dim;
    uint padding;
    T alpha;
    T beta;
}

struct CoopTile
{
    uint batch_offset_a;
    uint batch_offset_b;
    uint batch_offset_out;
    uint tile_m;
    uint tile_n;
    uint warp_row_base;
    uint warp_col_base;
    uint total_threads;
}

CoopTile makeTile<Acc>(uint3 wg, uint ti, CoopPushConstants<Acc> pcs)
{
    CoopTile tile;
    tile.batch_offset_a = wg.z * pcs.strides_a.batch;
    tile.batch_offset_b = wg.z * pcs.strides_b.batch;
    tile.batch_offset_out = wg.z * pcs.strides_out.batch;

    tile.tile_m = wg.y * (WARPS_M * WARP_FRAGS_M * BLOCK_SIZE);
    tile.tile_n = wg.x * (WARPS_N * WARP_FRAGS_N * BLOCK_SIZE);

    uint warp_id = ti / SUBGROUP_SIZE;
    uint warp_row = warp_id / WARPS_N;
    uint warp_col = warp_id % WARPS_N;
    tile.warp_row_base = warp_row * WARP_FRAGS_M * BLOCK_SIZE;
    tile.warp_col_base = warp_col * WARP_FRAGS_N * BLOCK_SIZE;
    tile.total_threads = WARPS_M * WARPS_N * SUBGROUP_SIZE;
    return tile;
}

// slangc 2026.10.2 crashes on the splat constructor vector<BFloat16, N>(x)
vector<T, VEC> zeroVector<T : ITorchVulkan, let VEC : int>()
{
    vector<T, VEC> v;
    [unroll] for (uint e = 0; e < VEC; ++e) v[e] = T.ext_zero();
    return v;
}

vector<T, VEC> loadVectorA<T : ITorchVulkan & ICoopElement, let VEC : int>(CoopTile tile, CoopPushConstants<T.Accum> pcs, uint i, uint k)
{
    T* tensor_a = (T*)pcs.tensor_a;
    vector<T, VEC> v = zeroVector<T, VEC>();
    if (A_TRANSPOSED == 1)
    {
        uint kk = k + i / (SMEM_ROWS_A / VEC);
        uint m = tile.tile_m + (i % (SMEM_ROWS_A / VEC)) * VEC;
        if (kk >= pcs.k_dim) return v;
        uint base = tile.batch_offset_a + kk * pcs.strides_a.col;
        if (A_VEC_ALIGNED == 1 && (M_ALIGNED == 1 || m + VEC <= pcs.m_dim)) return loadAligned<VEC * sizeof(T)>((vector<T, VEC>*)tensor_a + (base + m) / VEC);
        [unroll] for (uint e = 0; e < VEC; ++e) v[e] = tensor_a[base + min(m + e, pcs.m_dim - 1) * pcs.strides_a.row];
        return v;
    }

    uint m = tile.tile_m + i / (BK / VEC);
    uint kk = k + (i % (BK / VEC)) * VEC;
    if (M_ALIGNED == 0) m = min(m, pcs.m_dim - 1);
    uint base = tile.batch_offset_a + m * pcs.strides_a.row;
    if (A_VEC_ALIGNED == 1 && kk + VEC <= pcs.k_dim) return loadAligned<VEC * sizeof(T)>((vector<T, VEC>*)tensor_a + (base + kk) / VEC);
    [unroll] for (uint e = 0; e < VEC; ++e)
    {
        if (kk + e < pcs.k_dim) v[e] = tensor_a[base + (kk + e) * pcs.strides_a.col];
    }
    return v;
}

vector<T, VEC> loadVectorB<T : ITorchVulkan & ICoopElement, let VEC : int>(CoopTile tile, CoopPushConstants<T.Accum> pcs, uint i, uint k)
{
    T* tensor_b = (T*)pcs.tensor_b;
    vector<T, VEC> v = zeroVector<T, VEC>();
    if (B_TRANSPOSED == 1)
    {
        uint n = tile.tile_n + i / (BK / VEC);
        uint kk = k + (i % (BK / VEC)) * VEC;
        if (N_ALIGNED == 0) n = min(n, pcs.n_dim - 1);
        uint base = tile.batch_offset_b + n * pcs.strides_b.col;
        if (B_VEC_ALIGNED == 1 && kk + VEC <= pcs.k_dim) return loadAligned<VEC * sizeof(T)>((vector<T, VEC>*)tensor_b + (base + kk) / VEC);
        [unroll] for (uint e = 0; e < VEC; ++e)
        {
            if (kk + e < pcs.k_dim) v[e] = tensor_b[base + (kk + e) * pcs.strides_b.row];
        }
        return v;
    }

    uint kk = k + i / (SMEM_COLS_B / VEC);
    uint n = tile.tile_n + (i % (SMEM_COLS_B / VEC)) * VEC;
    if (kk >= pcs.k_dim) return v;
    uint base = tile.batch_offset_b + kk * pcs.strides_b.row;
    if (B_VEC_ALIGNED == 1 && (N_ALIGNED == 1 || n + VEC <= pcs.n_dim)) return loadAligned<VEC * sizeof(T)>((vector<T, VEC>*)tensor_b + (base + n) / VEC);
    [unroll] for (uint e = 0; e < VEC; ++e) v[e] = tensor_b[base + min(n + e, pcs.n_dim - 1) * pcs.strides_b.col];
    return v;
}

void loadTile<T : ITorchVulkan & ICoopElement, let VEC : int>(
    uint ti, CoopTile tile, CoopPushConstants<T.Accum> pcs, uint k, inout vector<T, VEC> regA[LOADS_A], inout vector<T, VEC> regB[LOADS_B])
{
    [unroll] for (uint j = 0; j < LOADS_A; ++j)
    {
        uint i = ti + j * WORKGROUP_SIZE;
        if (i < SMEM_ROWS_A * BK / VEC) regA[j] = loadVectorA<T, VEC>(tile, pcs, i, k);
    }
    [unroll] for (uint j = 0; j < LOADS_B; ++j)
    {
        uint i = ti + j * WORKGROUP_SIZE;
        if (i < BK * SMEM_COLS_B / VEC) regB[j] = loadVectorB<T, VEC>(tile, pcs, i, k);
    }
}

void storeTile<T : ITorchVulkan & ICoopElement, let VEC : int>(
    uint ti, uint stage, vector<T, VEC> regA[LOADS_A], vector<T, VEC> regB[LOADS_B])
{
    [unroll] for (uint j = 0; j < LOADS_A; ++j)
    {
        uint i = ti + j * WORKGROUP_SIZE;
        if (i < SMEM_ROWS_A * BK / VEC)
        {
            uint off = A_TRANSPOSED == 1
                ? (i / (SMEM_ROWS_A / VEC)) * (SMEM_ROWS_A + PAD) + (i % (SMEM_ROWS_A / VEC)) * VEC
                : (i / (BK / VEC)) * (BK + PAD) + (i % (BK / VEC)) * VEC;
            [unroll] for (uint e = 0; e < VEC; ++e) Shared<T>.smemA[stage * stageA() + off + e] = regA[j][e];
        }
    }
    [unroll] for (uint j = 0; j < LOADS_B; ++j)
    {
        uint i = ti + j * WORKGROUP_SIZE;
        if (i < BK * SMEM_COLS_B / VEC)
        {
            uint off = B_TRANSPOSED == 1
                ? (i / (BK / VEC)) * (BK + PAD) + (i % (BK / VEC)) * VEC
                : (i / (SMEM_COLS_B / VEC)) * (SMEM_COLS_B + PAD) + (i % (SMEM_COLS_B / VEC)) * VEC;
            [unroll] for (uint e = 0; e < VEC; ++e) Shared<T>.smemB[stage * stageB() + off + e] = regB[j][e];
        }
    }
}

void accumulateStaged<T : ITorchVulkan & ICoopElement, let VEC : int>(
    uint ti, CoopTile tile, CoopPushConstants<T.Accum> pcs, inout Accumulator<T.Accum> acc[WARP_FRAGS_M * WARP_FRAGS_N])
{
    vector<T, VEC> regA[LOADS_A];
    vector<T, VEC> regB[LOADS_B];

    uint stage = 0;
    if (pcs.k_dim > 0)
    {
        loadTile<T, VEC>(ti, tile, pcs, 0, regA, regB);
        storeTile<T, VEC>(ti, 0, regA, regB);
        GroupMemoryBarrierWithGroupSync();
    }

    for (uint k = 0; k < pcs.k_dim; k += BK)
    {
        bool has_next = k + BK < pcs.k_dim;
        if (has_next) loadTile<T, VEC>(ti, tile, pcs, k + BK, regA, regB);

        uint base_a = stage * stageA();
        uint base_b = stage * stageB();
        for (uint step = 0; step < BK; step += BLOCK_SIZE)
        {
            FragmentA<T> matA[WARP_FRAGS_M];
            [unroll] for (uint mi = 0; mi < WARP_FRAGS_M; ++mi)
            {
                uint row = tile.warp_row_base + mi * BLOCK_SIZE;
                if (A_TRANSPOSED == 1)
                {
                    matA[mi] = matA[mi].Load<linalg::CoopMatMatrixLayout.ColumnMajor>(Shared<T>.smemA, base_a + step * (SMEM_ROWS_A + PAD) + row, SMEM_ROWS_A + PAD);
                }
                else
                {
                    matA[mi] = matA[mi].Load<linalg::CoopMatMatrixLayout.RowMajor>(Shared<T>.smemA, base_a + row * (BK + PAD) + step, BK + PAD);
                }
            }

            FragmentB<T> matB[WARP_FRAGS_N];
            [unroll] for (uint nj = 0; nj < WARP_FRAGS_N; ++nj)
            {
                uint col = tile.warp_col_base + nj * BLOCK_SIZE;
                if (B_TRANSPOSED == 1)
                {
                    matB[nj] = matB[nj].Load<linalg::CoopMatMatrixLayout.ColumnMajor>(Shared<T>.smemB, base_b + col * (BK + PAD) + step, BK + PAD);
                }
                else
                {
                    matB[nj] = matB[nj].Load<linalg::CoopMatMatrixLayout.RowMajor>(Shared<T>.smemB, base_b + step * (SMEM_COLS_B + PAD) + col, SMEM_COLS_B + PAD);
                }
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

        if (has_next) storeTile<T, VEC>(ti, stage ^ 1, regA, regB);
        GroupMemoryBarrierWithGroupSync();
        stage ^= 1;
    }
}

FragmentA<T> loadA<T : ITorchVulkan & ICoopElement>(CoopTile tile, CoopPushConstants<T.Accum> pcs, uint row, uint k)
{
    T* tensor_a = (T*)pcs.tensor_a;
    uint idx = tile.batch_offset_a + row * pcs.strides_a.row + k * pcs.strides_a.col;

    FragmentA<T> m;
    if (A_TRANSPOSED == 1) return m.Load<linalg::CoopMatMatrixLayout.ColumnMajor>(tensor_a, idx, pcs.strides_a.col);
    return m.Load<linalg::CoopMatMatrixLayout.RowMajor>(tensor_a, idx, pcs.strides_a.row);
}

FragmentB<T> loadB<T : ITorchVulkan & ICoopElement>(CoopTile tile, CoopPushConstants<T.Accum> pcs, uint k, uint col)
{
    T* tensor_b = (T*)pcs.tensor_b;
    uint idx = tile.batch_offset_b + k * pcs.strides_b.row + col * pcs.strides_b.col;

    FragmentB<T> m;
    if (B_TRANSPOSED == 1) return m.Load<linalg::CoopMatMatrixLayout.ColumnMajor>(tensor_b, idx, pcs.strides_b.col);
    return m.Load<linalg::CoopMatMatrixLayout.RowMajor>(tensor_b, idx, pcs.strides_b.row);
}

// STEPS slices of K: all loads first, then all multiplies, so the loads overlap
void accumulateSteps<T : ITorchVulkan & ICoopElement, let STEPS : uint>(
    CoopTile tile, CoopPushConstants<T.Accum> pcs, uint a_rows[WARP_FRAGS_M], uint b_cols[WARP_FRAGS_N], uint k,
    inout Accumulator<T.Accum> acc[WARP_FRAGS_M * WARP_FRAGS_N])
{
    FragmentA<T> matA[STEPS][WARP_FRAGS_M];
    [ForceUnroll] for (uint s = 0; s < STEPS; ++s)
    {
        [unroll] for (uint mi = 0; mi < WARP_FRAGS_M; ++mi) matA[s][mi] = loadA<T>(tile, pcs, a_rows[mi], k + s * BLOCK_SIZE);
    }

    FragmentB<T> matB[STEPS][WARP_FRAGS_N];
    [ForceUnroll] for (uint s = 0; s < STEPS; ++s)
    {
        [unroll] for (uint nj = 0; nj < WARP_FRAGS_N; ++nj) matB[s][nj] = loadB<T>(tile, pcs, k + s * BLOCK_SIZE, b_cols[nj]);
    }

    [ForceUnroll] for (uint s = 0; s < STEPS; ++s)
    {
        [unroll] for (uint mi = 0; mi < WARP_FRAGS_M; ++mi)
        {
            [unroll] for (uint nj = 0; nj < WARP_FRAGS_N; ++nj)
            {
                uint flat_idx = mi * WARP_FRAGS_N + nj;
                acc[flat_idx] = linalg::coopMatMulAdd<T.Accum, false>(matA[s][mi], matB[s][nj], acc[flat_idx]);
            }
        }
    }
}

void accumulateDirect<T : ITorchVulkan & ICoopElement>(
    uint ti, CoopTile tile, CoopPushConstants<T.Accum> pcs, inout Accumulator<T.Accum> acc[WARP_FRAGS_M * WARP_FRAGS_N])
{
    T* tensor_a = (T*)pcs.tensor_a;
    T* tensor_b = (T*)pcs.tensor_b;

    uint a_rows[WARP_FRAGS_M];
    [unroll] for (uint mi = 0; mi < WARP_FRAGS_M; ++mi)
    {
        a_rows[mi] = tile.tile_m + tile.warp_row_base + mi * BLOCK_SIZE;
        if (M_ALIGNED == 0) a_rows[mi] = min(a_rows[mi], pcs.m_dim - BLOCK_SIZE);
    }
    uint b_cols[WARP_FRAGS_N];
    [unroll] for (uint nj = 0; nj < WARP_FRAGS_N; ++nj)
    {
        b_cols[nj] = tile.tile_n + tile.warp_col_base + nj * BLOCK_SIZE;
        if (N_ALIGNED == 0) b_cols[nj] = min(b_cols[nj], pcs.n_dim - BLOCK_SIZE);
    }

    uint k_main_end = (pcs.k_dim / BK) * BK;
    uint k = 0;

    for (; k + 8 * BLOCK_SIZE <= k_main_end; k += 8 * BLOCK_SIZE) accumulateSteps<T, 8>(tile, pcs, a_rows, b_cols, k, acc);
    for (; k + 4 * BLOCK_SIZE <= k_main_end; k += 4 * BLOCK_SIZE) accumulateSteps<T, 4>(tile, pcs, a_rows, b_cols, k, acc);
    for (; k + 2 * BLOCK_SIZE <= k_main_end; k += 2 * BLOCK_SIZE) accumulateSteps<T, 2>(tile, pcs, a_rows, b_cols, k, acc);
    for (; k + BLOCK_SIZE <= k_main_end; k += BLOCK_SIZE) accumulateSteps<T, 1>(tile, pcs, a_rows, b_cols, k, acc);

    if (k >= pcs.k_dim) return;

    // the K tail: staged through shared memory and zero padded to a whole BK tile
    uint smem_b_cols = WARPS_N * WARP_FRAGS_N * BLOCK_SIZE;

    for (uint i = ti; i < (WARPS_M * WARP_FRAGS_M * BLOCK_SIZE) * BK; i += tile.total_threads)
    {
        uint r = i / BK;
        uint c = i % BK;
        uint a_row = (M_ALIGNED == 1 ? (tile.tile_m + r) : min(tile.tile_m + r, pcs.m_dim - 1));
        T val = T.ext_zero();
        if ((k + c) < pcs.k_dim)
        {
            uint global_idx = tile.batch_offset_a + a_row * pcs.strides_a.row + (k + c) * pcs.strides_a.col;
            val = tensor_a[global_idx];
        }
        Shared<T>.smemA[r * BK + c] = val;
    }

    for (uint i = ti; i < BK * smem_b_cols; i += tile.total_threads)
    {
        uint r = i / smem_b_cols;
        uint c = i % smem_b_cols;
        uint b_col = (N_ALIGNED == 1 ? (tile.tile_n + c) : min(tile.tile_n + c, pcs.n_dim - 1));
        T val = T.ext_zero();
        if ((k + r) < pcs.k_dim)
        {
            uint global_idx = tile.batch_offset_b + (k + r) * pcs.strides_b.row + b_col * pcs.strides_b.col;
            val = tensor_b[global_idx];
        }
        Shared<T>.smemB[r * smem_b_cols + c] = val;
    }

    GroupMemoryBarrierWithGroupSync();

    for (uint step = 0; step < BK; step += BLOCK_SIZE)
    {
        FragmentA<T> matA[WARP_FRAGS_M];
        [unroll] for (uint mi = 0; mi < WARP_FRAGS_M; ++mi)
        {
            uint off = (tile.warp_row_base + mi * BLOCK_SIZE) * BK + step;
            matA[mi] = matA[mi].Load<linalg::CoopMatMatrixLayout.RowMajor>(Shared<T>.smemA, off, BK);
        }

        FragmentB<T> matB[WARP_FRAGS_N];
        [unroll] for (uint nj = 0; nj < WARP_FRAGS_N; ++nj)
        {
            uint off = step * smem_b_cols + (tile.warp_col_base + nj * BLOCK_SIZE);
            matB[nj] = matB[nj].Load<linalg::CoopMatMatrixLayout.RowMajor>(Shared<T>.smemB, off, smem_b_cols);
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

void finishTile<T : ITorchVulkan & ICoopElement>(
    uint3 wg, uint ti, CoopTile tile, CoopPushConstants<T.Accum> pcs, inout Accumulator<T.Accum> acc[WARP_FRAGS_M * WARP_FRAGS_N])
{
    typealias A = T.Accum;
    T* tensor_out = (T*)pcs.tensor_out;
    T* tensor_bias = (T*)pcs.tensor_bias;
    uint lane = ti % SUBGROUP_SIZE;
    uint scratch = (ti / SUBGROUP_SIZE) * BLOCK_SIZE * BLOCK_SIZE;
    bool direct = tile.tile_m + SMEM_ROWS_A <= pcs.m_dim && tile.tile_n + SMEM_COLS_B <= pcs.n_dim
        && OUT_ALIGNED == 1 && HAS_BIAS == 0 && ACTIVATION == 0;

    // no continue or early exit: Metal spills the accumulators when the control flow isn't structured
    [unroll] for (uint mi = 0; mi < WARP_FRAGS_M; ++mi)
    {
        [unroll] for (uint nj = 0; nj < WARP_FRAGS_N; ++nj)
        {
            uint row = tile.tile_m + tile.warp_row_base + mi * BLOCK_SIZE;
            uint col = tile.tile_n + tile.warp_col_base + nj * BLOCK_SIZE;
            Accumulator<A> value = acc[mi * WARP_FRAGS_N + nj];
            if (HAS_ALPHA == 1) value = value * pcs.alpha;

            if (direct)
            {
                uint out_idx = tile.batch_offset_out + row * pcs.strides_out.row + col * pcs.strides_out.col;
                Output<T>(value).Store<linalg::CoopMatMatrixLayout.RowMajor>(tensor_out, out_idx, pcs.strides_out.row);
            }
            else
            {
                value.Store<linalg::CoopMatMatrixLayout.RowMajor>(EpilogueShared<A>.smem, scratch, BLOCK_SIZE);
                GroupMemoryBarrierWithWaveSync();
                for (uint i = lane; i < BLOCK_SIZE * BLOCK_SIZE; i += SUBGROUP_SIZE)
                {
                    uint r = row + i / BLOCK_SIZE;
                    uint c = col + i % BLOCK_SIZE;
                    if (r < pcs.m_dim && c < pcs.n_dim)
                    {
                        A v = EpilogueShared<A>.smem[scratch + i];
                        if (HAS_BIAS == 1)
                        {
                            uint bias_idx = wg.z * pcs.strides_bias.batch + r * pcs.strides_bias.row + c * pcs.strides_bias.col;
                            v = v + pcs.beta * tensor_bias[bias_idx].toAccum();
                        }
                        if (ACTIVATION == 1) v = A.ext_max(v, A.ext_zero());
                        if (ACTIVATION == 2) v = A.ext_gelu(v);
                        tensor_out[tile.batch_offset_out + r * pcs.strides_out.row + c * pcs.strides_out.col] = T.fromAccum(v);
                    }
                }
                GroupMemoryBarrierWithWaveSync();
            }
        }
    }
}

void coopMatmulStaged<T : ITorchVulkan & ICoopElement, let VEC : int>(
    uint3 wg, uint ti, CoopPushConstants<T.Accum> pcs)
{
    CoopTile tile = makeTile<T.Accum>(wg, ti, pcs);

    Accumulator<T.Accum> acc[WARP_FRAGS_M * WARP_FRAGS_N];
    [unroll] for (uint i = 0; i < WARP_FRAGS_M * WARP_FRAGS_N; ++i) acc[i] = Accumulator<T.Accum>(T.Accum.ext_zero());
    GroupMemoryBarrierWithGroupSync();

    accumulateStaged<T, VEC>(ti, tile, pcs, acc);
    finishTile<T>(wg, ti, tile, pcs, acc);
}

void coopMatmulDirect<T : ITorchVulkan & ICoopElement>(
    uint3 wg, uint ti, CoopPushConstants<T.Accum> pcs)
{
    CoopTile tile = makeTile<T.Accum>(wg, ti, pcs);

    Accumulator<T.Accum> acc[WARP_FRAGS_M * WARP_FRAGS_N];
    [unroll] for (uint i = 0; i < WARP_FRAGS_M * WARP_FRAGS_N; ++i) acc[i] = Accumulator<T.Accum>(T.Accum.ext_zero());
    GroupMemoryBarrierWithGroupSync();

    accumulateDirect<T>(ti, tile, pcs, acc);
    finishTile<T>(wg, ti, tile, pcs, acc);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void matmul_coop_main<T : ITorchVulkan & ICoopElement, let VEC : int>(
    uint3 wg : SV_GroupID,
    uint ti : SV_GroupIndex,
    uniform CoopPushConstants<T.Accum> pcs)
{
    if (USE_SHARED_MEMORY == 1) coopMatmulStaged<T, VEC>(wg, ti, pcs);
    else coopMatmulDirect<T>(wg, ti, pcs);
}
