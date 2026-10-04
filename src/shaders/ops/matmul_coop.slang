module matmul_coop;

__exported import core.dtype;

[[vk::constant_id(0)]] public const uint WORKGROUP_SIZE = 128;
[[vk::constant_id(1)]] public const uint SUBGROUP_SIZE = 32;
[[vk::constant_id(2)]] public const uint WARPS_M = 2;
[[vk::constant_id(3)]] public const uint WARPS_N = 2;
[[vk::constant_id(4)]] public const uint WARP_FRAGS_M = 2;
[[vk::constant_id(5)]] public const uint WARP_FRAGS_N = 2;
[[vk::constant_id(6)]] public const uint BK = 16;

[[vk::constant_id(7)]] public const uint HAS_ALPHA = 0;
[[vk::constant_id(8)]] public const uint HAS_BETA = 0;
[[vk::constant_id(9)]] public const uint HAS_BIAS = 0;

[[vk::constant_id(10)]] public const uint A_TRANSPOSED = 0;
[[vk::constant_id(11)]] public const uint B_TRANSPOSED = 0;
[[vk::constant_id(12)]] public const uint OUT_TRANSPOSED = 0;
[[vk::constant_id(13)]] public const uint BIAS_TRANSPOSED = 0;

[[vk::constant_id(14)]] public const uint M_ALIGNED = 0;
[[vk::constant_id(15)]] public const uint N_ALIGNED = 0;

[[vk::constant_id(16)]] public const uint A_VEC_ALIGNED = 0;
[[vk::constant_id(17)]] public const uint B_VEC_ALIGNED = 0;

[[vk::constant_id(18)]] public const uint BLOCK_SIZE = 16;

[[vk::constant_id(19)]] public const uint USE_SHARED_MEMORY = 1;

// padding per shared-memory row, so consecutive rows fall in different banks
static const uint PAD = 4;

public struct TensorStrides
{
    public uint batch;
    public uint row;
    public uint col;
    public uint padding;
}

public struct CoopPushConstants<T>
{
    public uint64_t tensor_a;
    public uint64_t tensor_b;
    public uint64_t tensor_out;
    public uint64_t tensor_bias;
    public TensorStrides strides_a;
    public TensorStrides strides_b;
    public TensorStrides strides_out;
    public TensorStrides strides_bias;
    public uint m_dim;
    public uint n_dim;
    public uint k_dim;
    public uint padding;
    public T alpha;
    public T beta;
}

static const uint SMEM_ROWS_A = WARPS_M * WARP_FRAGS_M * BLOCK_SIZE;
static const uint SMEM_COLS_B = WARPS_N * WARP_FRAGS_N * BLOCK_SIZE;

struct Shared<T>
{
    static groupshared T smemA[USE_SHARED_MEMORY == 1 ? 2 * SMEM_ROWS_A * (BK + PAD) : SMEM_ROWS_A * BK];
    static groupshared T smemB[USE_SHARED_MEMORY == 1 ? 2 * BK * (SMEM_COLS_B + PAD) : BK * SMEM_COLS_B];
}

typealias Accumulator<U : __BuiltinArithmeticType, let BLOCK : uint> =
    linalg::CoopMat<U, MemoryScope.Subgroup, BLOCK, BLOCK, linalg::CoopMatMatrixUse.MatrixAccumulator>;
typealias Output<T : ICoopElement, let BLOCK : uint> =
    linalg::CoopMat<T, MemoryScope.Subgroup, BLOCK, BLOCK, linalg::CoopMatMatrixUse.MatrixAccumulator>;
typealias FragmentA<T : ICoopElement, let BLOCK : uint> =
    linalg::CoopMat<T, MemoryScope.Subgroup, BLOCK, BLOCK, linalg::CoopMatMatrixUse.MatrixA>;
typealias FragmentB<T : ICoopElement, let BLOCK : uint> =
    linalg::CoopMat<T, MemoryScope.Subgroup, BLOCK, BLOCK, linalg::CoopMatMatrixUse.MatrixB>;

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

CoopTile makeTile<Acc, let BLOCK : uint>(uint3 wg, uint ti, CoopPushConstants<Acc> pcs)
{
    CoopTile tile;
    tile.batch_offset_a = wg.z * pcs.strides_a.batch;
    tile.batch_offset_b = wg.z * pcs.strides_b.batch;
    tile.batch_offset_out = wg.z * pcs.strides_out.batch;

    tile.tile_m = wg.y * (WARPS_M * WARP_FRAGS_M * BLOCK);
    tile.tile_n = wg.x * (WARPS_N * WARP_FRAGS_N * BLOCK);

    uint warp_id = ti / SUBGROUP_SIZE;
    uint warp_row = warp_id / WARPS_N;
    uint warp_col = warp_id % WARPS_N;
    tile.warp_row_base = warp_row * WARP_FRAGS_M * BLOCK;
    tile.warp_col_base = warp_col * WARP_FRAGS_N * BLOCK;
    tile.total_threads = WARPS_M * WARPS_N * SUBGROUP_SIZE;
    return tile;
}

// stages the BK-deep slice of A and B starting at k into shared-memory buffer `stage`
void stageTile<T : ITorchVulkan & ICoopElement, let VEC : int, let BLOCK : uint>(
    uint ti, CoopTile tile, CoopPushConstants<T.Accum> pcs, uint k, uint stage)
{
    T* tensor_a = (T*)pcs.tensor_a;
    T* tensor_b = (T*)pcs.tensor_b;

    uint smem_a_rows = WARPS_M * WARP_FRAGS_M * BLOCK;
    uint smem_b_cols = WARPS_N * WARP_FRAGS_N * BLOCK;
    uint smem_a_offset = stage * smem_a_rows * (BK + PAD);
    uint smem_b_offset = stage * BK * (smem_b_cols + PAD);

    uint total_vec_threads_a = (smem_a_rows * BK) / VEC;
    uint total_vec_threads_b = (BK * smem_b_cols) / VEC;

    if (A_TRANSPOSED == 1)
    {
        for (uint i = ti; i < total_vec_threads_a; i += tile.total_threads)
        {
            [unroll] for (uint v = 0; v < VEC; ++v)
            {
                uint flat_smem_idx = i * VEC + v;
                uint r = flat_smem_idx % smem_a_rows;
                uint c = flat_smem_idx / smem_a_rows;
                uint a_row = (M_ALIGNED == 1 ? (tile.tile_m + r) : min(tile.tile_m + r, pcs.m_dim - 1));
                T val = T.ext_zero();
                if ((k + c) < pcs.k_dim)
                {
                    uint global_idx = tile.batch_offset_a + a_row * pcs.strides_a.row + (k + c) * pcs.strides_a.col;
                    val = tensor_a[global_idx];
                }
                Shared<T>.smemA[smem_a_offset + r * (BK + PAD) + c] = val;
            }
        }
    }
    else
    {
        for (uint i = ti; i < total_vec_threads_a; i += tile.total_threads)
        {
            uint flat_smem_idx = i * VEC;
            uint r = flat_smem_idx / BK;
            uint c = flat_smem_idx % BK;
            uint a_row = (M_ALIGNED == 1 ? (tile.tile_m + r) : min(tile.tile_m + r, pcs.m_dim - 1));
            uint padded_smem_idx = r * (BK + PAD) + c;

            if (A_VEC_ALIGNED == 1 && (k + c + VEC) <= pcs.k_dim)
            {
                uint global_idx = tile.batch_offset_a + a_row * pcs.strides_a.row + (k + c);
                [unroll] for (uint v = 0; v < VEC; ++v) Shared<T>.smemA[smem_a_offset + padded_smem_idx + v] = tensor_a[global_idx + v];
            }
            else
            {
                [unroll] for (uint v = 0; v < VEC; ++v)
                {
                    uint k_idx = k + c + v;
                    T val = T.ext_zero();
                    if (k_idx < pcs.k_dim)
                    {
                        uint global_idx = tile.batch_offset_a + a_row * pcs.strides_a.row + k_idx;
                        val = tensor_a[global_idx];
                    }
                    Shared<T>.smemA[smem_a_offset + padded_smem_idx + v] = val;
                }
            }
        }
    }

    if (B_TRANSPOSED == 1)
    {
        for (uint i = ti; i < total_vec_threads_b; i += tile.total_threads)
        {
            [unroll] for (uint v = 0; v < VEC; ++v)
            {
                uint flat_smem_idx = i * VEC + v;
                uint r = flat_smem_idx % BK;
                uint c = flat_smem_idx / BK;
                uint b_col = (N_ALIGNED == 1 ? (tile.tile_n + c) : min(tile.tile_n + c, pcs.n_dim - 1));
                T val = T.ext_zero();
                if ((k + r) < pcs.k_dim)
                {
                    uint global_idx = tile.batch_offset_b + (k + r) * pcs.strides_b.row + b_col * pcs.strides_b.col;
                    val = tensor_b[global_idx];
                }
                Shared<T>.smemB[smem_b_offset + r * (smem_b_cols + PAD) + c] = val;
            }
        }
    }
    else
    {
        for (uint i = ti; i < total_vec_threads_b; i += tile.total_threads)
        {
            uint flat_smem_idx = i * VEC;
            uint r = flat_smem_idx / smem_b_cols;
            uint c = flat_smem_idx % smem_b_cols;
            uint padded_smem_idx = r * (smem_b_cols + PAD) + c;

            if ((k + r) >= pcs.k_dim)
            {
                [unroll] for (uint v = 0; v < VEC; ++v) Shared<T>.smemB[smem_b_offset + padded_smem_idx + v] = T.ext_zero();
            }
            else if (B_VEC_ALIGNED == 1 && (tile.tile_n + c + VEC) <= pcs.n_dim)
            {
                uint global_idx = tile.batch_offset_b + (k + r) * pcs.strides_b.row + (tile.tile_n + c) * pcs.strides_b.col;
                [unroll] for (uint v = 0; v < VEC; ++v) Shared<T>.smemB[smem_b_offset + padded_smem_idx + v] = tensor_b[global_idx + v];
            }
            else
            {
                [unroll] for (uint v = 0; v < VEC; ++v)
                {
                    uint b_col = (N_ALIGNED == 1 ? (tile.tile_n + c + v) : min(tile.tile_n + c + v, pcs.n_dim - 1));
                    uint global_idx = tile.batch_offset_b + (k + r) * pcs.strides_b.row + b_col * pcs.strides_b.col;
                    Shared<T>.smemB[smem_b_offset + padded_smem_idx + v] = tensor_b[global_idx];
                }
            }
        }
    }
}

void accumulateStaged<T : ITorchVulkan & ICoopElement, let VEC : int, let BLOCK : uint>(
    uint ti, CoopTile tile, CoopPushConstants<T.Accum> pcs, inout Accumulator<T.Accum, BLOCK> acc[WARP_FRAGS_M * WARP_FRAGS_N])
{
    uint smem_a_rows = WARPS_M * WARP_FRAGS_M * BLOCK;
    uint smem_b_cols = WARPS_N * WARP_FRAGS_N * BLOCK;
    uint smem_a_size = smem_a_rows * (BK + PAD);
    uint smem_b_size = BK * (smem_b_cols + PAD);

    uint stage = 0;
    if (pcs.k_dim > 0)
    {
        stageTile<T, VEC, BLOCK>(ti, tile, pcs, 0, 0);
        GroupMemoryBarrierWithGroupSync();
    }

    for (uint k = 0; k < pcs.k_dim; k += BK)
    {
        uint next_k = k + BK;
        uint next_stage = stage ^ 1;

        // load the next tile into the other buffer while this one multiplies
        if (next_k < pcs.k_dim) stageTile<T, VEC, BLOCK>(ti, tile, pcs, next_k, next_stage);

        uint current_smem_a_offset = stage * smem_a_size;
        uint current_smem_b_offset = stage * smem_b_size;

        for (uint step = 0; step < BK; step += BLOCK)
        {
            FragmentA<T, BLOCK> matA[WARP_FRAGS_M];
            [unroll] for (uint mi = 0; mi < WARP_FRAGS_M; ++mi)
            {
                uint off = current_smem_a_offset + (tile.warp_row_base + mi * BLOCK) * (BK + PAD) + step;
                matA[mi] = matA[mi].Load<linalg::CoopMatMatrixLayout.RowMajor>(Shared<T>.smemA, off, BK + PAD);
            }

            FragmentB<T, BLOCK> matB[WARP_FRAGS_N];
            [unroll] for (uint nj = 0; nj < WARP_FRAGS_N; ++nj)
            {
                uint off = current_smem_b_offset + step * (smem_b_cols + PAD) + (tile.warp_col_base + nj * BLOCK);
                matB[nj] = matB[nj].Load<linalg::CoopMatMatrixLayout.RowMajor>(Shared<T>.smemB, off, smem_b_cols + PAD);
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

        GroupMemoryBarrierWithGroupSync();
        stage = next_stage;
    }
}

FragmentA<T, BLOCK> loadA<T : ITorchVulkan & ICoopElement, let BLOCK : uint>(CoopTile tile, CoopPushConstants<T.Accum> pcs, uint row, uint k)
{
    T* tensor_a = (T*)pcs.tensor_a;
    uint idx = tile.batch_offset_a + row * pcs.strides_a.row + k * pcs.strides_a.col;

    FragmentA<T, BLOCK> m;
    if (A_TRANSPOSED == 1) return m.Load<linalg::CoopMatMatrixLayout.ColumnMajor>(tensor_a, idx, pcs.strides_a.col);
    return m.Load<linalg::CoopMatMatrixLayout.RowMajor>(tensor_a, idx, pcs.strides_a.row);
}

FragmentB<T, BLOCK> loadB<T : ITorchVulkan & ICoopElement, let BLOCK : uint>(CoopTile tile, CoopPushConstants<T.Accum> pcs, uint k, uint col)
{
    T* tensor_b = (T*)pcs.tensor_b;
    uint idx = tile.batch_offset_b + k * pcs.strides_b.row + col * pcs.strides_b.col;

    FragmentB<T, BLOCK> m;
    if (B_TRANSPOSED == 1) return m.Load<linalg::CoopMatMatrixLayout.ColumnMajor>(tensor_b, idx, pcs.strides_b.col);
    return m.Load<linalg::CoopMatMatrixLayout.RowMajor>(tensor_b, idx, pcs.strides_b.row);
}

// STEPS slices of K: all loads first, then all multiplies, so the loads overlap
void accumulateSteps<T : ITorchVulkan & ICoopElement, let BLOCK : uint, let STEPS : uint>(
    CoopTile tile, CoopPushConstants<T.Accum> pcs, uint a_rows[WARP_FRAGS_M], uint b_cols[WARP_FRAGS_N], uint k,
    inout Accumulator<T.Accum, BLOCK> acc[WARP_FRAGS_M * WARP_FRAGS_N])
{
    FragmentA<T, BLOCK> matA[STEPS][WARP_FRAGS_M];
    [ForceUnroll] for (uint s = 0; s < STEPS; ++s)
    {
        [unroll] for (uint mi = 0; mi < WARP_FRAGS_M; ++mi) matA[s][mi] = loadA<T, BLOCK>(tile, pcs, a_rows[mi], k + s * BLOCK);
    }

    FragmentB<T, BLOCK> matB[STEPS][WARP_FRAGS_N];
    [ForceUnroll] for (uint s = 0; s < STEPS; ++s)
    {
        [unroll] for (uint nj = 0; nj < WARP_FRAGS_N; ++nj) matB[s][nj] = loadB<T, BLOCK>(tile, pcs, k + s * BLOCK, b_cols[nj]);
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

void accumulateDirect<T : ITorchVulkan & ICoopElement, let BLOCK : uint>(
    uint ti, CoopTile tile, CoopPushConstants<T.Accum> pcs, inout Accumulator<T.Accum, BLOCK> acc[WARP_FRAGS_M * WARP_FRAGS_N])
{
    T* tensor_a = (T*)pcs.tensor_a;
    T* tensor_b = (T*)pcs.tensor_b;

    uint a_rows[WARP_FRAGS_M];
    [unroll] for (uint mi = 0; mi < WARP_FRAGS_M; ++mi)
    {
        a_rows[mi] = tile.tile_m + tile.warp_row_base + mi * BLOCK;
        if (M_ALIGNED == 0) a_rows[mi] = min(a_rows[mi], pcs.m_dim - 1);
    }
    uint b_cols[WARP_FRAGS_N];
    [unroll] for (uint nj = 0; nj < WARP_FRAGS_N; ++nj)
    {
        b_cols[nj] = tile.tile_n + tile.warp_col_base + nj * BLOCK;
        if (N_ALIGNED == 0) b_cols[nj] = min(b_cols[nj], pcs.n_dim - 1);
    }

    uint k_main_end = (pcs.k_dim / BK) * BK;
    uint k = 0;

    for (; k + 8 * BLOCK <= k_main_end; k += 8 * BLOCK) accumulateSteps<T, BLOCK, 8>(tile, pcs, a_rows, b_cols, k, acc);
    for (; k + 4 * BLOCK <= k_main_end; k += 4 * BLOCK) accumulateSteps<T, BLOCK, 4>(tile, pcs, a_rows, b_cols, k, acc);
    for (; k + 2 * BLOCK <= k_main_end; k += 2 * BLOCK) accumulateSteps<T, BLOCK, 2>(tile, pcs, a_rows, b_cols, k, acc);
    for (; k + BLOCK <= k_main_end; k += BLOCK) accumulateSteps<T, BLOCK, 1>(tile, pcs, a_rows, b_cols, k, acc);

    if (k >= pcs.k_dim) return;

    // the K tail: staged through shared memory and zero padded to a whole BK tile
    uint smem_b_cols = WARPS_N * WARP_FRAGS_N * BLOCK;

    for (uint i = ti; i < (WARPS_M * WARP_FRAGS_M * BLOCK) * BK; i += tile.total_threads)
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

    for (uint step = 0; step < BK; step += BLOCK)
    {
        FragmentA<T, BLOCK> matA[WARP_FRAGS_M];
        [unroll] for (uint mi = 0; mi < WARP_FRAGS_M; ++mi)
        {
            uint off = (tile.warp_row_base + mi * BLOCK) * BK + step;
            matA[mi] = matA[mi].Load<linalg::CoopMatMatrixLayout.RowMajor>(Shared<T>.smemA, off, BK);
        }

        FragmentB<T, BLOCK> matB[WARP_FRAGS_N];
        [unroll] for (uint nj = 0; nj < WARP_FRAGS_N; ++nj)
        {
            uint off = step * smem_b_cols + (tile.warp_col_base + nj * BLOCK);
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

// alpha, then beta * out, then beta * bias, then the store; out-of-bounds blocks are skipped
void finishTile<T : ITorchVulkan & ICoopElement, let BLOCK : uint>(
    uint3 wg, CoopTile tile, CoopPushConstants<T.Accum> pcs, inout Accumulator<T.Accum, BLOCK> acc[WARP_FRAGS_M * WARP_FRAGS_N])
{
    T* tensor_out = (T*)pcs.tensor_out;
    T* tensor_bias = (T*)pcs.tensor_bias;

    [unroll] for (uint mi = 0; mi < WARP_FRAGS_M; ++mi)
    {
        [unroll] for (uint nj = 0; nj < WARP_FRAGS_N; ++nj)
        {
            uint flat_idx = mi * WARP_FRAGS_N + nj;
            uint row = tile.tile_m + tile.warp_row_base + mi * BLOCK;
            uint col = tile.tile_n + tile.warp_col_base + nj * BLOCK;

            if (HAS_ALPHA == 1)
            {
                acc[flat_idx] = acc[flat_idx] * pcs.alpha;
            }

            if (HAS_BETA == 1)
            {
                uint c_idx = tile.batch_offset_out + (row * pcs.strides_out.row) + (col * pcs.strides_out.col);
                Output<T, BLOCK> matC;
                if (OUT_TRANSPOSED == 1) {
                    matC = matC.Load<linalg::CoopMatMatrixLayout.ColumnMajor>(tensor_out, c_idx, pcs.strides_out.col);
                } else {
                    matC = matC.Load<linalg::CoopMatMatrixLayout.RowMajor>(tensor_out, c_idx, pcs.strides_out.row);
                }
                acc[flat_idx] = acc[flat_idx] + (Accumulator<T.Accum, BLOCK>(matC) * pcs.beta);
            }

            if (HAS_BIAS == 1)
            {
                // bias is not padded to the tile size either - clamp separately from the (unclamped) row/col used for `out`
                uint bias_row = min(row, pcs.m_dim - 1);
                uint bias_col = min(col, pcs.n_dim - 1);
                uint bias_idx = (wg.z * pcs.strides_bias.batch) + (bias_row * pcs.strides_bias.row) + (bias_col * pcs.strides_bias.col);
                Output<T, BLOCK> matBias;
                if (BIAS_TRANSPOSED == 1) {
                    matBias = matBias.Load<linalg::CoopMatMatrixLayout.ColumnMajor>(tensor_bias, bias_idx, pcs.strides_bias.col);
                } else {
                    matBias = matBias.Load<linalg::CoopMatMatrixLayout.RowMajor>(tensor_bias, bias_idx, pcs.strides_bias.row);
                }
                acc[flat_idx] = acc[flat_idx] + (Accumulator<T.Accum, BLOCK>(matBias) * pcs.beta);
            }
        }
    }

    [unroll] for (uint mi = 0; mi < WARP_FRAGS_M; ++mi)
    {
        [unroll] for (uint nj = 0; nj < WARP_FRAGS_N; ++nj)
        {
            uint row = tile.tile_m + tile.warp_row_base + mi * BLOCK;
            uint col = tile.tile_n + tile.warp_col_base + nj * BLOCK;

            // no need to write out-of-bounds values
            if (M_ALIGNED == 0 && row >= pcs.m_dim) continue;
            if (N_ALIGNED == 0 && col >= pcs.n_dim) continue;

            uint out_idx = tile.batch_offset_out + row * pcs.strides_out.row + col * pcs.strides_out.col;
            Output<T, BLOCK> result = Output<T, BLOCK>(acc[mi * WARP_FRAGS_N + nj]);
            if (OUT_TRANSPOSED == 1) {
                result.Store<linalg::CoopMatMatrixLayout.ColumnMajor>(tensor_out, out_idx, pcs.strides_out.col);
            } else {
                result.Store<linalg::CoopMatMatrixLayout.RowMajor>(tensor_out, out_idx, pcs.strides_out.row);
            }
        }
    }
}

void coopMatmulStaged<T : ITorchVulkan & ICoopElement, let VEC : int, let BLOCK : uint>(
    uint3 wg, uint ti, CoopPushConstants<T.Accum> pcs)
{
    CoopTile tile = makeTile<T.Accum, BLOCK>(wg, ti, pcs);

    Accumulator<T.Accum, BLOCK> acc[WARP_FRAGS_M * WARP_FRAGS_N];
    [unroll] for (uint i = 0; i < WARP_FRAGS_M * WARP_FRAGS_N; ++i) acc[i] = Accumulator<T.Accum, BLOCK>(T.Accum.ext_zero());
    GroupMemoryBarrierWithGroupSync();

    accumulateStaged<T, VEC, BLOCK>(ti, tile, pcs, acc);
    finishTile<T, BLOCK>(wg, tile, pcs, acc);
}

void coopMatmulDirect<T : ITorchVulkan & ICoopElement, let BLOCK : uint>(
    uint3 wg, uint ti, CoopPushConstants<T.Accum> pcs)
{
    CoopTile tile = makeTile<T.Accum, BLOCK>(wg, ti, pcs);

    Accumulator<T.Accum, BLOCK> acc[WARP_FRAGS_M * WARP_FRAGS_N];
    [unroll] for (uint i = 0; i < WARP_FRAGS_M * WARP_FRAGS_N; ++i) acc[i] = Accumulator<T.Accum, BLOCK>(T.Accum.ext_zero());
    GroupMemoryBarrierWithGroupSync();

    accumulateDirect<T, BLOCK>(ti, tile, pcs, acc);
    finishTile<T, BLOCK>(wg, tile, pcs, acc);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void matmul_coop_main<T : ITorchVulkan & ICoopElement, let VEC : int, let BLOCK : uint>(
    uint3 wg : SV_GroupID,
    uint ti : SV_GroupIndex,
    uniform CoopPushConstants<T.Accum> pcs)
{
    if (USE_SHARED_MEMORY == 1) coopMatmulStaged<T, VEC, BLOCK>(wg, ti, pcs);
    else coopMatmulDirect<T, BLOCK>(wg, ti, pcs);
}
