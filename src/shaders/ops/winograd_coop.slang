module winograd_coop;

import helpers.gemm_coop;
import helpers.winograd;

static const uint TILE_ROWS = WARPS_M * WARP_FRAGS_M * BLOCK_SIZE;
static const uint TILE_COLS = WARPS_N * WARP_FRAGS_N * BLOCK_SIZE;
static const uint FRAGS = WARP_FRAGS_M * WARP_FRAGS_N;
static const uint OWNED = FRAGS * BLOCK_SIZE * BLOCK_SIZE / SUBGROUP_SIZE;

struct FusedShared<T>
{
    static groupshared T U[16 * TILE_ROWS * (BK + PAD)];
    static groupshared T V[16 * BK * (TILE_COLS + PAD)];
}

struct FusedScratch<U>
{
    static groupshared U M[TILE_ROWS * TILE_COLS];
}

void stageChunk<T : ITorchVulkan & ICoopElement>(uint ti, uint row0, uint col0, uint c0, WinogradPushConstants pcs)
{
    typealias A = T.Accum;
    for (uint i = ti; i < BK * TILE_COLS; i += WORKGROUP_SIZE)
    {
        uint c = i / TILE_COLS;
        uint col = i - c * TILE_COLS;
        A v[16];
        inputTransform<T>(col0 + col, c0 + c, pcs.cols, col0 + col < pcs.tiles && c0 + c < pcs.cols, pcs, v);
        [ForceUnroll] for (int k = 0; k < 16; k++) FusedShared<T>.V[(k * BK + c) * (TILE_COLS + PAD) + col] = T.fromAccum(v[k]);
    }
    for (uint i = ti; i < TILE_ROWS * BK; i += WORKGROUP_SIZE)
    {
        uint r = i / BK;
        uint c = i - r * BK;
        A u[16];
        filterTransform<T>(row0 + r, c0 + c, row0 + r < pcs.rows && c0 + c < pcs.cols, pcs, pcs.weight, u);
        [ForceUnroll] for (int k = 0; k < 16; k++) FusedShared<T>.U[(k * TILE_ROWS + r) * (BK + PAD) + c] = T.fromAccum(u[k]);
    }
}

void fusedConv<T : ITorchVulkan & ICoopElement>(uint3 wg, uint ti, WinogradPushConstants pcs)
{
    typealias A = T.Accum;
    uint row0 = wg.y * TILE_ROWS;
    uint col0 = wg.x * TILE_COLS;
    uint warp = ti / SUBGROUP_SIZE;
    uint warp_row = (warp / WARPS_N) * WARP_FRAGS_M * BLOCK_SIZE;
    uint warp_col = (warp % WARPS_N) * WARP_FRAGS_N * BLOCK_SIZE;

    Accumulator<A> acc[16 * FRAGS];
    [unroll] for (uint i = 0; i < 16 * FRAGS; ++i) acc[i] = Accumulator<A>(A.ext_zero());

    for (uint c0 = 0; c0 < pcs.cols; c0 += BK)
    {
        stageChunk<T>(ti, row0, col0, c0, pcs);
        GroupMemoryBarrierWithGroupSync();
        for (uint step = 0; step < BK; step += BLOCK_SIZE)
        {
            [ForceUnroll] for (int k = 0; k < 16; k++)
            {
                FragmentA<T> matA[WARP_FRAGS_M];
                [unroll] for (uint mi = 0; mi < WARP_FRAGS_M; ++mi)
                {
                    uint at = (k * TILE_ROWS + warp_row + mi * BLOCK_SIZE) * (BK + PAD) + step;
                    matA[mi] = matA[mi].Load<linalg::CoopMatMatrixLayout.RowMajor>(FusedShared<T>.U, at, BK + PAD);
                }
                [unroll] for (uint nj = 0; nj < WARP_FRAGS_N; ++nj)
                {
                    uint at = (k * BK + step) * (TILE_COLS + PAD) + warp_col + nj * BLOCK_SIZE;
                    FragmentB<T> matB;
                    matB = matB.Load<linalg::CoopMatMatrixLayout.RowMajor>(FusedShared<T>.V, at, TILE_COLS + PAD);
                    [unroll] for (uint mi = 0; mi < WARP_FRAGS_M; ++mi)
                    {
                        uint f = k * FRAGS + mi * WARP_FRAGS_N + nj;
                        acc[f] = linalg::coopMatMulAdd<A, false>(matA[mi], matB, acc[f]);
                    }
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    A y[OWNED * 4];
    [unroll] for (uint j = 0; j < OWNED * 4; ++j) y[j] = A.ext_zero();
    [ForceUnroll] for (int k = 0; k < 16; k++)
    {
        [unroll] for (uint mi = 0; mi < WARP_FRAGS_M; ++mi)
        {
            [unroll] for (uint nj = 0; nj < WARP_FRAGS_N; ++nj)
            {
                uint at = (warp_row + mi * BLOCK_SIZE) * TILE_COLS + warp_col + nj * BLOCK_SIZE;
                acc[k * FRAGS + mi * WARP_FRAGS_N + nj].Store<linalg::CoopMatMatrixLayout.RowMajor>(FusedScratch<A>.M, at, TILE_COLS);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        A w00 = A.castFromF32(outputWeight(0, k / 4) * outputWeight(0, k % 4));
        A w01 = A.castFromF32(outputWeight(0, k / 4) * outputWeight(1, k % 4));
        A w10 = A.castFromF32(outputWeight(1, k / 4) * outputWeight(0, k % 4));
        A w11 = A.castFromF32(outputWeight(1, k / 4) * outputWeight(1, k % 4));
        [unroll] for (uint j = 0; j < OWNED; ++j)
        {
            A m = FusedScratch<A>.M[ti + j * WORKGROUP_SIZE];
            y[j * 4 + 0] = y[j * 4 + 0] + w00 * m;
            y[j * 4 + 1] = y[j * 4 + 1] + w01 * m;
            y[j * 4 + 2] = y[j * 4 + 2] + w10 * m;
            y[j * 4 + 3] = y[j * 4 + 3] + w11 * m;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    [unroll] for (uint j = 0; j < OWNED; ++j)
    {
        uint e = ti + j * WORKGROUP_SIZE;
        uint row = row0 + e / TILE_COLS;
        uint tile = col0 + e % TILE_COLS;
        if (row < pcs.rows && tile < pcs.tiles)
        {
            A bias = HAS_BIAS == 1 ? ((T*)pcs.bias)[row].toAccum() : A.ext_zero();
            A out[4];
            [ForceUnroll] for (int q = 0; q < 4; q++) out[q] = y[j * 4 + q] + bias;
            storeTile<T>(tile, row, pcs.out_height, pcs.out_width, out, pcs);
        }
    }
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void winograd_coop_main<T : ITorchVulkan & ICoopElement>(
    uint3 wg : SV_GroupID,
    uint ti : SV_GroupIndex,
    uniform WinogradPushConstants pcs)
{
    fusedConv<T>(wg, ti, pcs);
}
