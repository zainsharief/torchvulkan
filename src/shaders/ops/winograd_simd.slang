module winograd_simd;

import helpers.winograd;

[[vk::constant_id(0)]] const uint WG_x = 64;
[[vk::constant_id(1)]] const uint HAS_BIAS = 0;
[[vk::constant_id(2)]] const uint ROW_THREADS = 4;
[[vk::constant_id(3)]] const uint THREAD_ROWS = 1;
[[vk::constant_id(4)]] const uint THREAD_COLS = 1;
[[vk::constant_id(5)]] const uint BK = 8;

static const uint COL_THREADS = WG_x / ROW_THREADS;
static const uint TILE_ROWS = ROW_THREADS * THREAD_ROWS;
static const uint TILE_COLS = COL_THREADS * THREAD_COLS;
static const uint OWNED = THREAD_ROWS * THREAD_COLS;

struct Shared<T>
{
    static groupshared T U[16 * BK * TILE_ROWS];
    static groupshared T V[16 * BK * TILE_COLS];
}

void stageChunk<T : ITorchVulkan>(uint ti, uint row0, uint col0, uint c0, WinogradPushConstants pcs)
{
    typealias A = T.Accum;
    for (uint i = ti; i < BK * TILE_COLS; i += WG_x)
    {
        uint c = i / TILE_COLS;
        uint col = i - c * TILE_COLS;
        A v[16];
        inputTransform<T>(col0 + col, c0 + c, pcs.cols, col0 + col < pcs.tiles && c0 + c < pcs.cols, pcs, v);
        [ForceUnroll] for (int k = 0; k < 16; k++) Shared<T>.V[(k * BK + c) * TILE_COLS + col] = T.fromAccum(v[k]);
    }
    for (uint i = ti; i < TILE_ROWS * BK; i += WG_x)
    {
        uint r = i / BK;
        uint c = i - r * BK;
        A u[16];
        filterTransform<T>(row0 + r, c0 + c, row0 + r < pcs.rows && c0 + c < pcs.cols, pcs, pcs.weight, u);
        [ForceUnroll] for (int k = 0; k < 16; k++) Shared<T>.U[(k * BK + c) * TILE_ROWS + r] = T.fromAccum(u[k]);
    }
}

// a thread owns rows tr + i * ROW_THREADS and tiles tc + j * COL_THREADS, all 16 products of each, so the
// output transform needs no shared memory
void fusedConv<T : ITorchVulkan>(uint3 wg, uint ti, WinogradPushConstants pcs)
{
    typealias A = T.Accum;
    uint row0 = wg.y * TILE_ROWS;
    uint col0 = wg.x * TILE_COLS;
    uint tr = ti / COL_THREADS;
    uint tc = ti - tr * COL_THREADS;

    A acc[16 * OWNED];
    [unroll] for (uint i = 0; i < 16 * OWNED; ++i) acc[i] = A.ext_zero();

    for (uint c0 = 0; c0 < pcs.cols; c0 += BK)
    {
        stageChunk<T>(ti, row0, col0, c0, pcs);
        GroupMemoryBarrierWithGroupSync();
        for (uint c = 0; c < BK; ++c)
        {
            [ForceUnroll] for (int k = 0; k < 16; k++)
            {
                A u[THREAD_ROWS];
                [unroll] for (uint i = 0; i < THREAD_ROWS; ++i) u[i] = Shared<T>.U[(k * BK + c) * TILE_ROWS + tr + i * ROW_THREADS].toAccum();
                [unroll] for (uint j = 0; j < THREAD_COLS; ++j)
                {
                    A v = Shared<T>.V[(k * BK + c) * TILE_COLS + tc + j * COL_THREADS].toAccum();
                    [unroll] for (uint i = 0; i < THREAD_ROWS; ++i) acc[k * OWNED + i * THREAD_COLS + j] = acc[k * OWNED + i * THREAD_COLS + j] + u[i] * v;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    [unroll] for (uint i = 0; i < THREAD_ROWS; ++i)
    {
        [unroll] for (uint j = 0; j < THREAD_COLS; ++j)
        {
            uint row = row0 + tr + i * ROW_THREADS;
            uint tile = col0 + tc + j * COL_THREADS;
            if (row < pcs.rows && tile < pcs.tiles)
            {
                A bias = HAS_BIAS == 1 ? ((T*)pcs.bias)[row].toAccum() : A.ext_zero();
                A y[4];
                [ForceUnroll] for (int q = 0; q < 4; q++) y[q] = bias;
                [ForceUnroll] for (int k = 0; k < 16; k++)
                {
                    A m = acc[k * OWNED + i * THREAD_COLS + j];
                    [ForceUnroll] for (int q = 0; q < 4; q++) y[q] = y[q] + A.castFromF32(outputWeight(q / 2, k / 4) * outputWeight(q % 2, k % 4)) * m;
                }
                storeTile<T>(tile, row, pcs.out_height, pcs.out_width, y, pcs);
            }
        }
    }
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void winograd_simd_main<T : ITorchVulkan>(
    uint3 wg : SV_GroupID,
    uint ti : SV_GroupIndex,
    uniform WinogradPushConstants pcs)
{
    fusedConv<T>(wg, ti, pcs);
}
