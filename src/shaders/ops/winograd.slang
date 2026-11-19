module winograd;

#ifdef COOPMAT
__exported import ops.matmul_coop;
#define WINOGRAD_SPEC 21
#define WINOGRAD_ELEMENT ITorchVulkan & ICoopElement
#else
__exported import core.dtype;
#define WINOGRAD_SPEC 0
#define WINOGRAD_ELEMENT ITorchVulkan
[[vk::constant_id(3)]] public const uint HAS_BIAS = 0;
#endif

[[vk::constant_id(WINOGRAD_SPEC + 0)]] public const uint MODE = 0;
[[vk::constant_id(WINOGRAD_SPEC + 1)]] public const uint WG_x = 64;
[[vk::constant_id(WINOGRAD_SPEC + 2)]] public const uint FLIP = 0;

public struct WinogradPushConstants
{
    public uint64_t src;
    public uint64_t dst;
    public uint64_t bias;
    public uint64_t weight;
    public uint rows;
    public uint cols;
    public uint height;
    public uint width;
    public uint tiles_h;
    public uint tiles_w;
    public uint tiles;
    public int pad_h;
    public int pad_w;
    public uint out_height;
    public uint out_width;
}

// G g G^T, with FLIP the filter [col, row] rotated 180 degrees
void filterTransform<T : ITorchVulkan>(uint row, uint col, bool live, WinogradPushConstants pcs, uint64_t weight, out T.Accum u[16])
{
    typealias A = T.Accum;
    A half = A.castFromF32(0.5);
    uint r = min(row, pcs.rows - 1);
    uint c = min(col, pcs.cols - 1);
    A g[3][3];
    [ForceUnroll] for (int i = 0; i < 9; i++)
    {
        uint at = FLIP == 1 ? (c * pcs.rows + r) * 9 + 8 - i : (r * pcs.cols + c) * 9 + i;
        A v = ((T*)weight)[at].toAccum();
        g[i / 3][i % 3] = live ? v : A.ext_zero();
    }

    A t[4][3];
    [ForceUnroll] for (int j = 0; j < 3; j++)
    {
        t[0][j] = g[0][j];
        t[1][j] = half * (g[0][j] + g[1][j] + g[2][j]);
        t[2][j] = half * (g[0][j] - g[1][j] + g[2][j]);
        t[3][j] = g[2][j];
    }
    [ForceUnroll] for (int i = 0; i < 4; i++)
    {
        u[i * 4 + 0] = t[i][0];
        u[i * 4 + 1] = half * (t[i][0] + t[i][1] + t[i][2]);
        u[i * 4 + 2] = half * (t[i][0] - t[i][1] + t[i][2]);
        u[i * 4 + 3] = t[i][2];
    }
}

struct TileAt
{
    uint item;
    uint h;
    uint w;

    __init(uint tile, WinogradPushConstants pcs)
    {
        uint per_item = pcs.tiles_h * pcs.tiles_w;
        item = tile / per_item;
        uint rest = tile - item * per_item;
        uint th = rest / pcs.tiles_w;
        h = 2 * th;
        w = 2 * (rest - th * pcs.tiles_w);
    }
}

// B^T d B, zero outside the input
void inputTransform<T : ITorchVulkan>(uint tile, uint channel, uint channels, bool live, WinogradPushConstants pcs, out T.Accum v[16])
{
    typealias A = T.Accum;
    TileAt at = TileAt(min(tile, pcs.tiles - 1), pcs);
    int h0 = int(at.h) - pcs.pad_h;
    int w0 = int(at.w) - pcs.pad_w;
    uint base = (at.item * channels + min(channel, channels - 1)) * pcs.height * pcs.width;

    A d[4][4];
    [ForceUnroll] for (int a = 0; a < 4; a++)
    {
        [ForceUnroll] for (int b = 0; b < 4; b++)
        {
            uint h = uint(h0 + a);
            uint w = uint(w0 + b);
            bool inside = live && h < pcs.height && w < pcs.width;
            A x = ((T*)pcs.src)[base + min(h, pcs.height - 1) * pcs.width + min(w, pcs.width - 1)].toAccum();
            d[a][b] = inside ? x : A.ext_zero();
        }
    }

    A t[4][4];
    [ForceUnroll] for (int b = 0; b < 4; b++)
    {
        t[0][b] = d[0][b] - d[2][b];
        t[1][b] = d[1][b] + d[2][b];
        t[2][b] = d[2][b] - d[1][b];
        t[3][b] = d[1][b] - d[3][b];
    }
    [ForceUnroll] for (int a = 0; a < 4; a++)
    {
        v[a * 4 + 0] = t[a][0] - t[a][2];
        v[a * 4 + 1] = t[a][1] + t[a][2];
        v[a * 4 + 2] = t[a][2] - t[a][1];
        v[a * 4 + 3] = t[a][1] - t[a][3];
    }
}

// A^T = [[1, 1, 1, 0], [0, 1, -1, -1]]
float outputWeight(int i, int a)
{
    if (i == 0) return a == 3 ? 0.0 : 1.0;
    return a == 0 ? 0.0 : (a == 1 ? 1.0 : -1.0);
}

void storeTile<T : ITorchVulkan>(uint tile, uint channel, uint height, uint width, T.Accum y[4], WinogradPushConstants pcs)
{
    TileAt at = TileAt(tile, pcs);
    uint base = (at.item * pcs.rows + channel) * height * width;
    [ForceUnroll] for (int i = 0; i < 2; i++)
    {
        if (at.h + i < height)
        {
            ((T*)pcs.dst)[base + (at.h + i) * width + at.w] = T.fromAccum(y[i * 2]);
            if (at.w + 1 < width) ((T*)pcs.dst)[base + (at.h + i) * width + at.w + 1] = T.fromAccum(y[i * 2 + 1]);
        }
    }
}

#ifdef COOPMAT

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
void winograd_main<T : WINOGRAD_ELEMENT>(
    uint3 wg : SV_GroupID,
    uint ti : SV_GroupIndex,
    uniform WinogradPushConstants pcs)
{
    fusedConv<T>(wg, ti, pcs);
}

#else

void transformFilter<T : ITorchVulkan>(uint col, uint row, WinogradPushConstants pcs)
{
    if (col >= pcs.cols) return;
    T.Accum u[16];
    filterTransform<T>(row, col, true, pcs, pcs.src, u);
    [ForceUnroll] for (int k = 0; k < 16; k++) ((T*)pcs.dst)[(k * pcs.rows + row) * pcs.cols + col] = T.fromAccum(u[k]);
}

void transformInput<T : ITorchVulkan>(uint tile, uint channel, WinogradPushConstants pcs)
{
    if (tile >= pcs.tiles) return;
    T.Accum v[16];
    inputTransform<T>(tile, channel, pcs.rows, true, pcs, v);
    [ForceUnroll] for (int k = 0; k < 16; k++) ((T*)pcs.dst)[(k * pcs.rows + channel) * pcs.tiles + tile] = T.fromAccum(v[k]);
}

void transformOutput<T : ITorchVulkan>(uint tile, uint channel, WinogradPushConstants pcs)
{
    typealias A = T.Accum;
    if (tile >= pcs.tiles) return;
    A bias = HAS_BIAS == 1 ? ((T*)pcs.bias)[channel].toAccum() : A.ext_zero();
    A y[4];
    [ForceUnroll] for (int q = 0; q < 4; q++) y[q] = bias;
    [ForceUnroll] for (int k = 0; k < 16; k++)
    {
        A m = ((T*)pcs.src)[(k * pcs.rows + channel) * pcs.tiles + tile].toAccum();
        [ForceUnroll] for (int q = 0; q < 4; q++) y[q] = y[q] + A.castFromF32(outputWeight(q / 2, k / 4) * outputWeight(q % 2, k % 4)) * m;
    }
    storeTile<T>(tile, channel, pcs.height, pcs.width, y, pcs);
}

void transformOutputGrad<T : ITorchVulkan>(uint tile, uint channel, WinogradPushConstants pcs)
{
    typealias A = T.Accum;
    if (tile >= pcs.tiles) return;
    TileAt at = TileAt(tile, pcs);
    uint base = (at.item * pcs.rows + channel) * pcs.height * pcs.width;

    A dy[2][2];
    [ForceUnroll] for (int i = 0; i < 2; i++)
    {
        [ForceUnroll] for (int j = 0; j < 2; j++)
        {
            uint h = at.h + i;
            uint w = at.w + j;
            A v = ((T*)pcs.src)[base + min(h, pcs.height - 1) * pcs.width + min(w, pcs.width - 1)].toAccum();
            dy[i][j] = h < pcs.height && w < pcs.width ? v : A.ext_zero();
        }
    }

    // A dy A^T
    A t[4][2];
    [ForceUnroll] for (int j = 0; j < 2; j++)
    {
        t[0][j] = dy[0][j];
        t[1][j] = dy[0][j] + dy[1][j];
        t[2][j] = dy[0][j] - dy[1][j];
        t[3][j] = -dy[1][j];
    }
    [ForceUnroll] for (int a = 0; a < 4; a++)
    {
        A m[4];
        m[0] = t[a][0];
        m[1] = t[a][0] + t[a][1];
        m[2] = t[a][0] - t[a][1];
        m[3] = -t[a][1];
        [ForceUnroll] for (int b = 0; b < 4; b++) ((T*)pcs.dst)[((a * 4 + b) * pcs.rows + channel) * pcs.tiles + tile] = T.fromAccum(m[b]);
    }
}

void transformFilterGrad<T : ITorchVulkan>(uint col, uint row, WinogradPushConstants pcs)
{
    typealias A = T.Accum;
    if (col >= pcs.cols) return;
    A half = A.castFromF32(0.5);

    A u[4][4];
    [ForceUnroll] for (int i = 0; i < 16; i++) u[i / 4][i % 4] = ((T*)pcs.src)[(i * pcs.rows + row) * pcs.cols + col].toAccum();

    // G^T dU G
    A t[3][4];
    [ForceUnroll] for (int j = 0; j < 4; j++)
    {
        t[0][j] = u[0][j] + half * (u[1][j] + u[2][j]);
        t[1][j] = half * (u[1][j] - u[2][j]);
        t[2][j] = half * (u[1][j] + u[2][j]) + u[3][j];
    }
    [ForceUnroll] for (int r = 0; r < 3; r++)
    {
        uint at = (row * pcs.cols + col) * 9 + r * 3;
        ((T*)pcs.dst)[at] = T.fromAccum(t[r][0] + half * (t[r][1] + t[r][2]));
        ((T*)pcs.dst)[at + 1] = T.fromAccum(half * (t[r][1] - t[r][2]));
        ((T*)pcs.dst)[at + 2] = T.fromAccum(half * (t[r][1] + t[r][2]) + t[r][3]);
    }
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void winograd_main<T : WINOGRAD_ELEMENT>(
    uint3 gi : SV_DispatchThreadID,
    uniform WinogradPushConstants pcs)
{
    if (MODE == 0) transformFilter<T>(gi.x, gi.y, pcs);
    else if (MODE == 1) transformInput<T>(gi.x, gi.y, pcs);
    else if (MODE == 2) transformOutput<T>(gi.x, gi.y, pcs);
    else if (MODE == 3) transformOutputGrad<T>(gi.x, gi.y, pcs);
    else transformFilterGrad<T>(gi.x, gi.y, pcs);
}

#endif
