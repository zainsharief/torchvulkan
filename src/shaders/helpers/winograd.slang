module winograd;

__exported import core.dtype;

[[vk::constant_id(24)]] public const uint FLIP = 0;

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
public void filterTransform<T : ITorchVulkan>(uint row, uint col, bool live, WinogradPushConstants pcs, uint64_t weight, out T.Accum u[16])
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

public struct TileAt
{
    public uint item;
    public uint h;
    public uint w;

    public __init(uint tile, WinogradPushConstants pcs)
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
public void inputTransform<T : ITorchVulkan>(uint tile, uint channel, uint channels, bool live, WinogradPushConstants pcs, out T.Accum v[16])
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
public float outputWeight(int i, int a)
{
    if (i == 0) return a == 3 ? 0.0 : 1.0;
    return a == 0 ? 0.0 : (a == 1 ? 1.0 : -1.0);
}

public void storeTile<T : ITorchVulkan>(uint tile, uint channel, uint height, uint width, T.Accum y[4], WinogradPushConstants pcs)
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
