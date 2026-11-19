module winograd_transform;

import helpers.winograd;

[[vk::constant_id(0)]] const uint MODE = 0;
[[vk::constant_id(1)]] const uint WG_x = 64;
[[vk::constant_id(2)]] const uint HAS_BIAS = 0;

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
void winograd_transform_main<T : ITorchVulkan>(
    uint3 gi : SV_DispatchThreadID,
    uniform WinogradPushConstants pcs)
{
    if (MODE == 0) transformFilter<T>(gi.x, gi.y, pcs);
    else if (MODE == 1) transformInput<T>(gi.x, gi.y, pcs);
    else if (MODE == 2) transformOutput<T>(gi.x, gi.y, pcs);
    else if (MODE == 3) transformOutputGrad<T>(gi.x, gi.y, pcs);
    else transformFilterGrad<T>(gi.x, gi.y, pcs);
}
