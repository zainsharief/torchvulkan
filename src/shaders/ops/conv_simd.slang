module conv_simd;

import helpers.gemm_simd;
import helpers.conv;

[[vk::constant_id(6)]] const uint DIRECT = 0;

struct BiasEpilogue<T : ITorchVulkan> : IMatmulEpilogue<T.Accum>
{
    uint64_t bias;
    uint rows;
    uint group;

    T.Accum apply(uint batch, uint row, uint col, T.Accum acc)
    {
        if (HAS_BIAS == 0) return acc;
        return acc + ((T*)bias)[group * rows + row].toAccum();
    }
}

struct ColumnLoader<T : ITorchVulkan, B : IColumnGather<T>> : IMatmulLoader<T.Accum>
{
    B b;

    T.Accum load(uint batch, uint row, uint col) { return b.load(b.column(col), b.row(row)).toAccum(); }
}

struct RowLoader<T : ITorchVulkan, A : IColumnGather<T>> : IMatmulLoader<T.Accum>
{
    A a;

    T.Accum load(uint batch, uint row, uint col) { return a.load(a.column(row), a.row(col)).toAccum(); }
}

public static const int DIRECT_ROWS = 8;
public static const int DIRECT_COLUMNS = 8;

void directDot<T : __BuiltinArithmeticType, A : IMatmulLoader<T>, B : IMatmulLoader<T>, E : IMatmulEpilogue<T>, S : IMatmulStore<T>>(
    uint batch, uint row0, uint col, A a, B b, E epilogue, S out, uint M, uint N, uint K)
{
    if (col >= N) return;
    uint rows = min(M - row0, uint(DIRECT_ROWS));
    T acc[DIRECT_ROWS];
    [ForceUnroll] for (int r = 0; r < DIRECT_ROWS; r++) acc[r] = T(0);
    for (uint k = 0; k < K; k++)
    {
        T value = b.load(batch, k, col);
        [ForceUnroll] for (int r = 0; r < DIRECT_ROWS; r++)
        {
            if (r < rows) acc[r] += a.load(batch, row0 + r, k) * value;
        }
    }
    [ForceUnroll] for (int r = 0; r < DIRECT_ROWS; r++)
    {
        if (r < rows) out.store(batch, row0 + r, col, epilogue.apply(batch, row0 + r, col, acc[r]));
    }
}

void runGemm<T : __BuiltinArithmeticType, let VEC : int,
    A : IMatmulLoader<T>, B : IMatmulLoader<T>, E : IMatmulEpilogue<T>, S : IMatmulStore<T>>(
    uint3 li, uint3 wg, __ref T smemA[TILE_M * (TILE_K + 1)], __ref vector<T, VEC> smemB[TILE_K * (TILE_N / VEC)],
    A a, B b, E epilogue, S out, uint M, uint N, uint K)
{
    if (DIRECT == 1)
    {
        uint threads = WG_x * WG_y;
        uint col = wg.x * threads * DIRECT_COLUMNS + li.y * WG_x + li.x;
        for (int round = 0; round < DIRECT_COLUMNS; round++)
        {
            directDot<T, A, B, E, S>(wg.z, wg.y * DIRECT_ROWS, col + round * threads, a, b, epilogue, out, M, N, K);
        }
    }
    else
    {
        matmulTile<T, VEC, A, B, E, S>(li, wg, wg.z, smemA, smemB, a, b, epilogue, out, M, N, K);
    }
}

[shader("compute")]
[numthreads(WG_x, WG_y, 1)]
void conv_simd_main<T : ITorchVulkan, let VEC : int>(
    uint3 li : SV_GroupThreadID,
    uint3 wg : SV_GroupID,
    uniform ConvPushConstants pcs)
{
    typealias Acc = T.Accum;
    static groupshared Acc smemA[TILE_M * (TILE_K + 1)];
    static groupshared vector<Acc, VEC> smemB[TILE_K * (TILE_N / VEC)];
    uint ti = li.y * WG_x + li.x;
    uint threads = WG_x * WG_y;

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
        ColumnLoader<T, ForwardColumns<T>> columns = { b };
        BiasEpilogue<T> bias = { pcs.bias, g.cy, group };
        runGemm<Acc, VEC, WeightRows<T>, ColumnLoader<T, ForwardColumns<T>>, BiasEpilogue<T>, GroupedStore<T>>(
            li, wg, smemA, smemB, a, columns, bias, out, M, N, K);
    }
    else if (MODE == 1 && SUBPIXEL == 1)
    {
        Phase phase = loadPhase(outer, g, ti);
        WeightPhaseColumns<T> a = { pcs.weight, g, group };
        PhaseColumns<T> b = { pcs.y, g, phase, group };
        PhaseStore<T> out = { pcs.x, g, phase, group };
        uint M = g.cx, N = pcs.n * phase.positions, K = g.cy * phase.taps;
        ColumnLoader<T, PhaseColumns<T>> columns = { b };
        BiasEpilogue<T> bias = { pcs.bias, g.cx, group };
        runGemm<Acc, VEC, WeightPhaseColumns<T>, ColumnLoader<T, PhaseColumns<T>>, BiasEpilogue<T>, PhaseStore<T>>(
            li, wg, smemA, smemB, a, columns, bias, out, M, N, K);
    }
    else if (MODE == 1)
    {
        WeightColumns<T> a = { pcs.weight, g };
        GradInputColumns<T> b = { pcs.y, g, group };
        GroupedStore<T> out = { pcs.x, g.groups, g.cx, g.inputs };
        uint M = g.cx, N = pcs.n * g.inputs.d, K = g.cy * TAPS;
        ColumnLoader<T, GradInputColumns<T>> columns = { b };
        BiasEpilogue<T> bias = { pcs.bias, g.cx, group };
        runGemm<Acc, VEC, WeightColumns<T>, ColumnLoader<T, GradInputColumns<T>>, BiasEpilogue<T>, GroupedStore<T>>(
            li, wg, smemA, smemB, a, columns, bias, out, M, N, K);
    }
    else
    {
        uint first = outer * pcs.chunk;
        OutputRows<T> a = { pcs.y, g, group, first, count };
        GradWeightColumns<T> b = { pcs.x, g, group, first, count };
        BlockStore<T> out = { pcs.weight, g.cy, g.cx * TAPS };
        uint M = g.cy, N = g.cx * TAPS, K = pcs.chunk;
        RowLoader<T, OutputRows<T>> rows = { a };
        ColumnLoader<T, GradWeightColumns<T>> columns = { b };
        BiasEpilogue<T> bias = { pcs.bias, g.cy, group };
        runGemm<Acc, VEC, RowLoader<T, OutputRows<T>>, ColumnLoader<T, GradWeightColumns<T>>, BiasEpilogue<T>, BlockStore<T>>(
            li, wg, smemA, smemB, rows, columns, bias, out, M, N, K);
    }
}
