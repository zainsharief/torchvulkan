module conv;

#ifdef COOPMAT
__exported import ops.matmul_coop;
#define CONV_SPEC 21
#define CONV_ELEMENT ITorchVulkan & ICoopElement
static const uint GATHER_COLUMNS = 4;
#else
__exported import ops.matmul_simd;
#define CONV_SPEC 6
#define CONV_ELEMENT ITorchVulkan
[[vk::constant_id(CONV_SPEC + 16)]] public const uint DIRECT = 0;
#endif

[[vk::constant_id(CONV_SPEC + 0)]] public const uint MODE = 0;
[[vk::constant_id(CONV_SPEC + 1)]] public const uint KERNEL_D = 1;
[[vk::constant_id(CONV_SPEC + 2)]] public const uint KERNEL_H = 1;
[[vk::constant_id(CONV_SPEC + 3)]] public const uint KERNEL_W = 1;
[[vk::constant_id(CONV_SPEC + 4)]] public const uint STRIDE_D = 1;
[[vk::constant_id(CONV_SPEC + 5)]] public const uint STRIDE_H = 1;
[[vk::constant_id(CONV_SPEC + 6)]] public const uint STRIDE_W = 1;
[[vk::constant_id(CONV_SPEC + 7)]] public const uint PAD_D = 0;
[[vk::constant_id(CONV_SPEC + 8)]] public const uint PAD_H = 0;
[[vk::constant_id(CONV_SPEC + 9)]] public const uint PAD_W = 0;
[[vk::constant_id(CONV_SPEC + 10)]] public const uint DILATION_D = 1;
[[vk::constant_id(CONV_SPEC + 11)]] public const uint DILATION_H = 1;
[[vk::constant_id(CONV_SPEC + 12)]] public const uint DILATION_W = 1;
[[vk::constant_id(CONV_SPEC + 13)]] public const uint PADDED = 0;
[[vk::constant_id(CONV_SPEC + 14)]] public const uint UNIT_STRIDE = 1;
[[vk::constant_id(CONV_SPEC + 15)]] public const uint SUBPIXEL = 0;

static const uint TAPS = KERNEL_D * KERNEL_H * KERNEL_W;

static const uint MAX_PHASE_TAPS = 64;
groupshared uint phaseTap[MAX_PHASE_TAPS];
groupshared int3 phaseShift[MAX_PHASE_TAPS];
groupshared uint phaseTapCount;

public struct Metadata
{
    public uint64_t params;
}

public struct ConvPushConstants
{
    public uint64_t x;
    public uint64_t weight;
    public uint64_t y;
    public uint64_t bias;
    public uint64_t metadata;
    public uint n;
    public uint splits;
    public uint chunk;
}

// the metadata's params must match conv_metadata in conv.cpp
static const int PARAMS = 33;
groupshared uint params[PARAMS];

// ATen's IntDivider, exact for n below 2^31
struct Divider
{
    uint d;
    uint multiplier;
    uint shift;

    __init(int at)
    {
        d = params[at];
        multiplier = params[at + 1];
        shift = params[at + 2];
    }

    uint div(uint n) { return uint((uint64_t(uint((uint64_t(n) * multiplier) >> 32)) + n) >> shift); }
}

struct Size3
{
    int3 size;
    Divider y;
    Divider z;

    __init(int at)
    {
        size = int3(params[at], params[at + 1], params[at + 2]);
        y = Divider(at + 3);
        z = Divider(at + 6);
    }

    int3 unflatten(uint index)
    {
        uint rest = z.div(index);
        uint first = y.div(rest);
        return int3(first, rest - first * size.y, index - rest * size.z);
    }

    uint flatten(int3 at) { return uint((at.x * size.y + at.y) * size.z + at.z); }
}

struct Geometry
{
    Size3 inputSize;
    Size3 outputSize;
    Divider inputs;
    Divider outputs;
    uint groups;
    Divider groupsDiv;
    uint cx;
    uint cy;
    Divider cyDiv;

    __init()
    {
        inputSize = Size3(0);
        outputSize = Size3(9);
        groups = params[18];
        cx = params[19];
        cy = params[20];
        inputs = Divider(21);
        outputs = Divider(24);
        groupsDiv = Divider(27);
        cyDiv = Divider(30);
    }
}

struct Phase
{
    int3 phase;
    int3 size;
    uint positions;
    uint taps;

    int3 unflatten(uint index)
    {
        uint rest = index / uint(size.z);
        uint first = rest / uint(size.y);
        return int3(first, rest - first * size.y, index - rest * size.z);
    }
}

// every thread must call this
Phase loadPhase(uint index, Geometry g, uint ti)
{
    Phase p;
    p.phase = int3(index / (STRIDE_H * STRIDE_W), (index / STRIDE_W) % STRIDE_H, index % STRIDE_W);
    int3 in = g.inputSize.size;
    p.size = int3(in.x > p.phase.x ? (in.x - 1 - p.phase.x) / int(STRIDE_D) + 1 : 0,
                  in.y > p.phase.y ? (in.y - 1 - p.phase.y) / int(STRIDE_H) + 1 : 0,
                  in.z > p.phase.z ? (in.z - 1 - p.phase.z) / int(STRIDE_W) + 1 : 0);
    p.positions = uint(p.size.x * p.size.y * p.size.z);

    if (ti == 0)
    {
        uint count = 0;
        for (uint kd = 0; kd < KERNEL_D; kd++)
        {
            int ed = p.phase.x + int(PAD_D) - int(kd * DILATION_D);
            for (uint kh = 0; kh < KERNEL_H; kh++)
            {
                int eh = p.phase.y + int(PAD_H) - int(kh * DILATION_H);
                for (uint kw = 0; kw < KERNEL_W; kw++)
                {
                    int ew = p.phase.z + int(PAD_W) - int(kw * DILATION_W);
                    // truncating division is exact on multiples, negative ones included
                    if (ed % int(STRIDE_D) == 0 && eh % int(STRIDE_H) == 0 && ew % int(STRIDE_W) == 0 && count < MAX_PHASE_TAPS)
                    {
                        phaseTap[count] = (kd * KERNEL_H + kh) * KERNEL_W + kw;
                        phaseShift[count] = int3(ed / int(STRIDE_D), eh / int(STRIDE_H), ew / int(STRIDE_W));
                        count++;
                    }
                }
            }
        }
        phaseTapCount = count;
    }
    GroupMemoryBarrierWithGroupSync();
    p.taps = phaseTapCount;
    return p;
}

// component by component: SPIRV-Cross rejects a vector made only of specialization constants
int3 timesStride(int3 v) { return int3(v.x * int(STRIDE_D), v.y * int(STRIDE_H), v.z * int(STRIDE_W)); }
int3 overStride(int3 v) { return int3(v.x / int(STRIDE_D), v.y / int(STRIDE_H), v.z / int(STRIDE_W)); }
int3 plusPadding(int3 v) { return int3(v.x + int(PAD_D), v.y + int(PAD_H), v.z + int(PAD_W)); }
int3 minusPadding(int3 v) { return int3(v.x - int(PAD_D), v.y - int(PAD_H), v.z - int(PAD_W)); }

int3 tapOffset(uint tap)
{
    uint w = tap % KERNEL_W;
    uint rest = tap / KERNEL_W;
    uint h = rest % KERNEL_H;
    uint d = rest / KERNEL_H;
    return int3(d * DILATION_D, h * DILATION_H, w * DILATION_W);
}

#ifdef COOPMAT
interface IMatmulLoader<T>
{
    T load(uint batch, uint row, uint col);
}

interface IMatmulStore<T>
{
    void store(uint batch, uint row, uint col, T value);
}
#endif

interface IColumnGather<T : ITorchVulkan>
{
    associatedtype Column;
    associatedtype Row;
    Column column(uint col);
    Row row(uint row);
    T load(Column column, Row row);
}

struct WeightRows<T : ITorchVulkan> : IMatmulLoader<T.Accum>
{
    uint64_t weight;
    Geometry g;

    T.Accum load(uint group, uint row, uint col)
    {
        return ((T*)weight)[(group * g.cy + row) * g.cx * TAPS + col].toAccum();
    }
}

struct WeightColumns<T : ITorchVulkan> : IMatmulLoader<T.Accum>
{
    uint64_t weight;
    Geometry g;

    T.Accum load(uint group, uint row, uint col)
    {
        uint m = col / TAPS;
        return ((T*)weight)[((group * g.cy + m) * g.cx + row) * TAPS + col - m * TAPS].toAccum();
    }
}

struct OutputLine
{
    uint offset;
}

struct OutputPosition
{
    uint base;
    bool valid;
}

struct OutputRows<T : ITorchVulkan> : IColumnGather<T>
{
    uint64_t y;
    Geometry g;
    uint group;
    uint first;
    uint count;

    typealias Column = OutputLine;
    typealias Row = OutputPosition;

    OutputLine column(uint m)
    {
        OutputLine line;
        line.offset = (group * g.cy + m) * g.outputs.d;
        return line;
    }

    OutputPosition row(uint k)
    {
        uint at = min(first + k, count - 1);
        uint item = g.outputs.div(at);
        OutputPosition p;
        p.base = item * g.groups * g.cy * g.outputs.d + at - item * g.outputs.d;
        p.valid = first + k < count;
        return p;
    }

    T load(OutputLine line, OutputPosition p)
    {
        T v = ((T*)y)[p.base + line.offset];
        return p.valid ? v : T.ext_zero();
    }
}

struct UnfoldColumn
{
    int3 origin;
    uint base;
}

struct TapRow
{
    int3 tap;
    uint channel;
    uint offset;
}

struct ForwardColumns<T : ITorchVulkan> : IColumnGather<T>
{
    uint64_t x;
    Geometry g;
    uint group;

    typealias Column = UnfoldColumn;
    typealias Row = TapRow;

    UnfoldColumn column(uint col)
    {
        uint item = g.outputs.div(col);
        UnfoldColumn c;
        c.origin = minusPadding(timesStride(g.outputSize.unflatten(col - item * g.outputs.d)));
        c.base = (item * g.groups + group) * g.cx * g.inputs.d + g.inputSize.flatten(c.origin);
        return c;
    }

    TapRow row(uint row)
    {
        uint c = row / TAPS;
        TapRow r;
        r.tap = tapOffset(row - c * TAPS);
        r.offset = c * g.inputs.d + g.inputSize.flatten(r.tap);
        return r;
    }

    T load(UnfoldColumn column, TapRow row)
    {
        if (PADDED == 1 && any(uint3(column.origin + row.tap) >= uint3(g.inputSize.size))) return T.ext_zero();
        return ((T*)x)[column.base + row.offset];
    }
}

struct GatherColumn
{
    int3 shifted;
    uint base;
}

struct GradInputColumns<T : ITorchVulkan> : IColumnGather<T>
{
    uint64_t y;
    Geometry g;
    uint group;

    typealias Column = GatherColumn;
    typealias Row = TapRow;

    GatherColumn column(uint col)
    {
        uint item = g.inputs.div(col);
        GatherColumn c;
        c.shifted = plusPadding(g.inputSize.unflatten(col - item * g.inputs.d));
        c.base = (item * g.groups + group) * g.cy * g.outputs.d;
        if (UNIT_STRIDE == 1) c.base += g.outputSize.flatten(c.shifted);
        return c;
    }

    TapRow row(uint row)
    {
        uint m = row / TAPS;
        TapRow r;
        r.tap = tapOffset(row - m * TAPS);
        r.channel = m;
        r.offset = m * g.outputs.d - g.outputSize.flatten(r.tap);
        return r;
    }

    T load(GatherColumn column, TapRow row)
    {
        int3 o = column.shifted - row.tap;
        if (UNIT_STRIDE == 1)
        {
            if (any(uint3(o) >= uint3(g.outputSize.size))) return T.ext_zero();
            return ((T*)y)[column.base + row.offset];
        }
        if (any(o < int3(0))) return T.ext_zero();
        int3 q = overStride(o);
        if (any(timesStride(q) != o) || any(q >= g.outputSize.size)) return T.ext_zero();
        return ((T*)y)[column.base + row.channel * g.outputs.d + g.outputSize.flatten(q)];
    }
}

struct WeightPhaseColumns<T : ITorchVulkan> : IMatmulLoader<T.Accum>
{
    uint64_t weight;
    Geometry g;
    uint group;

    T.Accum load(uint batch, uint row, uint col)
    {
        uint j = g.cyDiv.div(col);
        uint m = col - j * g.cy;
        return ((T*)weight)[((group * g.cy + m) * g.cx + row) * TAPS + phaseTap[j]].toAccum();
    }
}

struct PhaseColumn
{
    int3 at;
    uint base;
}

struct PhaseColumns<T : ITorchVulkan> : IColumnGather<T>
{
    uint64_t y;
    Geometry g;
    Phase phase;
    uint group;

    typealias Column = PhaseColumn;
    typealias Row = TapRow;

    PhaseColumn column(uint col)
    {
        uint item = col / phase.positions;
        PhaseColumn c;
        c.at = phase.unflatten(col - item * phase.positions);
        c.base = (item * g.groups + group) * g.cy * g.outputs.d + g.outputSize.flatten(c.at);
        return c;
    }

    TapRow row(uint row)
    {
        uint j = g.cyDiv.div(row);
        TapRow r;
        r.tap = phaseShift[j];
        r.offset = (row - j * g.cy) * g.outputs.d + g.outputSize.flatten(r.tap);
        return r;
    }

    T load(PhaseColumn column, TapRow row)
    {
        if (any(uint3(column.at + row.tap) >= uint3(g.outputSize.size))) return T.ext_zero();
        return ((T*)y)[column.base + row.offset];
    }
}

struct PhaseStore<T : ITorchVulkan> : IMatmulStore<T.Accum>
{
    uint64_t x;
    Geometry g;
    Phase phase;
    uint group;

    void store(uint batch, uint row, uint col, T.Accum value)
    {
        uint item = col / phase.positions;
        int3 i = timesStride(phase.unflatten(col - item * phase.positions)) + phase.phase;
        ((T*)x)[((item * g.groups + group) * g.cx + row) * g.inputs.d + g.inputSize.flatten(i)] = T.fromAccum(value);
    }
}

struct PositionRow
{
    int3 origin;
    uint base;
    bool valid;
}

struct WeightColumn
{
    int3 tap;
    uint offset;
}

struct GradWeightColumns<T : ITorchVulkan> : IColumnGather<T>
{
    uint64_t x;
    Geometry g;
    uint group;
    uint first;
    uint count;

    typealias Column = WeightColumn;
    typealias Row = PositionRow;

    WeightColumn column(uint col)
    {
        uint c = col / TAPS;
        WeightColumn w;
        w.tap = tapOffset(col - c * TAPS);
        w.offset = (group * g.cx + c) * g.inputs.d + g.inputSize.flatten(w.tap);
        return w;
    }

    PositionRow row(uint row)
    {
        uint k = min(first + row, count - 1);
        uint item = g.outputs.div(k);
        PositionRow r;
        r.origin = minusPadding(timesStride(g.outputSize.unflatten(k - item * g.outputs.d)));
        r.base = item * g.groups * g.cx * g.inputs.d + g.inputSize.flatten(r.origin);
        r.valid = first + row < count;
        return r;
    }

    T load(WeightColumn column, PositionRow row)
    {
        if (!row.valid) return T.ext_zero();
        if (PADDED == 1 && any(uint3(row.origin + column.tap) >= uint3(g.inputSize.size))) return T.ext_zero();
        return ((T*)x)[row.base + column.offset];
    }
}

struct GroupedStore<T : ITorchVulkan> : IMatmulStore<T.Accum>
{
    uint64_t address;
    uint groups;
    uint channels;
    Divider positions;

    void store(uint group, uint row, uint col, T.Accum value)
    {
        uint item = positions.div(col);
        uint channel = (item * groups + group) * channels + row;
        ((T*)address)[channel * positions.d + col - item * positions.d] = T.fromAccum(value);
    }
}

struct BlockStore<T : ITorchVulkan> : IMatmulStore<T.Accum>
{
    uint64_t address;
    uint rows;
    uint columns;

    void store(uint batch, uint row, uint col, T.Accum value)
    {
        ((T.Accum*)address)[(batch * rows + row) * columns + col] = value;
    }
}

#ifdef COOPMAT

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
        T loaded = T.fromAccum(a.load(tile.batch, min(row, tile.M - 1), min(kk, tile.K - 1)));
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

    // structured control flow only, as in matmul_coop's epilogue
    [unroll] for (uint mi = 0; mi < WARP_FRAGS_M; ++mi)
    {
        [unroll] for (uint nj = 0; nj < WARP_FRAGS_N; ++nj)
        {
            uint row = tile.row0 + tile.warp_row + mi * BLOCK_SIZE;
            uint col = tile.col0 + tile.warp_col + nj * BLOCK_SIZE;
            if (row < tile.M && col < tile.N)
            {
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

#else

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

#endif

[shader("compute")]
#ifdef COOPMAT
[numthreads(WORKGROUP_SIZE, 1, 1)]
#else
[numthreads(WG_x, WG_y, 1)]
#endif
void conv_main<T : CONV_ELEMENT, let VEC : int>(
    uint3 li : SV_GroupThreadID,
    uint3 wg : SV_GroupID,
    uniform ConvPushConstants pcs)
{
#ifdef COOPMAT
    uint ti = li.x;
    uint threads = WORKGROUP_SIZE;
#else
    typealias Acc = T.Accum;
    static groupshared Acc smemA[TILE_M * (TILE_K + 1)];
    static groupshared vector<Acc, VEC> smemB[TILE_K * (TILE_N / VEC)];
    uint ti = li.y * WG_x + li.x;
    uint threads = WG_x * WG_y;
#endif

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
#ifdef COOPMAT
        convGemm<T, WeightRows<T>, ForwardColumns<T>, GroupedStore<T>>(wg, ti, a, b, out, pcs.bias, group, M, N, K);
#else
        ColumnLoader<T, ForwardColumns<T>> columns = { b };
        BiasEpilogue<T> bias = { pcs.bias, g.cy, group };
        runGemm<Acc, VEC, WeightRows<T>, ColumnLoader<T, ForwardColumns<T>>, BiasEpilogue<T>, GroupedStore<T>>(
            li, wg, smemA, smemB, a, columns, bias, out, M, N, K);
#endif
    }
    else if (MODE == 1 && SUBPIXEL == 1)
    {
        Phase phase = loadPhase(outer, g, ti);
        WeightPhaseColumns<T> a = { pcs.weight, g, group };
        PhaseColumns<T> b = { pcs.y, g, phase, group };
        PhaseStore<T> out = { pcs.x, g, phase, group };
        uint M = g.cx, N = pcs.n * phase.positions, K = g.cy * phase.taps;
#ifdef COOPMAT
        convGemm<T, WeightPhaseColumns<T>, PhaseColumns<T>, PhaseStore<T>>(wg, ti, a, b, out, pcs.bias, group, M, N, K);
#else
        ColumnLoader<T, PhaseColumns<T>> columns = { b };
        BiasEpilogue<T> bias = { pcs.bias, g.cx, group };
        runGemm<Acc, VEC, WeightPhaseColumns<T>, ColumnLoader<T, PhaseColumns<T>>, BiasEpilogue<T>, PhaseStore<T>>(
            li, wg, smemA, smemB, a, columns, bias, out, M, N, K);
#endif
    }
    else if (MODE == 1)
    {
        WeightColumns<T> a = { pcs.weight, g };
        GradInputColumns<T> b = { pcs.y, g, group };
        GroupedStore<T> out = { pcs.x, g.groups, g.cx, g.inputs };
        uint M = g.cx, N = pcs.n * g.inputs.d, K = g.cy * TAPS;
#ifdef COOPMAT
        convGemm<T, WeightColumns<T>, GradInputColumns<T>, GroupedStore<T>>(wg, ti, a, b, out, pcs.bias, group, M, N, K);
#else
        ColumnLoader<T, GradInputColumns<T>> columns = { b };
        BiasEpilogue<T> bias = { pcs.bias, g.cx, group };
        runGemm<Acc, VEC, WeightColumns<T>, ColumnLoader<T, GradInputColumns<T>>, BiasEpilogue<T>, GroupedStore<T>>(
            li, wg, smemA, smemB, a, columns, bias, out, M, N, K);
#endif
    }
    else
    {
        uint first = outer * pcs.chunk;
        OutputRows<T> a = { pcs.y, g, group, first, count };
        GradWeightColumns<T> b = { pcs.x, g, group, first, count };
        BlockStore<T> out = { pcs.weight, g.cy, g.cx * TAPS };
        uint M = g.cy, N = g.cx * TAPS, K = pcs.chunk;
#ifdef COOPMAT
        convGemmK<T, OutputRows<T>, GradWeightColumns<T>, BlockStore<T>>(wg, ti, a, b, out, M, N, K);
#else
        RowLoader<T, OutputRows<T>> rows = { a };
        ColumnLoader<T, GradWeightColumns<T>> columns = { b };
        BiasEpilogue<T> bias = { pcs.bias, g.cy, group };
        runGemm<Acc, VEC, RowLoader<T, OutputRows<T>>, ColumnLoader<T, GradWeightColumns<T>>, BiasEpilogue<T>, BlockStore<T>>(
            li, wg, smemA, smemB, rows, columns, bias, out, M, N, K);
#endif
    }
}
