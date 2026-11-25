module conv;

__exported import core.dtype;
__exported import helpers.gemm;

#define CONV_SPEC 24

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

public static const uint TAPS = KERNEL_D * KERNEL_H * KERNEL_W;

public static const uint MAX_PHASE_TAPS = 64;
public groupshared uint phaseTap[MAX_PHASE_TAPS];
public groupshared int3 phaseShift[MAX_PHASE_TAPS];
public groupshared uint phaseTapCount;

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
public static const int PARAMS = 33;
public groupshared uint params[PARAMS];

// ATen's IntDivider, exact for n below 2^31
public struct Divider
{
    public uint d;
    public uint multiplier;
    public uint shift;

    public __init(int at)
    {
        d = params[at];
        multiplier = params[at + 1];
        shift = params[at + 2];
    }

    public uint div(uint n) { return uint((uint64_t(uint((uint64_t(n) * multiplier) >> 32)) + n) >> shift); }
}

public struct Size3
{
    public int3 size;
    public Divider y;
    public Divider z;

    public __init(int at)
    {
        size = int3(params[at], params[at + 1], params[at + 2]);
        y = Divider(at + 3);
        z = Divider(at + 6);
    }

    public int3 unflatten(uint index)
    {
        uint rest = z.div(index);
        uint first = y.div(rest);
        return int3(first, rest - first * size.y, index - rest * size.z);
    }

    public uint flatten(int3 at) { return uint((at.x * size.y + at.y) * size.z + at.z); }
}

public struct Geometry
{
    public Size3 inputSize;
    public Size3 outputSize;
    public Divider inputs;
    public Divider outputs;
    public uint groups;
    public Divider groupsDiv;
    public uint cx;
    public uint cy;
    public Divider cyDiv;

    public __init()
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

public struct Phase
{
    public int3 phase;
    public int3 size;
    public uint positions;
    public uint taps;

    public int3 unflatten(uint index)
    {
        uint rest = index / uint(size.z);
        uint first = rest / uint(size.y);
        return int3(first, rest - first * size.y, index - rest * size.z);
    }
}

// every thread must call this
public Phase loadPhase(uint index, Geometry g, uint ti)
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
                    int3 e = int3(ed, eh, ew);
                    int3 shift = overStride(e);
                    // truncating division is exact on multiples, negative ones included. Not %: Vulkan leaves OpSRem
                    // undefined for negative operands (NVIDIA returns the unsigned remainder). And one vector compare,
                    // not a chain of &&: NVIDIA miscompiled that chain in conv_coop (phase 0 found no taps)
                    if (all(timesStride(shift) == e) && count < MAX_PHASE_TAPS)
                    {
                        phaseTap[count] = (kd * KERNEL_H + kh) * KERNEL_W + kw;
                        phaseShift[count] = shift;
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
public int3 timesStride(int3 v) { return int3(v.x * int(STRIDE_D), v.y * int(STRIDE_H), v.z * int(STRIDE_W)); }
public int3 overStride(int3 v) { return int3(v.x / int(STRIDE_D), v.y / int(STRIDE_H), v.z / int(STRIDE_W)); }
public int3 plusPadding(int3 v) { return int3(v.x + int(PAD_D), v.y + int(PAD_H), v.z + int(PAD_W)); }
public int3 minusPadding(int3 v) { return int3(v.x - int(PAD_D), v.y - int(PAD_H), v.z - int(PAD_W)); }

public int3 tapOffset(uint tap)
{
    uint w = tap % KERNEL_W;
    uint rest = tap / KERNEL_W;
    uint h = rest % KERNEL_H;
    uint d = rest / KERNEL_H;
    return int3(d * DILATION_D, h * DILATION_H, w * DILATION_W);
}

public interface IColumnGather<T : ITorchVulkan>
{
    associatedtype Column;
    associatedtype Row;
    Column column(uint col);
    Row row(uint row);
    T load(Column column, Row row);
}

public struct WeightRows<T : ITorchVulkan> : IMatmulLoader<T.Accum>
{
    public uint64_t weight;
    public Geometry g;

    public T.Accum load(uint group, uint row, uint col)
    {
        return ((T*)weight)[(group * g.cy + row) * g.cx * TAPS + col].toAccum();
    }
}

public struct WeightColumns<T : ITorchVulkan> : IMatmulLoader<T.Accum>
{
    public uint64_t weight;
    public Geometry g;

    public T.Accum load(uint group, uint row, uint col)
    {
        uint m = col / TAPS;
        return ((T*)weight)[((group * g.cy + m) * g.cx + row) * TAPS + col - m * TAPS].toAccum();
    }
}

public struct OutputLine
{
    public uint offset;
}

public struct OutputPosition
{
    public uint base;
    public bool valid;
}

public struct OutputRows<T : ITorchVulkan> : IColumnGather<T>
{
    public uint64_t y;
    public Geometry g;
    public uint group;
    public uint first;
    public uint count;

    public typealias Column = OutputLine;
    public typealias Row = OutputPosition;

    public OutputLine column(uint m)
    {
        OutputLine line;
        line.offset = (group * g.cy + m) * g.outputs.d;
        return line;
    }

    public OutputPosition row(uint k)
    {
        uint at = min(first + k, count - 1);
        uint item = g.outputs.div(at);
        OutputPosition p;
        p.base = item * g.groups * g.cy * g.outputs.d + at - item * g.outputs.d;
        p.valid = first + k < count;
        return p;
    }

    public T load(OutputLine line, OutputPosition p)
    {
        T v = ((T*)y)[p.base + line.offset];
        return p.valid ? v : T.ext_zero();
    }
}

public struct UnfoldColumn
{
    public int3 origin;
    public uint base;
}

public struct TapRow
{
    public int3 tap;
    public uint channel;
    public uint offset;
}

public struct ForwardColumns<T : ITorchVulkan> : IColumnGather<T>
{
    public uint64_t x;
    public Geometry g;
    public uint group;

    public typealias Column = UnfoldColumn;
    public typealias Row = TapRow;

    public UnfoldColumn column(uint col)
    {
        uint item = g.outputs.div(col);
        UnfoldColumn c;
        c.origin = minusPadding(timesStride(g.outputSize.unflatten(col - item * g.outputs.d)));
        c.base = (item * g.groups + group) * g.cx * g.inputs.d + g.inputSize.flatten(c.origin);
        return c;
    }

    public TapRow row(uint row)
    {
        uint c = row / TAPS;
        TapRow r;
        r.tap = tapOffset(row - c * TAPS);
        r.offset = c * g.inputs.d + g.inputSize.flatten(r.tap);
        return r;
    }

    public T load(UnfoldColumn column, TapRow row)
    {
        if (PADDED == 1 && any(uint3(column.origin + row.tap) >= uint3(g.inputSize.size))) return T.ext_zero();
        return ((T*)x)[column.base + row.offset];
    }
}

public struct GatherColumn
{
    public int3 shifted;
    public uint base;
}

public struct GradInputColumns<T : ITorchVulkan> : IColumnGather<T>
{
    public uint64_t y;
    public Geometry g;
    public uint group;

    public typealias Column = GatherColumn;
    public typealias Row = TapRow;

    public GatherColumn column(uint col)
    {
        uint item = g.inputs.div(col);
        GatherColumn c;
        c.shifted = plusPadding(g.inputSize.unflatten(col - item * g.inputs.d));
        c.base = (item * g.groups + group) * g.cy * g.outputs.d;
        if (UNIT_STRIDE == 1) c.base += g.outputSize.flatten(c.shifted);
        return c;
    }

    public TapRow row(uint row)
    {
        uint m = row / TAPS;
        TapRow r;
        r.tap = tapOffset(row - m * TAPS);
        r.channel = m;
        r.offset = m * g.outputs.d - g.outputSize.flatten(r.tap);
        return r;
    }

    public T load(GatherColumn column, TapRow row)
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

public struct WeightPhaseColumns<T : ITorchVulkan> : IMatmulLoader<T.Accum>
{
    public uint64_t weight;
    public Geometry g;
    public uint group;

    public T.Accum load(uint batch, uint row, uint col)
    {
        uint j = g.cyDiv.div(col);
        uint m = col - j * g.cy;
        return ((T*)weight)[((group * g.cy + m) * g.cx + row) * TAPS + phaseTap[j]].toAccum();
    }
}

public struct PhaseColumn
{
    public int3 at;
    public uint base;
}

public struct PhaseColumns<T : ITorchVulkan> : IColumnGather<T>
{
    public uint64_t y;
    public Geometry g;
    public Phase phase;
    public uint group;

    public typealias Column = PhaseColumn;
    public typealias Row = TapRow;

    public PhaseColumn column(uint col)
    {
        uint item = col / phase.positions;
        PhaseColumn c;
        c.at = phase.unflatten(col - item * phase.positions);
        c.base = (item * g.groups + group) * g.cy * g.outputs.d + g.outputSize.flatten(c.at);
        return c;
    }

    public TapRow row(uint row)
    {
        uint j = g.cyDiv.div(row);
        TapRow r;
        r.tap = phaseShift[j];
        r.offset = (row - j * g.cy) * g.outputs.d + g.outputSize.flatten(r.tap);
        return r;
    }

    public T load(PhaseColumn column, TapRow row)
    {
        if (any(uint3(column.at + row.tap) >= uint3(g.outputSize.size))) return T.ext_zero();
        return ((T*)y)[column.base + row.offset];
    }
}

public struct PhaseStore<T : ITorchVulkan> : IMatmulStore<T.Accum>
{
    public uint64_t x;
    public Geometry g;
    public Phase phase;
    public uint group;

    public void store(uint batch, uint row, uint col, T.Accum value)
    {
        uint item = col / phase.positions;
        int3 i = timesStride(phase.unflatten(col - item * phase.positions)) + phase.phase;
        ((T*)x)[((item * g.groups + group) * g.cx + row) * g.inputs.d + g.inputSize.flatten(i)] = T.fromAccum(value);
    }
}

public struct PositionRow
{
    public int3 origin;
    public uint base;
    public bool valid;
}

public struct WeightColumn
{
    public int3 tap;
    public uint offset;
}

public struct GradWeightColumns<T : ITorchVulkan> : IColumnGather<T>
{
    public uint64_t x;
    public Geometry g;
    public uint group;
    public uint first;
    public uint count;

    public typealias Column = WeightColumn;
    public typealias Row = PositionRow;

    public WeightColumn column(uint col)
    {
        uint c = col / TAPS;
        WeightColumn w;
        w.tap = tapOffset(col - c * TAPS);
        w.offset = (group * g.cx + c) * g.inputs.d + g.inputSize.flatten(w.tap);
        return w;
    }

    public PositionRow row(uint row)
    {
        uint k = min(first + row, count - 1);
        uint item = g.outputs.div(k);
        PositionRow r;
        r.origin = minusPadding(timesStride(g.outputSize.unflatten(k - item * g.outputs.d)));
        r.base = item * g.groups * g.cx * g.inputs.d + g.inputSize.flatten(r.origin);
        r.valid = first + row < count;
        return r;
    }

    public T load(WeightColumn column, PositionRow row)
    {
        if (!row.valid) return T.ext_zero();
        if (PADDED == 1 && any(uint3(row.origin + column.tap) >= uint3(g.inputSize.size))) return T.ext_zero();
        return ((T*)x)[row.base + column.offset];
    }
}

public struct GroupedStore<T : ITorchVulkan> : IMatmulStore<T.Accum>
{
    public uint64_t address;
    public uint groups;
    public uint channels;
    public Divider positions;

    public void store(uint group, uint row, uint col, T.Accum value)
    {
        uint item = positions.div(col);
        uint channel = (item * groups + group) * channels + row;
        ((T*)address)[channel * positions.d + col - item * positions.d] = T.fromAccum(value);
    }
}

public struct BlockStore<T : ITorchVulkan> : IMatmulStore<T.Accum>
{
    public uint64_t address;
    public uint rows;
    public uint columns;

    public void store(uint batch, uint row, uint col, T.Accum value)
    {
        ((T.Accum*)address)[(batch * rows + row) * columns + col] = value;
    }
}
