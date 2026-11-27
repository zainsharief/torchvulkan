module pool;

__exported import core.dtype;
import core.indexing;

[[vk::constant_id(0)]] public const uint MODE = 0;
[[vk::constant_id(1)]] public const uint WG_x = 0;

public struct Metadata
{
    public uint64_t params;
}

public struct PoolPushConstants
{
    public uint64_t input;
    public uint64_t output;
    public uint64_t indices;
    public uint64_t samples;
    public uint64_t metadata;
    public uint64_t count;
}

struct Geometry
{
    int in[3];
    int out[3];
    int kernel[3];
    int stride[3];
    int pad[3];
    int dilation[3];
    bool count_include_pad;
    int divisor_override;
    int sample_dims;
}

Geometry loadGeometry(uint64_t address)
{
    uint* p = (uint*)(*(Metadata*)address).params;
    Geometry g;
    [ForceUnroll] for (int a = 0; a < 3; a++)
    {
        g.in[a] = int(p[a]);
        g.out[a] = int(p[3 + a]);
        g.kernel[a] = int(p[6 + a]);
        g.stride[a] = int(p[9 + a]);
        g.pad[a] = int(p[12 + a]);
        g.dilation[a] = int(p[15 + a]);
    }
    g.count_include_pad = p[18] != 0;
    g.divisor_override = int(p[19]);
    g.sample_dims = int(p[20]);
    return g;
}

int adaptiveStart(int o, int input, int output) { return int((int64_t(o) * input) / output); }
int adaptiveEnd(int o, int input, int output) { return int((int64_t(o + 1) * input + output - 1) / output); }

// as the CPU's generate_intervals: alpha rounded to the input dtype
int fractionalStart<T : ITorchVulkan>(int o, T sample, int input, int output, int kernel)
{
    if (o == output - 1) return input - kernel;
    T.Accum alpha = T.ext_div(castValue<int64_t, T>(input - kernel), castValue<int64_t, T>(output - 1)).toAccum();
    T.Accum s = sample.toAccum();
    int64_t scaled = castValue<T.Accum, int64_t>(T.Accum.ext_mul(T.Accum.ext_add(castValue<int64_t, T.Accum>(o), s), alpha));
    return int(scaled - castValue<T.Accum, int64_t>(T.Accum.ext_mul(s, alpha)));
}

// the first maximum wins, and a NaN takes over wherever it appears, as in the CPU loop
void maxWindow<T : ITorchVulkan>(uint64_t input, int64_t base, int lo[3], int hi[3], int step[3], int in[3], out T value, out int64_t index)
{
    index = int64_t(lo[0]) * in[1] * in[2] + int64_t(lo[1]) * in[2] + lo[2];
    value = T.ext_lowest();
    for (int d = lo[0]; d < hi[0]; d += step[0])
    {
        for (int h = lo[1]; h < hi[1]; h += step[1])
        {
            for (int w = lo[2]; w < hi[2]; w += step[2])
            {
                int64_t position = int64_t(d) * in[1] * in[2] + int64_t(h) * in[2] + w;
                T v = T.load(input, uint64_t(base + position));
                if (T.ext_gt(v, value) || T.ext_isnan(v))
                {
                    value = v;
                    index = position;
                }
            }
        }
    }
}

void poolForward<T : ITorchVulkan>(uint64_t idx, PoolPushConstants pcs, Geometry g)
{
    int64_t outputs = int64_t(g.out[0]) * g.out[1] * g.out[2];
    int64_t plane = int64_t(idx) / outputs;
    int64_t rest = int64_t(idx) % outputs;
    int o[3];
    o[0] = int(rest / (int64_t(g.out[1]) * g.out[2]));
    o[1] = int((rest / g.out[2]) % g.out[1]);
    o[2] = int(rest % g.out[2]);

    int64_t base = plane * int64_t(g.in[0]) * g.in[1] * g.in[2];

    int lo[3];
    int hi[3];
    int step[3];
    if (MODE == 0)
    {
        [ForceUnroll] for (int a = 0; a < 3; a++)
        {
            int start = o[a] * g.stride[a] - g.pad[a];
            hi[a] = min(start + (g.kernel[a] - 1) * g.dilation[a] + 1, g.in[a]);
            while (start < 0) start += g.dilation[a];
            lo[a] = start;
            step[a] = g.dilation[a];
        }
    }
    else if (MODE == 4)
    {
        [ForceUnroll] for (int a = 0; a < 3; a++)
        {
            if (g.sample_dims == 2 && a == 0)
            {
                lo[a] = 0;
            }
            else
            {
                int k = (g.sample_dims == 2) ? (a == 2 ? 0 : 1) : a;
                lo[a] = fractionalStart<T>(o[a], T.load(pcs.samples, plane * g.sample_dims + k), g.in[a], g.out[a], g.kernel[a]);
            }
            hi[a] = lo[a] + g.kernel[a];
            step[a] = 1;
        }
    }
    else
    {
        [ForceUnroll] for (int a = 0; a < 3; a++)
        {
            lo[a] = adaptiveStart(o[a], g.in[a], g.out[a]);
            hi[a] = adaptiveEnd(o[a], g.in[a], g.out[a]);
            step[a] = 1;
        }
    }

    if (MODE == 0 || MODE == 2 || MODE == 4)
    {
        T value;
        int64_t index;
        maxWindow<T>(pcs.input, base, lo, hi, step, g.in, value, index);
        T.store(pcs.output, idx, value);
        ((int64_t*)pcs.indices)[idx] = index;
        return;
    }

    if (MODE == 1)
    {
        int64_t pool_size = 1;
        int64_t valid_size = 1;
        [ForceUnroll] for (int a = 0; a < 3; a++)
        {
            int start = o[a] * g.stride[a] - g.pad[a];
            int end = min(start + g.kernel[a], g.in[a] + g.pad[a]);
            pool_size *= end - start;
            lo[a] = max(start, 0);
            hi[a] = min(end, g.in[a]);
            valid_size *= hi[a] - lo[a];
        }
        T.Accum sum = T.Accum.ext_zero();
        for (int d = lo[0]; d < hi[0]; d++)
            for (int h = lo[1]; h < hi[1]; h++)
                for (int w = lo[2]; w < hi[2]; w++)
                    sum = T.Accum.ext_add(sum, T.load(pcs.input, uint64_t(base + int64_t(d) * g.in[1] * g.in[2] + int64_t(h) * g.in[2] + w)).toAccum());
        int64_t divisor = (g.divisor_override != 0) ? int64_t(g.divisor_override) : (g.count_include_pad ? pool_size : valid_size);
        T.store(pcs.output, idx, T.fromAccum(T.Accum.ext_div(sum, castValue<int64_t, T.Accum>(divisor))));
        return;
    }

    // divided by each window extent in turn, as the CPU
    T.Accum sum = T.Accum.ext_zero();
    for (int d = lo[0]; d < hi[0]; d++)
        for (int h = lo[1]; h < hi[1]; h++)
            for (int w = lo[2]; w < hi[2]; w++)
                sum = T.Accum.ext_add(sum, T.load(pcs.input, uint64_t(base + int64_t(d) * g.in[1] * g.in[2] + int64_t(h) * g.in[2] + w)).toAccum());
    [ForceUnroll] for (int a = 0; a < 3; a++) sum = T.Accum.ext_div(sum, castValue<int64_t, T.Accum>(int64_t(hi[a] - lo[a])));
    T.store(pcs.output, idx, T.fromAccum(sum));
}

// summed in output order, as the CPU's loop
void poolBackward<T : ITorchVulkan>(uint64_t idx, PoolPushConstants pcs, Geometry g)
{
    int64_t inputs = int64_t(g.in[0]) * g.in[1] * g.in[2];
    int64_t plane = int64_t(idx) / inputs;
    int64_t rest = int64_t(idx) % inputs;
    int i[3];
    i[0] = int(rest / (int64_t(g.in[1]) * g.in[2]));
    i[1] = int((rest / g.in[2]) % g.in[1]);
    i[2] = int(rest % g.in[2]);

    int64_t base = plane * int64_t(g.out[0]) * g.out[1] * g.out[2];

    int lo[3];
    int hi[3];
    [ForceUnroll] for (int a = 0; a < 3; a++)
    {
        if (MODE == 5)
        {
            int first = i[a] + g.pad[a] - g.kernel[a] + 1;
            lo[a] = max(0, first <= 0 ? 0 : (first + g.stride[a] - 1) / g.stride[a]);
            hi[a] = min(g.out[a], (i[a] + g.pad[a]) / g.stride[a] + 1);
        }
        else
        {
            lo[a] = max(0, int((int64_t(i[a]) * g.out[a]) / g.in[a]) - 1);
            hi[a] = min(g.out[a], int((int64_t(i[a] + 1) * g.out[a] + g.in[a] - 1) / g.in[a]) + 1);
        }
    }

    T acc = T.ext_zero();
    for (int d = lo[0]; d < hi[0]; d++)
    {
        for (int h = lo[1]; h < hi[1]; h++)
        {
            for (int w = lo[2]; w < hi[2]; w++)
            {
                int o[3] = { d, h, w };
                bool covers = true;
                int64_t pool_size = 1;
                int64_t valid_size = 1;
                [ForceUnroll] for (int a = 0; a < 3; a++)
                {
                    int start;
                    int end;
                    if (MODE == 5)
                    {
                        start = o[a] * g.stride[a] - g.pad[a];
                        end = min(start + g.kernel[a], g.in[a] + g.pad[a]);
                        pool_size *= end - start;
                        start = max(start, 0);
                        end = min(end, g.in[a]);
                        valid_size *= end - start;
                    }
                    else
                    {
                        start = adaptiveStart(o[a], g.in[a], g.out[a]);
                        end = adaptiveEnd(o[a], g.in[a], g.out[a]);
                    }
                    if (i[a] < start || i[a] >= end) covers = false;
                }
                if (!covers) continue;

                T grad = T.load(pcs.input, uint64_t(base + int64_t(d) * g.out[1] * g.out[2] + int64_t(h) * g.out[2] + w));
                if (MODE == 5)
                {
                    int64_t divisor = (g.divisor_override != 0) ? int64_t(g.divisor_override) : (g.count_include_pad ? pool_size : valid_size);
                    acc = T.ext_add(acc, T.ext_div(grad, castValue<int64_t, T>(divisor)));
                }
                else
                {
                    [ForceUnroll] for (int a = 0; a < 3; a++)
                    {
                        int extent = adaptiveEnd(o[a], g.in[a], g.out[a]) - adaptiveStart(o[a], g.in[a], g.out[a]);
                        grad = T.ext_div(grad, castValue<int64_t, T>(int64_t(extent)));
                    }
                    acc = T.ext_add(acc, grad);
                }
            }
        }
    }
    T.store(pcs.output, idx, acc);
}

void poolKernel<T : ITorchVulkan>(uint3 gi, PoolPushConstants pcs)
{
    uint64_t idx = uint64_t(flat_thread(gi, WG_x));
    if (idx >= pcs.count) return;
    Geometry g = loadGeometry(pcs.metadata);
    if (MODE == 5 || MODE == 6) poolBackward<T>(idx, pcs, g);
    else poolForward<T>(idx, pcs, g);
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void pool_main<T : ITorchVulkan>(
    uint3 gi : SV_DispatchThreadID,
    uniform PoolPushConstants pcs)
{
    poolKernel<T>(gi, pcs);
}
