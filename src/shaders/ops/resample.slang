module resample;

import core.indexing;

[[vk::constant_id(0)]] public const uint MODE = 0;
[[vk::constant_id(1)]] public const uint BACKWARD = 0;
[[vk::constant_id(2)]] public const uint ALIGN_CORNERS = 0;
[[vk::constant_id(3)]] public const uint WG_x = 0;

public struct ResamplePushConstants<T>
{
    public uint64_t input;
    public uint64_t output;
    public uint64_t count;
    public uint in_length;
    public uint out_length;
    public uint inner;
    public uint padding;
    public T scale;
    public T support;
}

// a whole-valued T is exact in float32 at index sizes, which bridges the generic conversion to int
int truncToInt<T : __BuiltinFloatingPointType>(T x) { return int(trunc(x).toFloat()); }
int floorToInt<T : __BuiltinFloatingPointType>(T x) { return int(floor(x).toFloat()); }

// trunc(x + 0.5) with the half added exactly, as the CPU's double arithmetic does
int truncHalfUp<T : __BuiltinFloatingPointType>(T x)
{
    if (x < T(0)) return truncToInt<T>(x + T(0.5));
    T whole = trunc(x);
    return truncToInt<T>(whole) + ((x - whole >= T(0.5)) ? 1 : 0);
}

// (d + 0.5) * scale rounded once, as the CPU's double: the sum's rounding error is carried through a fused multiply-add
T halfShiftedProduct<T : __BuiltinFloatingPointType>(T d, T scale)
{
    if (sizeof(T) == 8) return (d + T(0.5)) * scale;
    T hi = d + T(0.5);
    T lo = (abs(d) >= T(0.5)) ? (T(0.5) - (hi - d)) : (d - (hi - T(0.5)));
    return fma(hi, scale, lo * scale);
}

T cubicConvolution1<T : __BuiltinFloatingPointType>(T x, T a) { return ((a + T(2)) * x - (a + T(3))) * x * x + T(1); }
T cubicConvolution2<T : __BuiltinFloatingPointType>(T x, T a) { return ((a * x - T(5) * a) * x + T(8) * a) * x - T(4) * a; }

T sourceIndex<T : __BuiltinFloatingPointType>(T scale, int o, bool cubic)
{
    if (ALIGN_CORNERS != 0) return scale * T(o);
    T src = scale * (T(o) + T(0.5)) - T(0.5);
    return (!cubic && src < T(0)) ? T(0) : src;
}

T aaFilter<T : __BuiltinFloatingPointType>(T x)
{
    x = abs(x);
    if (MODE == 4) return x < T(1) ? T(1) - x : T(0);
    if (MODE == 6)
    {
        // Lanczos, a = 3
        if (x >= T(3)) return T(0);
        if (x == T(0)) return T(1);
        T px = T(3.14159265358979) * x;
        return T(3) * sin(px) * sin(px / T(3)) / (px * px);
    }
    if (x < T(1)) return cubicConvolution1<T>(x, T(-0.5));
    if (x < T(2)) return cubicConvolution2<T>(x, T(-0.5));
    return T(0);
}

struct Taps<T : __BuiltinFloatingPointType>
{
    int count;
    int first;
    int in_length;
    T t;
    T center;
    T invscale;
    T total;

    int index(int j)
    {
        if (MODE == 3) return clamp(first + j - 1, 0, in_length - 1);
        if (MODE == 2 && j == 1) return first + ((first < in_length - 1) ? 1 : 0);
        return first + j;
    }

    T weight(int j)
    {
        switch (MODE)
        {
        case 2:
            return (j == 0) ? T(1) - t : t;
        case 3:
        {
            T a = T(-0.75);
            if (j == 0) return cubicConvolution2<T>(t + T(1), a);
            if (j == 1) return cubicConvolution1<T>(t, a);
            if (j == 2) return cubicConvolution1<T>(T(1) - t, a);
            return cubicConvolution2<T>(T(1) - t + T(1), a);
        }
        case 4:
        case 5:
        case 6:
        {
            T w = aaFilter<T>(halfShiftedProduct<T>(T(j + first) - center, invscale));
            return (total != T(0)) ? w / total : w;
        }
        default:
            return T(1);
        }
    }

    int lowest() { return index(0); }
    int highest() { return index(count - 1); }
}

Taps<T> computeTaps<T : __BuiltinFloatingPointType>(int o, int in_length, int out_length, T scale, T support)
{
    Taps<T> taps;
    taps.in_length = in_length;
    taps.t = T(0);
    taps.center = T(0);
    taps.invscale = T(1);
    taps.total = T(0);
    if (MODE == 0 || MODE == 1)
    {
        T real = (MODE == 0) ? scale * T(o) : scale * (T(o) + T(0.5));
        taps.count = 1;
        taps.first = min(floorToInt<T>(real), in_length - 1);
    }
    else if (MODE == 2)
    {
        taps.count = 2;
        if (in_length == out_length)
        {
            // index(1) would step right, but an identity resample keeps both taps on o
            taps.count = 1;
            taps.first = o;
        }
        else
        {
            T real = sourceIndex<T>(scale, o, false);
            taps.first = min(floorToInt<T>(real), in_length - 1);
            taps.t = min(max(real - T(taps.first), T(0)), T(1));
        }
    }
    else if (MODE == 3)
    {
        T real = sourceIndex<T>(scale, o, true);
        taps.count = 4;
        taps.first = min(floorToInt<T>(real), in_length - 1);
        taps.t = min(max(real - T(taps.first), T(0)), T(1));
    }
    else
    {
        taps.center = scale * (T(o) + T(0.5));
        taps.invscale = (scale >= T(1)) ? T(1) / scale : T(1);
        taps.first = max(truncHalfUp<T>(taps.center - support), 0);
        taps.count = max(min(truncHalfUp<T>(taps.center + support), in_length) - taps.first, 0);
        for (int j = 0; j < taps.count; j++) taps.total += aaFilter<T>(halfShiftedProduct<T>(T(j + taps.first) - taps.center, taps.invscale));
    }
    return taps;
}

void resampleKernel<T : __BuiltinFloatingPointType>(uint3 gi, ResamplePushConstants<T> pcs)
{
    uint64_t idx = uint64_t(flat_thread(gi, WG_x));
    if (idx >= pcs.count) return;

    T* input = (T*)pcs.input;
    T* output = (T*)pcs.output;
    int in_length = int(pcs.in_length);
    int out_length = int(pcs.out_length);
    uint64_t inner = uint64_t(pcs.inner);

    if (BACKWARD == 0)
    {
        uint64_t outer = idx / (uint64_t(out_length) * inner);
        int o = int((idx / inner) % uint64_t(out_length));
        uint64_t k = idx % inner;
        uint64_t base = outer * uint64_t(in_length) * inner + k;

        Taps<T> taps = computeTaps<T>(o, in_length, out_length, pcs.scale, pcs.support);
        T acc = T(0);
        if (taps.count > 0) acc = input[base + uint64_t(taps.index(0)) * inner] * taps.weight(0);
        for (int j = 1; j < taps.count; j++) acc += input[base + uint64_t(taps.index(j)) * inner] * taps.weight(j);
        output[idx] = acc;
        return;
    }

    uint64_t outer = idx / (uint64_t(in_length) * inner);
    int i = int((idx / inner) % uint64_t(in_length));
    uint64_t k = idx % inner;
    uint64_t base = outer * uint64_t(out_length) * inner + k;

    int lo = 0;
    int hi = out_length;
    while (lo < hi)
    {
        int mid = (lo + hi) / 2;
        if (computeTaps<T>(mid, in_length, out_length, pcs.scale, pcs.support).highest() < i) lo = mid + 1;
        else hi = mid;
    }
    int first = lo;
    hi = out_length;
    while (lo < hi)
    {
        int mid = (lo + hi) / 2;
        if (computeTaps<T>(mid, in_length, out_length, pcs.scale, pcs.support).lowest() <= i) lo = mid + 1;
        else hi = mid;
    }

    T acc = T(0);
    for (int o = first; o < lo; o++)
    {
        Taps<T> taps = computeTaps<T>(o, in_length, out_length, pcs.scale, pcs.support);
        T g = input[base + uint64_t(o) * inner];
        for (int j = 0; j < taps.count; j++)
        {
            if (taps.index(j) == i) acc += taps.weight(j) * g;
        }
    }
    output[idx] = acc;
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void resample_main<T : __BuiltinFloatingPointType>(
    uint3 gi : SV_DispatchThreadID,
    uniform ResamplePushConstants<T> pcs)
{
    resampleKernel<T>(gi, pcs);
}
