module grid_sample;

import core.indexing;

[[vk::constant_id(0)]] public const uint MODE = 0;
[[vk::constant_id(1)]] public const uint DIMS = 2;
[[vk::constant_id(2)]] public const uint INTERPOLATION = 0;
[[vk::constant_id(3)]] public const uint PADDING = 0;
[[vk::constant_id(4)]] public const uint ALIGN_CORNERS = 0;
[[vk::constant_id(5)]] public const uint WG_x = 0;

public struct Metadata
{
    public uint64_t params;
}

public struct GridSamplePushConstants
{
    public uint64_t input;
    public uint64_t grid;
    public uint64_t output;
    public uint64_t grad_output;
    public uint64_t taps;
    public uint64_t weights;
    public uint64_t starts;
    public uint64_t metadata;
    public uint64_t count;
}

// as ATen/native/GridSampler.h

int toInt<T : __BuiltinFloatingPointType>(T x) { return int(trunc(x).toFloat()); }

T unnormalize<T : __BuiltinFloatingPointType>(T coord, int size, out T grad)
{
    if (ALIGN_CORNERS != 0)
    {
        grad = T(size - 1) / T(2);
        return ((coord + T(1)) / T(2)) * T(size - 1);
    }
    grad = T(size) / T(2);
    return ((coord + T(1)) * T(size) - T(1)) / T(2);
}

T clipCoordinates<T : __BuiltinFloatingPointType>(T x, int size, out T grad)
{
    if (x <= T(0)) { grad = T(0); return T(0); }
    T limit = T(size - 1);
    if (x >= limit) { grad = T(0); return limit; }
    grad = T(1);
    return x;
}

T reflectCoordinates<T : __BuiltinFloatingPointType>(T x, int twice_low, int twice_high, out T grad)
{
    if (twice_low == twice_high) { grad = T(0); return T(0); }
    T lo = T(twice_low) / T(2);
    T span = T(twice_high - twice_low) / T(2);
    x = x - lo;
    T sign = T(1);
    if (x < T(0)) { sign = T(-1); x = -x; }
    T extra = fmod(x, span);
    int flips = toInt<T>(floor(x / span));
    if (flips % 2 == 0) { grad = sign; return extra + lo; }
    grad = -sign;
    return span - extra + lo;
}

T applyPadding<T : __BuiltinFloatingPointType>(T x, int size, out T grad)
{
    grad = T(1);
    if (PADDING == 1) return clipCoordinates<T>(x, size, grad);
    if (PADDING == 2)
    {
        T reflect_grad;
        T clip_grad;
        x = (ALIGN_CORNERS != 0) ? reflectCoordinates<T>(x, 0, 2 * (size - 1), reflect_grad) : reflectCoordinates<T>(x, -1, 2 * size - 1, reflect_grad);
        x = clipCoordinates<T>(x, size, clip_grad);
        grad = reflect_grad * clip_grad;
    }
    return x;
}

T sourceIndex<T : __BuiltinFloatingPointType>(T coord, int size, out T grad)
{
    T unnormalize_grad;
    T padding_grad;
    T x = unnormalize<T>(coord, size, unnormalize_grad);
    x = applyPadding<T>(x, size, padding_grad);
    grad = unnormalize_grad * padding_grad;
    return x;
}

// round half to even, as std::nearbyint does
T roundEven<T : __BuiltinFloatingPointType>(T x)
{
    T r = floor(x);
    T d = x - r;
    if (d > T(0.5)) return r + T(1);
    if (d < T(0.5)) return r;
    return (fmod(r, T(2)) == T(0)) ? r : r + T(1);
}

void cubicCoefficients<T : __BuiltinFloatingPointType>(T t, out T c[4])
{
    T a = T(-0.75);
    T x1 = t;
    c[0] = ((a * (x1 + T(1)) - T(5) * a) * (x1 + T(1)) + T(8) * a) * (x1 + T(1)) - T(4) * a;
    c[1] = ((a + T(2)) * x1 - (a + T(3))) * x1 * x1 + T(1);
    T x2 = T(1) - t;
    c[2] = ((a + T(2)) * x2 - (a + T(3))) * x2 * x2 + T(1);
    c[3] = ((a * (x2 + T(1)) - T(5) * a) * (x2 + T(1)) + T(8) * a) * (x2 + T(1)) - T(4) * a;
}

void cubicCoefficientsGrad<T : __BuiltinFloatingPointType>(T t, out T c[4])
{
    T a = T(-0.75);
    T x = -T(1) - t;
    c[0] = (-T(3) * a * x - T(10) * a) * x - T(8) * a;
    x = -t;
    c[1] = (-T(3) * (a + T(2)) * x - T(2) * (a + T(3))) * x;
    x = T(1) - t;
    c[2] = (T(3) * (a + T(2)) * x - T(2) * (a + T(3))) * x;
    x = T(2) - t;
    c[3] = (T(3) * a * x - T(10) * a) * x + T(8) * a;
}

int boundedTap<T : __BuiltinFloatingPointType>(int position, int size)
{
    T grad;
    T x = applyPadding<T>(T(position), size, grad);
    return toInt<T>(x);
}

struct Geometry
{
    int n;
    int c;
    int in[3];
    int out[3];
}

Geometry loadGeometry(uint64_t address)
{
    uint* p = (uint*)(*(Metadata*)address).params;
    Geometry g;
    g.n = int(p[0]);
    g.c = int(p[1]);
    [ForceUnroll] for (int a = 0; a < 3; a++)
    {
        g.in[a] = int(p[2 + a]);
        g.out[a] = int(p[5 + a]);
    }
    return g;
}

int tapCount()
{
    if (INTERPOLATION == 1) return 1;
    if (INTERPOLATION == 2) return DIMS == 3 ? 64 : 16;
    return DIMS == 3 ? 8 : 4;
}

// one axis of a 3-D bicubic sample, as resolve_cubic_taps: taps sit around the unclipped index, and one the padding drops is -1
void cubicAxis<T : __BuiltinFloatingPointType>(T coord, int size, out T multiplier, out int index[4], out T coeff[4], out T coeff_grad[4])
{
    T ix = unnormalize<T>(coord, size, multiplier);
    T base = floor(ix);
    cubicCoefficients<T>(ix - base, coeff);
    cubicCoefficientsGrad<T>(ix - base, coeff_grad);
    [ForceUnroll] for (int i = 0; i < 4; i++)
    {
        T grad;
        T tap = applyPadding<T>(base + T(i - 1), size, grad);
        index[i] = (tap >= T(0) && tap < T(size)) ? toInt<T>(tap) : -1;
    }
}

void tricubicKernel<T : __BuiltinFloatingPointType>(int64_t idx, int64_t n, int64_t o, T coord[3], Geometry g, GridSamplePushConstants pcs)
{
    int64_t outputs = int64_t(g.out[0]) * g.out[1] * g.out[2];
    int64_t inputs = int64_t(g.in[0]) * g.in[1] * g.in[2];
    T mx;
    T my;
    T mz;
    int xi[4];
    int yi[4];
    int zi[4];
    T xc[4];
    T yc[4];
    T zc[4];
    T xg[4];
    T yg[4];
    T zg[4];
    cubicAxis<T>(coord[2], g.in[2], mx, xi, xc, xg);
    cubicAxis<T>(coord[1], g.in[1], my, yi, yc, yg);
    cubicAxis<T>(coord[0], g.in[0], mz, zi, zc, zg);

    if (MODE == 1)
    {
        int64_t* tap_out = (int64_t*)pcs.taps;
        T* weight_out = (T*)pcs.weights;
        for (int k = 0; k < 4; k++)
        {
            for (int j = 0; j < 4; j++)
            {
                bool plane_reads = zi[k] >= 0 && yi[j] >= 0;
                int64_t row = (int64_t(zi[k]) * g.in[1] + yi[j]) * g.in[2];
                for (int i = 0; i < 4; i++)
                {
                    int64_t e = idx * 64 + int64_t((k * 4 + j) * 4 + i);
                    tap_out[e] = (plane_reads && xi[i] >= 0) ? row + int64_t(xi[i]) : int64_t(-1);
                    weight_out[e] = xc[i] * yc[j] * zc[k];
                }
            }
        }
        return;
    }

    T* input = (T*)pcs.input;
    if (MODE == 0)
    {
        T* output = (T*)pcs.output;
        for (int c = 0; c < g.c; c++)
        {
            T* plane = input + (n * g.c + c) * inputs;
            T value = T(0);
            for (int k = 0; k < 4; k++)
            {
                for (int j = 0; j < 4; j++)
                {
                    T weight_zy = zc[k] * yc[j];
                    bool plane_reads = zi[k] >= 0 && yi[j] >= 0;
                    int64_t row = (int64_t(zi[k]) * g.in[1] + yi[j]) * g.in[2];
                    for (int i = 0; i < 4; i++)
                    {
                        T sample = (plane_reads && xi[i] >= 0) ? plane[row + xi[i]] : T(0);
                        value += sample * weight_zy * xc[i];
                    }
                }
            }
            output[(n * g.c + c) * outputs + o] = value;
        }
        return;
    }

    T* grad_output = (T*)pcs.grad_output;
    T* grad_grid = (T*)pcs.output;
    T gx = T(0);
    T gy = T(0);
    T gz = T(0);
    for (int c = 0; c < g.c; c++)
    {
        T* plane = input + (n * g.c + c) * inputs;
        T go = grad_output[(n * g.c + c) * outputs + o];
        for (int k = 0; k < 4; k++)
        {
            for (int j = 0; j < 4; j++)
            {
                bool plane_reads = zi[k] >= 0 && yi[j] >= 0;
                int64_t row = (int64_t(zi[k]) * g.in[1] + yi[j]) * g.in[2];
                for (int i = 0; i < 4; i++)
                {
                    T v = (plane_reads && xi[i] >= 0) ? plane[row + xi[i]] : T(0);
                    gx -= v * xg[i] * yc[j] * zc[k] * go;
                    gy -= v * xc[i] * yg[j] * zc[k] * go;
                    gz -= v * xc[i] * yc[j] * zg[k] * go;
                }
            }
        }
    }
    int64_t base = (n * outputs + o) * 3;
    grad_grid[base] = mx * gx;
    grad_grid[base + 1] = my * gy;
    grad_grid[base + 2] = mz * gz;
}

// in the CPU's corner order
void computeTaps<T : __BuiltinFloatingPointType>(T coord[3], Geometry g, out int position[16], out T weight[16])
{
    [ForceUnroll] for (int k = 0; k < 16; k++) { position[k] = -1; weight[k] = T(0); }
    int in_w = g.in[2];
    int in_h = g.in[1];
    int in_d = g.in[0];

    if (INTERPOLATION == 2)
    {
        T unused;
        T ix = unnormalize<T>(coord[2], in_w, unused);
        T iy = unnormalize<T>(coord[1], in_h, unused);
        T x0 = floor(ix);
        T y0 = floor(iy);
        T cx[4];
        T cy[4];
        cubicCoefficients<T>(ix - x0, cx);
        cubicCoefficients<T>(iy - y0, cy);
        for (int j = 0; j < 4; j++)
        {
            for (int i = 0; i < 4; i++)
            {
                int x = boundedTap<T>(toInt<T>(x0) - 1 + i, in_w);
                int y = boundedTap<T>(toInt<T>(y0) - 1 + j, in_h);
                bool inside = x >= 0 && x < in_w && y >= 0 && y < in_h;
                position[j * 4 + i] = inside ? y * in_w + x : -1;
                weight[j * 4 + i] = cx[i] * cy[j];
            }
        }
        return;
    }

    T grad;
    T ix = sourceIndex<T>(coord[2], in_w, grad);
    T iy = sourceIndex<T>(coord[1], in_h, grad);
    T iz = (DIMS == 3) ? sourceIndex<T>(coord[0], in_d, grad) : T(0);

    if (INTERPOLATION == 1)
    {
        int x = toInt<T>(roundEven<T>(ix));
        int y = toInt<T>(roundEven<T>(iy));
        int z = toInt<T>(roundEven<T>(iz));
        bool inside = x >= 0 && x < in_w && y >= 0 && y < in_h && z >= 0 && z < in_d;
        position[0] = inside ? (z * in_h + y) * in_w + x : -1;
        weight[0] = T(1);
        return;
    }

    T x0 = floor(ix);
    T y0 = floor(iy);
    T z0 = floor(iz);
    T dx[2] = { x0 + T(1) - ix, ix - x0 };
    T dy[2] = { y0 + T(1) - iy, iy - y0 };
    T dz[2] = { z0 + T(1) - iz, iz - z0 };
    int depth = (DIMS == 3) ? 2 : 1;
    for (int k = 0; k < depth; k++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int i = 0; i < 2; i++)
            {
                int x = toInt<T>(x0) + i;
                int y = toInt<T>(y0) + j;
                int z = toInt<T>(z0) + k;
                bool inside = x >= 0 && x < in_w && y >= 0 && y < in_h && z >= 0 && z < in_d;
                int tap = (k * 2 + j) * 2 + i;
                position[tap] = inside ? (z * in_h + y) * in_w + x : -1;
                weight[tap] = dx[i] * dy[j] * ((DIMS == 3) ? dz[k] : T(1));
            }
        }
    }
}

// summed in output order, as the CPU
void inputGradKernel<T : __BuiltinFloatingPointType>(uint64_t idx, Geometry g, GridSamplePushConstants pcs)
{
    int64_t outputs = int64_t(g.out[0]) * g.out[1] * g.out[2];
    int64_t inputs = int64_t(g.in[0]) * g.in[1] * g.in[2];
    int64_t taps = int64_t(tapCount());
    int64_t n = int64_t(idx) / (int64_t(g.c) * inputs);
    int64_t c = (int64_t(idx) / inputs) % int64_t(g.c);
    int64_t i = int64_t(idx) % inputs;

    int64_t* order = (int64_t*)pcs.taps;
    int64_t* starts = (int64_t*)pcs.starts;
    T* weights = (T*)pcs.weights;
    T* grad_output = (T*)pcs.grad_output + (n * g.c + c) * outputs;
    T acc = T(0);
    for (int64_t j = starts[n * inputs + i]; j < starts[n * inputs + i + 1]; j++)
    {
        int64_t e = order[j];
        acc += weights[e] * grad_output[(e / taps) % outputs];
    }
    ((T*)pcs.output)[idx] = acc;
}

void gridSampleKernel<T : __BuiltinFloatingPointType>(uint3 gi, GridSamplePushConstants pcs)
{
    uint64_t idx = uint64_t(flat_thread(gi, WG_x));
    if (idx >= pcs.count) return;
    Geometry g = loadGeometry(pcs.metadata);
    if (MODE == 3)
    {
        inputGradKernel<T>(idx, g, pcs);
        return;
    }

    int64_t outputs = int64_t(g.out[0]) * g.out[1] * g.out[2];
    int64_t inputs = int64_t(g.in[0]) * g.in[1] * g.in[2];
    int64_t n = int64_t(idx) / outputs;
    int64_t o = int64_t(idx) % outputs;

    int64_t dims = int64_t(int(DIMS));
    T* grid = (T*)pcs.grid;
    T coord[3];
    coord[0] = T(0);
    [ForceUnroll] for (int a = 0; a < 3; a++)
    {
        if (a >= 3 - int(dims)) coord[a] = grid[(n * outputs + o) * dims + int64_t(2 - a)];
    }
    if (DIMS == 3 && INTERPOLATION == 2)
    {
        tricubicKernel<T>(int64_t(idx), n, o, coord, g, pcs);
        return;
    }

    int position[16];
    T weight[16];
    computeTaps<T>(coord, g, position, weight);
    int taps = tapCount();

    if (MODE == 1)
    {
        int64_t* tap_out = (int64_t*)pcs.taps;
        T* weight_out = (T*)pcs.weights;
        for (int k = 0; k < taps; k++)
        {
            tap_out[int64_t(idx) * int64_t(taps) + int64_t(k)] = int64_t(position[k]);
            weight_out[int64_t(idx) * int64_t(taps) + int64_t(k)] = weight[k];
        }
        return;
    }

    T* input = (T*)pcs.input;
    if (MODE == 0)
    {
        T* output = (T*)pcs.output;
        for (int c = 0; c < g.c; c++)
        {
            T* plane = input + (n * g.c + c) * inputs;
            T acc = T(0);
            if (INTERPOLATION == 2)
            {
                T coefficients[4];
                T unused;
                T iy = unnormalize<T>(coord[1], g.in[1], unused);
                T cy[4];
                cubicCoefficients<T>(iy - floor(iy), cy);
                for (int j = 0; j < 4; j++)
                {
                    T row = T(0);
                    T ix = unnormalize<T>(coord[2], g.in[2], unused);
                    T cx[4];
                    cubicCoefficients<T>(ix - floor(ix), cx);
                    for (int i = 0; i < 4; i++)
                    {
                        int p = position[j * 4 + i];
                        T v = (p >= 0) ? plane[p] : T(0);
                        row = (i == 0) ? v * cx[i] : row + v * cx[i];
                    }
                    coefficients[j] = row;
                }
                acc = coefficients[0] * cy[0] + coefficients[1] * cy[1] + coefficients[2] * cy[2] + coefficients[3] * cy[3];
            }
            else
            {
                for (int k = 0; k < taps; k++)
                {
                    if (position[k] >= 0) acc += plane[position[k]] * weight[k];
                }
            }
            output[(n * g.c + c) * outputs + o] = acc;
        }
        return;
    }

    T* grad_output = (T*)pcs.grad_output;
    T* grad_grid = (T*)pcs.output;
    T multiplier[3];
    T unused;
    T ix;
    T iy;
    T iz = T(0);
    if (INTERPOLATION == 2)
    {
        ix = unnormalize<T>(coord[2], g.in[2], multiplier[2]);
        iy = unnormalize<T>(coord[1], g.in[1], multiplier[1]);
    }
    else
    {
        ix = sourceIndex<T>(coord[2], g.in[2], multiplier[2]);
        iy = sourceIndex<T>(coord[1], g.in[1], multiplier[1]);
        if (DIMS == 3) iz = sourceIndex<T>(coord[0], g.in[0], multiplier[0]);
    }

    T gx = T(0);
    T gy = T(0);
    T gz = T(0);
    if (INTERPOLATION != 1)
    {
        T x0 = floor(ix);
        T y0 = floor(iy);
        T z0 = floor(iz);
        T cx[4];
        T cy[4];
        T dcx[4];
        T dcy[4];
        if (INTERPOLATION == 2)
        {
            cubicCoefficients<T>(ix - x0, cx);
            cubicCoefficients<T>(iy - y0, cy);
            cubicCoefficientsGrad<T>(ix - x0, dcx);
            cubicCoefficientsGrad<T>(iy - y0, dcy);
        }
        for (int c = 0; c < g.c; c++)
        {
            T* plane = input + (n * g.c + c) * inputs;
            T go = grad_output[(n * g.c + c) * outputs + o];
            if (INTERPOLATION == 2)
            {
                for (int j = 0; j < 4; j++)
                {
                    for (int i = 0; i < 4; i++)
                    {
                        int p = position[j * 4 + i];
                        T v = (p >= 0) ? plane[p] : T(0);
                        gx -= v * dcx[i] * cy[j] * go;
                        gy -= v * cx[i] * dcy[j] * go;
                    }
                }
            }
            else
            {
                T dx[2] = { x0 + T(1) - ix, ix - x0 };
                T dy[2] = { y0 + T(1) - iy, iy - y0 };
                T dz[2] = { z0 + T(1) - iz, iz - z0 };
                int depth = (DIMS == 3) ? 2 : 1;
                for (int k = 0; k < depth; k++)
                {
                    for (int j = 0; j < 2; j++)
                    {
                        for (int i = 0; i < 2; i++)
                        {
                            int p = position[(k * 2 + j) * 2 + i];
                            if (p < 0) continue;
                            T v = plane[p] * go;
                            T sx = (i == 0) ? T(-1) : T(1);
                            T sy = (j == 0) ? T(-1) : T(1);
                            T sz = (k == 0) ? T(-1) : T(1);
                            T wz = (DIMS == 3) ? dz[k] : T(1);
                            gx += v * sx * dy[j] * wz;
                            gy += v * sy * dx[i] * wz;
                            if (DIMS == 3) gz += v * sz * dx[i] * dy[j];
                        }
                    }
                }
            }
        }
    }
    int64_t base = (n * outputs + o) * dims;
    grad_grid[base] = gx * multiplier[2];
    grad_grid[base + 1] = gy * multiplier[1];
    if (DIMS == 3) grad_grid[base + 2] = gz * multiplier[0];
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void grid_sample_main<T : __BuiltinFloatingPointType>(
    uint3 gi : SV_DispatchThreadID,
    uniform GridSamplePushConstants pcs)
{
    gridSampleKernel<T>(gi, pcs);
}
