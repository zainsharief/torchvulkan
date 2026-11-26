module jacobi;

[[vk::constant_id(0)]] public const uint MODE = 0;
[[vk::constant_id(1)]] public const uint WG_x = 0;
[[vk::constant_id(2)]] public const uint COMPLEX = 0;

static const int MAX_SWEEPS = 60;

public struct JacobiPushConstants
{
    public uint64_t a;
    public uint64_t v;
    public uint64_t rotations;
    public uint m;
    public uint n;
    public uint pairs;
}

T epsilon<T : __BuiltinFloatingPointType>() { return (sizeof(T) == 8) ? T(2.220446049250313e-16) : T(1.1920928955078125e-07); }

void roundPair(int round, int i, int players, out int p, out int q)
{
    int left = (i == 0) ? 0 : 1 + (round + i - 1) % (players - 1);
    int j = players - 1 - i;
    int right = 1 + (round + j - 1) % (players - 1);
    p = min(left, right);
    q = max(left, right);
}

// (cos, sin) of the phase of x^H y, which a complex rotation divides out of y
bool computeRotation<T : __BuiltinFloatingPointType>(T* a, int m, int n, int p, int q, out T c, out T s, out T phase_re, out T phase_im)
{
    c = T(1);
    s = T(0);
    phase_re = T(1);
    phase_im = T(0);
    T app;
    T aqq;
    T apq;
    if (MODE == 0 && COMPLEX != 0)
    {
        app = T(0);
        aqq = T(0);
        T dot_re = T(0);
        T dot_im = T(0);
        for (int i = 0; i < m; i++)
        {
            int64_t xi = (int64_t(i) * n + p) * 2;
            int64_t yi = (int64_t(i) * n + q) * 2;
            T xr = a[xi], xm = a[xi + 1], yr = a[yi], ym = a[yi + 1];
            app += xr * xr + xm * xm;
            aqq += yr * yr + ym * ym;
            dot_re += xr * yr + xm * ym;
            dot_im += xr * ym - xm * yr;
        }
        apq = sqrt(dot_re * dot_re + dot_im * dot_im);
        if (apq == T(0) || apq <= epsilon<T>() * sqrt(app * aqq)) return false;
        phase_re = dot_re / apq;
        phase_im = dot_im / apq;
    }
    else if (MODE == 0)
    {
        app = T(0);
        aqq = T(0);
        apq = T(0);
        for (int i = 0; i < m; i++)
        {
            T x = a[i * n + p];
            T y = a[i * n + q];
            app += x * x;
            aqq += y * y;
            apq += x * y;
        }
        if (apq == T(0) || abs(apq) <= epsilon<T>() * sqrt(app * aqq)) return false;
    }
    else
    {
        app = a[p * n + p];
        aqq = a[q * n + q];
        apq = a[p * n + q];
        if (apq == T(0) || abs(apq) <= epsilon<T>() * sqrt(abs(app) * abs(aqq)) * T(0.5)) return false;
    }
    T zeta = (aqq - app) / (T(2) * apq);
    T t = ((zeta >= T(0)) ? T(1) : T(-1)) / (abs(zeta) + sqrt(T(1) + zeta * zeta));
    c = T(1) / sqrt(T(1) + t * t);
    s = c * t;
    return true;
}

void applyRotations<T : __BuiltinFloatingPointType>(T* data, T* rotations, uint lane, int pairs, int round, int players, int length, int line_stride,
    int element_stride, int n)
{
    int items = pairs * length;
    for (int base = 0; base < items; base += int(WG_x))
    {
        int idx = base + int(lane);
        T x = T(0);
        T y = T(0);
        T c = T(1);
        T s = T(0);
        T xi = T(0);
        T yi = T(0);
        T ph_re = T(1);
        T ph_im = T(0);
        int64_t px = 0;
        int64_t qx = 0;
        bool active = false;
        if (idx < items)
        {
            int pair = idx / length;
            int e = idx % length;
            int p;
            int q;
            roundPair(round, pair, players, p, q);
            c = rotations[pair * 4];
            s = rotations[pair * 4 + 1];
            active = q < n && s != T(0);
            px = int64_t(p) * line_stride + int64_t(e) * element_stride;
            qx = int64_t(q) * line_stride + int64_t(e) * element_stride;
            if (active && COMPLEX == 0)
            {
                x = data[px];
                y = data[qx];
            }
            else if (active)
            {
                ph_re = rotations[pair * 4 + 2];
                ph_im = rotations[pair * 4 + 3];
                x = data[px * 2];
                xi = data[px * 2 + 1];
                y = data[qx * 2];
                yi = data[qx * 2 + 1];
            }
        }
        AllMemoryBarrierWithGroupSync();
        // one active check around both stores: AMD's 22.20 driver stores from inactive lanes (to data[0]) when the
        // complex case is an `else if (active)`
        if (active)
        {
            if (COMPLEX == 0)
            {
                data[px] = c * x - s * y;
                data[qx] = s * x + c * y;
            }
            else
            {
                // x' = c x - s conj(phase) y, y' = s phase x + c y
                data[px * 2] = c * x - s * (ph_re * y + ph_im * yi);
                data[px * 2 + 1] = c * xi - s * (ph_re * yi - ph_im * y);
                data[qx * 2] = s * (ph_re * x - ph_im * xi) + c * y;
                data[qx * 2 + 1] = s * (ph_re * xi + ph_im * x) + c * yi;
            }
        }
        AllMemoryBarrierWithGroupSync();
    }
}

void jacobiKernel<T : __BuiltinFloatingPointType>(uint b, uint lane, JacobiPushConstants pcs)
{
    static groupshared int rotated;
    int m = int(pcs.m);
    int n = int(pcs.n);
    int pairs = int(pcs.pairs);
    int players = pairs * 2;
    int width = (COMPLEX != 0) ? 2 : 1;
    T* a = (T*)pcs.a + int64_t(b) * m * n * width;
    T* v = (T*)pcs.v + int64_t(b) * n * n * width;
    T* rotations = (T*)pcs.rotations + int64_t(b) * pairs * 4;

    for (int idx = int(lane); idx < n * n * width; idx += int(WG_x)) v[idx] = (idx / width / n == idx / width % n && idx % width == 0) ? T(1) : T(0);
    AllMemoryBarrierWithGroupSync();

    for (int sweep = 0; sweep < MAX_SWEEPS && players > 1; sweep++)
    {
        if (lane == 0) rotated = 0;
        AllMemoryBarrierWithGroupSync();

        for (int round = 0; round < players - 1; round++)
        {
            for (int pair = int(lane); pair < pairs; pair += int(WG_x))
            {
                int p;
                int q;
                roundPair(round, pair, players, p, q);
                T c = T(1);
                T s = T(0);
                T phase_re = T(1);
                T phase_im = T(0);
                if (q < n && computeRotation<T>(a, m, n, p, q, c, s, phase_re, phase_im)) rotated = 1;
                rotations[pair * 4] = c;
                rotations[pair * 4 + 1] = s;
                rotations[pair * 4 + 2] = phase_re;
                rotations[pair * 4 + 3] = phase_im;
            }
            AllMemoryBarrierWithGroupSync();

            if (MODE == 0)
            {
                applyRotations<T>(a, rotations, lane, pairs, round, players, m, 1, n, n);
            }
            else
            {
                applyRotations<T>(a, rotations, lane, pairs, round, players, n, n, 1, n);
                applyRotations<T>(a, rotations, lane, pairs, round, players, n, 1, n, n);
            }
            applyRotations<T>(v, rotations, lane, pairs, round, players, n, 1, n, n);
        }
        AllMemoryBarrierWithGroupSync();
        if (rotated == 0) break;
        AllMemoryBarrierWithGroupSync();
    }
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void jacobi_main<T : __BuiltinFloatingPointType>(
    uint3 group : SV_GroupID,
    uint3 thread : SV_GroupThreadID,
    uniform JacobiPushConstants pcs)
{
    jacobiKernel<T>(group.x, thread.x, pcs);
}
