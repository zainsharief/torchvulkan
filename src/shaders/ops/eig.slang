module eig;

[[vk::constant_id(0)]] public const uint WG_x = 0;

public struct EigPushConstants
{
    public uint64_t h;
    public uint64_t v;
    public uint64_t d;
    public uint64_t e;
    public uint64_t ort;
    public uint64_t count;
    public uint n;
}

T ld<T : __BuiltinFloatingPointType>(T* p, int64_t i) { return p[max(i, int64_t(0))]; }

void st<T : __BuiltinFloatingPointType>(T* p, int64_t i, T x)
{
    AllMemoryBarrierWithGroupSync();
    p[i] = x;
}

struct Work<T : __BuiltinFloatingPointType>
{
    T* h;
    T* v;
    T* d;
    T* e;
    T* ort;
    int nn;

    T H(int i, int j) { return ld<T>(h, int64_t(i) * nn + j); }
    T V(int i, int j) { return ld<T>(v, int64_t(i) * nn + j); }
    [mutating] void setH(int i, int j, T x) { st<T>(h, int64_t(i) * nn + j, x); }
    [mutating] void setV(int i, int j, T x) { st<T>(v, int64_t(i) * nn + j, x); }
}

// JAMA's cdiv
void cdiv<T : __BuiltinFloatingPointType>(T xr, T xi, T yr, T yi, out T cr, out T ci)
{
    T r;
    T dd;
    if (abs(yr) > abs(yi))
    {
        r = yi / yr;
        dd = yr + r * yi;
        cr = (xr + r * xi) / dd;
        ci = (xi - r * xr) / dd;
    }
    else
    {
        r = yr / yi;
        dd = yi + r * yr;
        cr = (r * xr + xi) / dd;
        ci = (r * xi - xr) / dd;
    }
}

// JAMA's orthes
void orthes<T : __BuiltinFloatingPointType>(inout Work<T> w)
{
    int n = w.nn;
    int low = 0;
    int high = n - 1;
    for (int m = low + 1; m <= high - 1; m++)
    {
        T scale = T(0);
        for (int i = m; i <= high; i++) scale = scale + abs(w.H(i, m - 1));
        if (scale != T(0))
        {
            T hh = T(0);
            for (int i = high; i >= m; i--)
            {
                T o = w.H(i, m - 1) / scale;
                st<T>(w.ort, i, o);
                hh += o * o;
            }
            T g = sqrt(hh);
            if (ld<T>(w.ort, m) > T(0)) g = -g;
            hh = hh - ld<T>(w.ort, m) * g;
            st<T>(w.ort, m, ld<T>(w.ort, m) - g);
            for (int j = m; j < n; j++)
            {
                T f = T(0);
                for (int i = high; i >= m; i--) f += ld<T>(w.ort, i) * w.H(i, j);
                f = f / hh;
                for (int i = m; i <= high; i++) w.setH(i, j, w.H(i, j) - f * ld<T>(w.ort, i));
            }
            for (int i = 0; i <= high; i++)
            {
                T f = T(0);
                for (int j = high; j >= m; j--) f += ld<T>(w.ort, j) * w.H(i, j);
                f = f / hh;
                for (int j = m; j <= high; j++) w.setH(i, j, w.H(i, j) - f * ld<T>(w.ort, j));
            }
            st<T>(w.ort, m, scale * ld<T>(w.ort, m));
            w.setH(m, m - 1, scale * g);
        }
    }

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) w.setV(i, j, (i == j) ? T(1) : T(0));

    for (int m = high - 1; m >= low + 1; m--)
    {
        if (w.H(m, m - 1) != T(0))
        {
            for (int i = m + 1; i <= high; i++) st<T>(w.ort, i, w.H(i, m - 1));
            for (int j = m; j <= high; j++)
            {
                T g = T(0);
                for (int i = m; i <= high; i++) g += ld<T>(w.ort, i) * w.V(i, j);
                g = (g / ld<T>(w.ort, m)) / w.H(m, m - 1);
                for (int i = m; i <= high; i++) w.setV(i, j, w.V(i, j) + g * ld<T>(w.ort, i));
            }
        }
    }
}

// JAMA's hqr2
void hqr2<T : __BuiltinFloatingPointType>(inout Work<T> w)
{
    int nn = w.nn;
    int n = nn - 1;
    int low = 0;
    int high = nn - 1;
    T eps = (sizeof(T) == 8) ? T(2.220446049250313e-16) : T(1.1920928955078125e-07);
    T exshift = T(0);
    T p = T(0), q = T(0), r = T(0), s = T(0), z = T(0), t, x, y, ww;

    T norm = T(0);
    for (int i = 0; i < nn; i++)
        for (int j = max(i - 1, 0); j < nn; j++) norm = norm + abs(w.H(i, j));

    int iter = 0;
    int total = 0;
    while (n >= low && total < 100 * nn)
    {
        total++;
        int l = n;
        while (l > low)
        {
            s = abs(w.H(l - 1, l - 1)) + abs(w.H(l, l));
            if (s == T(0)) s = norm;
            if (abs(w.H(l, l - 1)) < eps * s) break;
            l--;
        }

        if (l == n)
        {
            w.setH(n, n, w.H(n, n) + exshift);
            st<T>(w.d, n, w.H(n, n));
            st<T>(w.e, n, T(0));
            n--;
            iter = 0;
        }
        else if (l == n - 1)
        {
            ww = w.H(n, n - 1) * w.H(n - 1, n);
            p = (w.H(n - 1, n - 1) - w.H(n, n)) / T(2);
            q = p * p + ww;
            z = sqrt(abs(q));
            w.setH(n, n, w.H(n, n) + exshift);
            w.setH(n - 1, n - 1, w.H(n - 1, n - 1) + exshift);
            x = w.H(n, n);
            if (q >= T(0))
            {
                z = (p >= T(0)) ? p + z : p - z;
                st<T>(w.d, n - 1, x + z);
                st<T>(w.d, n, x + z);
                if (z != T(0)) st<T>(w.d, n, x - ww / z);
                st<T>(w.e, n - 1, T(0));
                st<T>(w.e, n, T(0));
                x = w.H(n, n - 1);
                s = abs(x) + abs(z);
                p = x / s;
                q = z / s;
                r = sqrt(p * p + q * q);
                p = p / r;
                q = q / r;
                for (int j = n - 1; j < nn; j++)
                {
                    z = w.H(n - 1, j);
                    T below = w.H(n, j);
                    w.setH(n - 1, j, q * z + p * below);
                    w.setH(n, j, q * below - p * z);
                }
                for (int i = 0; i <= n; i++)
                {
                    z = w.H(i, n - 1);
                    T right = w.H(i, n);
                    w.setH(i, n - 1, q * z + p * right);
                    w.setH(i, n, q * right - p * z);
                }
                for (int i = low; i <= high; i++)
                {
                    z = w.V(i, n - 1);
                    T right = w.V(i, n);
                    w.setV(i, n - 1, q * z + p * right);
                    w.setV(i, n, q * right - p * z);
                }
            }
            else
            {
                st<T>(w.d, n - 1, x + p);
                st<T>(w.d, n, x + p);
                st<T>(w.e, n - 1, z);
                st<T>(w.e, n, -z);
            }
            n = n - 2;
            iter = 0;
        }
        else
        {
            x = w.H(n, n);
            y = T(0);
            ww = T(0);
            if (l < n)
            {
                y = w.H(n - 1, n - 1);
                ww = w.H(n, n - 1) * w.H(n - 1, n);
            }
            // Wilkinson's original ad hoc shift
            if (iter == 10)
            {
                exshift += x;
                for (int i = low; i <= n; i++) w.setH(i, i, w.H(i, i) - x);
                s = abs(w.H(n, n - 1)) + abs(w.H(n - 1, n - 2));
                x = T(0.75) * s;
                y = x;
                ww = T(-0.4375) * s * s;
            }
            // MATLAB's new ad hoc shift
            if (iter == 30)
            {
                s = (y - x) / T(2);
                s = s * s + ww;
                if (s > T(0))
                {
                    s = sqrt(s);
                    if (y < x) s = -s;
                    s = x - ww / ((y - x) / T(2) + s);
                    for (int i = low; i <= n; i++) w.setH(i, i, w.H(i, i) - s);
                    exshift += s;
                    x = T(0.964);
                    y = x;
                    ww = x;
                }
            }
            iter = iter + 1;

            int m = n - 2;
            while (m >= l)
            {
                z = w.H(m, m);
                r = x - z;
                s = y - z;
                p = (r * s - ww) / w.H(m + 1, m) + w.H(m, m + 1);
                q = w.H(m + 1, m + 1) - z - r - s;
                r = w.H(m + 2, m + 1);
                s = abs(p) + abs(q) + abs(r);
                p = p / s;
                q = q / s;
                r = r / s;
                if (m == l) break;
                if (abs(w.H(m, m - 1)) * (abs(q) + abs(r)) < eps * (abs(p) * (abs(w.H(m - 1, m - 1)) + abs(z) + abs(w.H(m + 1, m + 1))))) break;
                m--;
            }

            for (int i = m + 2; i <= n; i++)
            {
                w.setH(i, i - 2, T(0));
                if (i > m + 2) w.setH(i, i - 3, T(0));
            }

            for (int k = m; k <= n - 1; k++)
            {
                bool notlast = (k != n - 1);
                if (k != m)
                {
                    p = w.H(k, k - 1);
                    q = w.H(k + 1, k - 1);
                    r = notlast ? w.H(k + 2, k - 1) : T(0);
                    x = abs(p) + abs(q) + abs(r);
                    if (x == T(0)) continue;
                    p = p / x;
                    q = q / x;
                    r = r / x;
                }
                s = sqrt(p * p + q * q + r * r);
                if (p < T(0)) s = -s;
                if (s != T(0))
                {
                    if (k != m) w.setH(k, k - 1, -s * x);
                    else if (l != m) w.setH(k, k - 1, -w.H(k, k - 1));
                    p = p + s;
                    x = p / s;
                    y = q / s;
                    z = r / s;
                    q = q / p;
                    r = r / p;

                    for (int j = k; j < nn; j++)
                    {
                        T hk = w.H(k, j);
                        T hk1 = w.H(k + 1, j);
                        p = hk + q * hk1;
                        if (notlast)
                        {
                            T hk2 = w.H(k + 2, j);
                            p = p + r * hk2;
                            w.setH(k + 2, j, hk2 - p * z);
                        }
                        w.setH(k, j, hk - p * x);
                        w.setH(k + 1, j, hk1 - p * y);
                    }
                    for (int i = 0; i <= min(n, k + 3); i++)
                    {
                        T hk = w.H(i, k);
                        T hk1 = w.H(i, k + 1);
                        p = x * hk + y * hk1;
                        if (notlast)
                        {
                            T hk2 = w.H(i, k + 2);
                            p = p + z * hk2;
                            w.setH(i, k + 2, hk2 - p * r);
                        }
                        w.setH(i, k, hk - p);
                        w.setH(i, k + 1, hk1 - p * q);
                    }
                    for (int i = low; i <= high; i++)
                    {
                        T vk = w.V(i, k);
                        T vk1 = w.V(i, k + 1);
                        p = x * vk + y * vk1;
                        if (notlast)
                        {
                            T vk2 = w.V(i, k + 2);
                            p = p + z * vk2;
                            w.setV(i, k + 2, vk2 - p * r);
                        }
                        w.setV(i, k, vk - p);
                        w.setV(i, k + 1, vk1 - p * q);
                    }
                }
            }
        }
    }

    if (norm == T(0)) return;

    for (n = nn - 1; n >= 0; n--)
    {
        p = ld<T>(w.d, n);
        q = ld<T>(w.e, n);
        if (q == T(0))
        {
            int l = n;
            w.setH(n, n, T(1));
            for (int i = n - 1; i >= 0; i--)
            {
                ww = w.H(i, i) - p;
                r = T(0);
                for (int j = l; j <= n; j++) r = r + w.H(i, j) * w.H(j, n);
                if (ld<T>(w.e, i) < T(0))
                {
                    z = ww;
                    s = r;
                }
                else
                {
                    l = i;
                    if (ld<T>(w.e, i) == T(0))
                    {
                        w.setH(i, n, (ww != T(0)) ? -r / ww : -r / (eps * norm));
                    }
                    else
                    {
                        x = w.H(i, i + 1);
                        y = w.H(i + 1, i);
                        T di = ld<T>(w.d, i) - p;
                        T ei = ld<T>(w.e, i);
                        q = di * di + ei * ei;
                        t = (x * s - z * r) / q;
                        w.setH(i, n, t);
                        if (abs(x) > abs(z)) w.setH(i + 1, n, (-r - ww * t) / x);
                        else w.setH(i + 1, n, (-s - y * t) / z);
                    }
                    t = abs(w.H(i, n));
                    if ((eps * t) * t > T(1))
                    {
                        for (int j = i; j <= n; j++) w.setH(j, n, w.H(j, n) / t);
                    }
                }
            }
        }
        else if (q < T(0))
        {
            int l = n - 1;
            T cr;
            T ci;
            if (abs(w.H(n, n - 1)) > abs(w.H(n - 1, n)))
            {
                T below = w.H(n, n - 1);
                T corner = w.H(n, n);
                w.setH(n - 1, n - 1, q / below);
                w.setH(n - 1, n, -(corner - p) / below);
            }
            else
            {
                cdiv<T>(T(0), -w.H(n - 1, n), w.H(n - 1, n - 1) - p, q, cr, ci);
                w.setH(n - 1, n - 1, cr);
                w.setH(n - 1, n, ci);
            }
            w.setH(n, n - 1, T(0));
            w.setH(n, n, T(1));
            for (int i = n - 2; i >= 0; i--)
            {
                T ra = T(0);
                T sa = T(0);
                for (int j = l; j <= n; j++)
                {
                    ra = ra + w.H(i, j) * w.H(j, n - 1);
                    sa = sa + w.H(i, j) * w.H(j, n);
                }
                ww = w.H(i, i) - p;
                if (ld<T>(w.e, i) < T(0))
                {
                    z = ww;
                    r = ra;
                    s = sa;
                }
                else
                {
                    l = i;
                    if (ld<T>(w.e, i) == T(0))
                    {
                        cdiv<T>(-ra, -sa, ww, q, cr, ci);
                        w.setH(i, n - 1, cr);
                        w.setH(i, n, ci);
                    }
                    else
                    {
                        x = w.H(i, i + 1);
                        y = w.H(i + 1, i);
                        T di = ld<T>(w.d, i) - p;
                        T ei = ld<T>(w.e, i);
                        T vr = di * di + ei * ei - q * q;
                        T vi = di * T(2) * q;
                        if (vr == T(0) && vi == T(0)) vr = eps * norm * (abs(ww) + abs(q) + abs(x) + abs(y) + abs(z));
                        cdiv<T>(x * r - z * ra + q * sa, x * s - z * sa - q * ra, vr, vi, cr, ci);
                        w.setH(i, n - 1, cr);
                        w.setH(i, n, ci);
                        T re = w.H(i, n - 1);
                        T im = w.H(i, n);
                        if (abs(x) > (abs(z) + abs(q)))
                        {
                            w.setH(i + 1, n - 1, (-ra - ww * re + q * im) / x);
                            w.setH(i + 1, n, (-sa - ww * im - q * re) / x);
                        }
                        else
                        {
                            cdiv<T>(-r - y * re, -s - y * im, z, q, cr, ci);
                            w.setH(i + 1, n - 1, cr);
                            w.setH(i + 1, n, ci);
                        }
                    }
                    t = max(abs(w.H(i, n - 1)), abs(w.H(i, n)));
                    if ((eps * t) * t > T(1))
                    {
                        for (int j = i; j <= n; j++)
                        {
                            w.setH(j, n - 1, w.H(j, n - 1) / t);
                            w.setH(j, n, w.H(j, n) / t);
                        }
                    }
                }
            }
        }
    }

    for (int j = nn - 1; j >= low; j--)
    {
        for (int i = low; i <= high; i++)
        {
            z = T(0);
            for (int k = low; k <= min(j, high); k++) z = z + w.V(i, k) * w.H(k, j);
            w.setV(i, j, z);
        }
    }
}

void eigKernel<T : __BuiltinFloatingPointType>(uint64_t idx, EigPushConstants pcs)
{
    if (idx >= pcs.count) return;
    int n = int(pcs.n);
    int64_t matrix = int64_t(idx) * n * n;
    int64_t vector = int64_t(idx) * n;
    Work<T> w;
    w.h = (T*)pcs.h + matrix;
    w.v = (T*)pcs.v + matrix;
    w.d = (T*)pcs.d + vector;
    w.e = (T*)pcs.e + vector;
    w.ort = (T*)pcs.ort + vector;
    w.nn = n;
    for (int i = 0; i < n; i++)
    {
        st<T>(w.d, i, T(0));
        st<T>(w.e, i, T(0));
        st<T>(w.ort, i, T(0));
    }
    orthes<T>(w);
    hqr2<T>(w);
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void eig_main<T : __BuiltinFloatingPointType>(
    uint3 gi : SV_DispatchThreadID,
    uniform EigPushConstants pcs)
{
    eigKernel<T>(uint64_t(gi.x), pcs);
}
