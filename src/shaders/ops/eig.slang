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
    T* tau;
    T* scale;
    T* scratch;
    int nn;

    T H(int i, int j) { return ld<T>(h, int64_t(i) * nn + j); }
    T V(int i, int j) { return ld<T>(v, int64_t(i) * nn + j); }
    [mutating] void setH(int i, int j, T x) { st<T>(h, int64_t(i) * nn + j, x); }
    [mutating] void setV(int i, int j, T x) { st<T>(v, int64_t(i) * nn + j, x); }

    // LAPACK's 1-based A(i, j) and Z(i, j)
    T A(int i, int j) { return H(i - 1, j - 1); }
    T Z(int i, int j) { return V(i - 1, j - 1); }
    [mutating] void setA(int i, int j, T x) { setH(i - 1, j - 1, x); }
    [mutating] void setZ(int i, int j, T x) { setV(i - 1, j - 1, x); }

    // staged through scratch: MoltenVK's translation can re-read a loaded value at its use, after the swap
    [mutating] void swapA(int i1, int j1, int i2, int j2)
    {
        st<T>(scratch, 0, A(i1, j1));
        st<T>(scratch, 1, A(i2, j2));
        setA(i1, j1, ld<T>(scratch, 1));
        setA(i2, j2, ld<T>(scratch, 0));
    }
    [mutating] void swapZ(int i1, int j1, int i2, int j2)
    {
        st<T>(scratch, 0, Z(i1, j1));
        st<T>(scratch, 1, Z(i2, j2));
        setZ(i1, j1, ld<T>(scratch, 1));
        setZ(i2, j2, ld<T>(scratch, 0));
    }
}

T ulpOf<T : __BuiltinFloatingPointType>() { return (sizeof(T) == 8) ? T(2.220446049250313e-16) : T(1.1920928955078125e-07); }
T safeMinimum<T : __BuiltinFloatingPointType>() { return (sizeof(T) == 8) ? T(2.2250738585072014e-308) : T(1.1754943508222875e-38); }

// Fortran's SIGN(a, b)
T fsign<T : __BuiltinFloatingPointType>(T a, T b) { return (b >= T(0)) ? abs(a) : -abs(a); }

// LAPACK's dlapy2
T lapy2<T : __BuiltinFloatingPointType>(T x, T y)
{
    T w = max(abs(x), abs(y));
    T z = min(abs(x), abs(y));
    if (z == T(0)) return w;
    T q = z / w;
    return w * sqrt(T(1) + q * q);
}

// one step of the reference dnrm2's scaled sum of squares
void nrm2Step<T : __BuiltinFloatingPointType>(T x, inout T scale, inout T ssq)
{
    if (x == T(0)) return;
    T ax = abs(x);
    if (scale < ax)
    {
        T q = scale / ax;
        ssq = T(1) + ssq * q * q;
        scale = ax;
    }
    else
    {
        T q = ax / scale;
        ssq = ssq + q * q;
    }
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

// LAPACK's dgebal with JOB = 'B': permutations that isolate eigenvalues, then power-of-2 scaling; returns ilo, ihi
void exchange<T : __BuiltinFloatingPointType>(inout Work<T> w, int j, int m, int k, int l)
{
    st<T>(w.scale, m - 1, T(j));
    if (j == m) return;
    for (int i = 1; i <= l; i++) w.swapA(i, j, i, m);
    for (int i = k; i <= w.nn; i++) w.swapA(j, i, m, i);
}

int2 gebal<T : __BuiltinFloatingPointType>(inout Work<T> w)
{
    int n = w.nn;
    int k = 1;
    int l = n;
    // rows isolating an eigenvalue, pushed down
    bool found = true;
    while (found)
    {
        found = false;
        for (int j = l; j >= 1; j--)
        {
            bool isolated = true;
            for (int i = 1; i <= l; i++)
            {
                if (i != j && w.A(j, i) != T(0)) { isolated = false; break; }
            }
            if (isolated)
            {
                exchange<T>(w, j, l, k, l);
                if (l == 1) return int2(1, 1);
                l--;
                found = true;
                break;
            }
        }
    }
    // columns isolating an eigenvalue, pushed left
    found = true;
    while (found)
    {
        found = false;
        for (int j = k; j <= l; j++)
        {
            bool isolated = true;
            for (int i = k; i <= l; i++)
            {
                if (i != j && w.A(i, j) != T(0)) { isolated = false; break; }
            }
            if (isolated)
            {
                exchange<T>(w, j, k, k, l);
                k++;
                found = true;
                break;
            }
        }
    }
    for (int i = k; i <= l; i++) st<T>(w.scale, i - 1, T(1));

    T sclfac = T(2);
    T factor = T(0.95);
    T sfmin1 = safeMinimum<T>() / ulpOf<T>();
    T sfmax1 = T(1) / sfmin1;
    T sfmin2 = sfmin1 * sclfac;
    T sfmax2 = T(1) / sfmin2;
    bool noconv = true;
    for (int sweep = 0; noconv && sweep < 1000; sweep++)
    {
        noconv = false;
        for (int i = k; i <= l; i++)
        {
            T c = T(0);
            T r = T(0);
            T cs = T(0), cq = T(1), rs = T(0), rq = T(1);
            for (int j = k; j <= l; j++)
            {
                nrm2Step<T>(w.A(j, i), cs, cq);
                nrm2Step<T>(w.A(i, j), rs, rq);
            }
            c = cs * sqrt(cq);
            r = rs * sqrt(rq);
            T ca = T(0);
            for (int j = 1; j <= l; j++) ca = max(ca, abs(w.A(j, i)));
            T ra = T(0);
            for (int j = k; j <= n; j++) ra = max(ra, abs(w.A(i, j)));
            if (c == T(0) || r == T(0)) continue;

            T g = r / sclfac;
            T f = T(1);
            T s = c + r;
            while (c < g && max(f, max(c, ca)) < sfmax2 && min(r, min(g, ra)) > sfmin2)
            {
                f = f * sclfac;
                c = c * sclfac;
                ca = ca * sclfac;
                r = r / sclfac;
                g = g / sclfac;
                ra = ra / sclfac;
            }
            g = c / sclfac;
            while (g >= r && max(r, ra) < sfmax2 && min(min(f, c), min(g, ca)) > sfmin2)
            {
                f = f / sclfac;
                c = c / sclfac;
                g = g / sclfac;
                ca = ca / sclfac;
                r = r * sclfac;
                ra = ra * sclfac;
            }
            if (c + r >= factor * s) continue;
            T previous = ld<T>(w.scale, i - 1);
            if (f < T(1) && previous < T(1) && f * previous <= sfmin1) continue;
            if (f > T(1) && previous > T(1) && previous >= sfmax1 / f) continue;
            g = T(1) / f;
            st<T>(w.scale, i - 1, previous * f);
            noconv = true;
            for (int j = k; j <= n; j++) w.setA(i, j, w.A(i, j) * g);
            for (int j = 1; j <= l; j++) w.setA(j, i, w.A(j, i) * f);
        }
    }
    return int2(k, l);
}

// LAPACK's dgehd2 on rows and columns ilo..ihi, then dorghr's Q into Z
void gehd2<T : __BuiltinFloatingPointType>(inout Work<T> w, int ilo, int ihi)
{
    int n = w.nn;
    for (int i = ilo; i < ihi; i++)
    {
        // dlarfg: the reflector (1, A(i+2:ihi, i)) that annihilates A(i+2:ihi, i)
        T alpha = w.A(i + 1, i);
        T xs = T(0), xq = T(1);
        for (int r = i + 2; r <= ihi; r++) nrm2Step<T>(w.A(r, i), xs, xq);
        T xnorm = xs * sqrt(xq);
        T tau = T(0);
        T beta = alpha;
        if (xnorm != T(0))
        {
            beta = -fsign<T>(lapy2<T>(alpha, xnorm), alpha);
            tau = (beta - alpha) / beta;
            T sc = T(1) / (alpha - beta);
            for (int r = i + 2; r <= ihi; r++) w.setA(r, i, w.A(r, i) * sc);
        }
        st<T>(w.tau, i - 1, tau);
        if (tau != T(0))
        {
            // dlarf from the right on A(1:ihi, i+1:ihi)
            for (int r = 1; r <= ihi; r++)
            {
                T sum = w.A(r, i + 1);
                for (int c = i + 2; c <= ihi; c++) sum = sum + w.A(r, c) * w.A(c, i);
                w.setA(r, i + 1, w.A(r, i + 1) + sum * (-tau));
                for (int c = i + 2; c <= ihi; c++) w.setA(r, c, w.A(r, c) + sum * (-tau * w.A(c, i)));
            }
            // dlarf from the left on A(i+1:ihi, i+1:n)
            for (int c = i + 1; c <= n; c++)
            {
                T sum = w.A(i + 1, c);
                for (int r = i + 2; r <= ihi; r++) sum = sum + w.A(r, c) * w.A(r, i);
                T temp = -tau * sum;
                w.setA(i + 1, c, w.A(i + 1, c) + temp);
                for (int r = i + 2; r <= ihi; r++) w.setA(r, c, w.A(r, c) + w.A(r, i) * temp);
            }
        }
        w.setA(i + 1, i, beta);
    }

    // Q = H(ilo) H(ilo+1) ... H(ihi-1), applied to the identity from the last reflector
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++) w.setZ(i, j, (i == j) ? T(1) : T(0));
    for (int i = ihi - 1; i >= ilo; i--)
    {
        T tau = ld<T>(w.tau, i - 1);
        if (tau == T(0)) continue;
        for (int c = i + 1; c <= ihi; c++)
        {
            T sum = w.Z(i + 1, c);
            for (int r = i + 2; r <= ihi; r++) sum = sum + w.Z(r, c) * w.A(r, i);
            T temp = -tau * sum;
            w.setZ(i + 1, c, w.Z(i + 1, c) + temp);
            for (int r = i + 2; r <= ihi; r++) w.setZ(r, c, w.Z(r, c) + w.A(r, i) * temp);
        }
    }
    for (int j = 1; j <= n; j++)
        for (int i = j + 2; i <= n; i++) w.setA(i, j, T(0));
}

// LAPACK's dlanv2: the Schur factorization of a real 2x2 block in standard form
void lanv2<T : __BuiltinFloatingPointType>(inout T a, inout T b, inout T c, inout T d, out T rt1r, out T rt1i, out T rt2r, out T rt2i, out T cs, out T sn)
{
    T eps = ulpOf<T>();
    if (c == T(0))
    {
        cs = T(1);
        sn = T(0);
    }
    else if (b == T(0))
    {
        cs = T(0);
        sn = T(1);
        T temp = d;
        d = a;
        a = temp;
        b = -c;
        c = T(0);
    }
    else if ((a - d) == T(0) && fsign<T>(T(1), b) != fsign<T>(T(1), c))
    {
        cs = T(1);
        sn = T(0);
    }
    else
    {
        T temp = a - d;
        T p = T(0.5) * temp;
        T bcmax = max(abs(b), abs(c));
        T bcmis = min(abs(b), abs(c)) * fsign<T>(T(1), b) * fsign<T>(T(1), c);
        T scale = max(abs(p), bcmax);
        T z = (p / scale) * p + (bcmax / scale) * bcmis;
        if (z >= T(4) * eps)
        {
            // real eigenvalues
            z = p + fsign<T>(sqrt(scale) * sqrt(z), p);
            a = d + z;
            d = d - (bcmax / z) * bcmis;
            T tau = lapy2<T>(c, z);
            cs = z / tau;
            sn = c / tau;
            b = b - c;
            c = T(0);
        }
        else
        {
            // complex or nearly equal real eigenvalues: make the diagonal equal
            T sigma = b + c;
            T tau = lapy2<T>(sigma, temp);
            cs = sqrt(T(0.5) * (T(1) + abs(sigma) / tau));
            sn = -(p / (tau * cs)) * fsign<T>(T(1), sigma);
            T aa = a * cs + b * sn;
            T bb = -a * sn + b * cs;
            T cc = c * cs + d * sn;
            T dd = -c * sn + d * cs;
            a = aa * cs + cc * sn;
            b = bb * cs + dd * sn;
            c = -aa * sn + cc * cs;
            d = -bb * sn + dd * cs;
            temp = T(0.5) * (a + d);
            a = temp;
            d = temp;
            if (c != T(0))
            {
                if (b != T(0))
                {
                    if (fsign<T>(T(1), b) == fsign<T>(T(1), c))
                    {
                        // real eigenvalues: reduce to upper triangular
                        T sab = sqrt(abs(b));
                        T sac = sqrt(abs(c));
                        p = fsign<T>(sab * sac, c);
                        tau = T(1) / sqrt(abs(b + c));
                        a = temp + p;
                        d = temp - p;
                        b = b - c;
                        c = T(0);
                        T cs1 = sab * tau;
                        T sn1 = sac * tau;
                        temp = cs * cs1 - sn * sn1;
                        sn = cs * sn1 + sn * cs1;
                        cs = temp;
                    }
                }
                else
                {
                    b = -c;
                    c = T(0);
                    temp = cs;
                    cs = -sn;
                    sn = temp;
                }
            }
        }
    }
    rt1r = a;
    rt2r = d;
    if (c == T(0))
    {
        rt1i = T(0);
        rt2i = T(0);
    }
    else
    {
        rt1i = sqrt(abs(b)) * sqrt(abs(c));
        rt2i = -rt1i;
    }
}

// LAPACK's dlahqr with WANTT and WANTZ: the real Schur form of the Hessenberg rows ilo..ihi, eigenvalues into d, e
void lahqr<T : __BuiltinFloatingPointType>(inout Work<T> w, int ilo, int ihi)
{
    int n = w.nn;
    for (int i = 1; i < ilo; i++) { st<T>(w.d, i - 1, w.A(i, i)); st<T>(w.e, i - 1, T(0)); }
    for (int i = ihi + 1; i <= n; i++) { st<T>(w.d, i - 1, w.A(i, i)); st<T>(w.e, i - 1, T(0)); }
    if (ilo == ihi)
    {
        st<T>(w.d, ilo - 1, w.A(ilo, ilo));
        st<T>(w.e, ilo - 1, T(0));
        return;
    }
    for (int j = ilo; j <= ihi - 3; j++)
    {
        w.setA(j + 2, j, T(0));
        w.setA(j + 3, j, T(0));
    }
    if (ilo <= ihi - 2) w.setA(ihi, ihi - 2, T(0));

    T ulp = ulpOf<T>();
    int nh = ihi - ilo + 1;
    T smlnum = safeMinimum<T>() * (T(nh) / ulp);
    int i1 = 1;
    int i2 = n;
    int itmax = 30 * max(10, nh);
    int kdefl = 0;

    int i = ihi;
    while (i >= ilo)
    {
        int l = ilo;
        bool split = false;
        for (int its = 0; its <= itmax; its++)
        {
            // a single small subdiagonal element (Ahues and Kressner's criterion)
            int k = i;
            for (; k > l; k--)
            {
                T sub = abs(w.A(k, k - 1));
                if (sub <= smlnum) break;
                T tst = abs(w.A(k - 1, k - 1)) + abs(w.A(k, k));
                if (tst == T(0))
                {
                    if (k - 2 >= ilo) tst = tst + abs(w.A(k - 1, k - 2));
                    if (k + 1 <= ihi) tst = tst + abs(w.A(k + 1, k));
                }
                if (sub <= ulp * tst)
                {
                    T up = abs(w.A(k - 1, k));
                    T ab = max(sub, up);
                    T ba = min(sub, up);
                    T gap = abs(w.A(k - 1, k - 1) - w.A(k, k));
                    T aa = max(abs(w.A(k, k)), gap);
                    T bb = min(abs(w.A(k, k)), gap);
                    T s = aa + ab;
                    if (ba * (ab / s) <= max(smlnum, ulp * (bb * (aa / s)))) break;
                }
            }
            l = k;
            if (l > ilo) w.setA(l, l - 1, T(0));
            if (l >= i - 1)
            {
                split = true;
                break;
            }
            kdefl++;

            T h11, h12, h21, h22;
            bool exceptionalI = kdefl % 20 == 0;
            bool exceptionalL = kdefl % 20 != 0 && kdefl % 10 == 0;
            if (exceptionalI)
            {
                T s = abs(w.A(i, i - 1)) + abs(w.A(i - 1, i - 2));
                h11 = T(0.75) * s + w.A(i, i);
                h12 = T(-0.4375) * s;
                h21 = s;
                h22 = h11;
            }
            else if (exceptionalL)
            {
                T s = abs(w.A(l + 1, l)) + abs(w.A(l + 2, l + 1));
                h11 = T(0.75) * s + w.A(l, l);
                h12 = T(-0.4375) * s;
                h21 = s;
                h22 = h11;
            }
            else
            {
                h11 = w.A(i - 1, i - 1);
                h21 = w.A(i, i - 1);
                h12 = w.A(i - 1, i);
                h22 = w.A(i, i);
            }
            T rt1r = T(0), rt1i = T(0), rt2r = T(0), rt2i = T(0);
            T s = abs(h11) + abs(h12) + abs(h21) + abs(h22);
            if (s != T(0))
            {
                h11 = h11 / s;
                h21 = h21 / s;
                h12 = h12 / s;
                h22 = h22 / s;
                T tr = (h11 + h22) / T(2);
                T det = (h11 - tr) * (h22 - tr) - h12 * h21;
                T rtdisc = sqrt(abs(det));
                if (det >= T(0))
                {
                    rt1r = tr * s;
                    rt2r = rt1r;
                    rt1i = rtdisc * s;
                    rt2i = -rt1i;
                }
                else
                {
                    rt1r = tr + rtdisc;
                    rt2r = tr - rtdisc;
                    if (abs(rt1r - h22) <= abs(rt2r - h22))
                    {
                        rt1r = rt1r * s;
                        rt2r = rt1r;
                    }
                    else
                    {
                        rt2r = rt2r * s;
                        rt1r = rt2r;
                    }
                }
            }

            // two consecutive small subdiagonal elements
            int m = i - 2;
            T v1 = T(0), v2 = T(0), v3 = T(0);
            for (; m >= l; m--)
            {
                T hmm = w.A(m, m);
                T h21s = w.A(m + 1, m);
                T sm = abs(hmm - rt2r) + abs(rt2i) + abs(h21s);
                h21s = w.A(m + 1, m) / sm;
                v1 = h21s * w.A(m, m + 1) + (hmm - rt1r) * ((hmm - rt2r) / sm) - rt1i * (rt2i / sm);
                v2 = h21s * (hmm + w.A(m + 1, m + 1) - rt1r - rt2r);
                v3 = h21s * w.A(m + 2, m + 1);
                sm = abs(v1) + abs(v2) + abs(v3);
                v1 = v1 / sm;
                v2 = v2 / sm;
                v3 = v3 / sm;
                if (m == l) break;
                T h00 = abs(w.A(m, m - 1)) * (abs(v2) + abs(v3));
                T h01 = ulp * abs(v1) * (abs(w.A(m - 1, m - 1)) + abs(hmm) + abs(w.A(m + 1, m + 1)));
                if (h00 <= h01) break;
            }

            // the double-shift QR step: a bulge from row m chased to the bottom
            for (int kk = m; kk <= i - 1; kk++)
            {
                int nr = min(3, i - kk + 1);
                T x1 = v1, x2 = v2, x3 = v3;
                if (kk > m)
                {
                    x1 = w.A(kk, kk - 1);
                    x2 = w.A(kk + 1, kk - 1);
                    x3 = (nr == 3) ? w.A(kk + 2, kk - 1) : T(0);
                }
                // dlarfg on (x1, x2[, x3])
                T xs = T(0), xq = T(1);
                nrm2Step<T>(x2, xs, xq);
                if (nr == 3) nrm2Step<T>(x3, xs, xq);
                T xnorm = xs * sqrt(xq);
                T t1 = T(0);
                T beta = x1;
                if (xnorm != T(0))
                {
                    beta = -fsign<T>(lapy2<T>(x1, xnorm), x1);
                    t1 = (beta - x1) / beta;
                    T sc = T(1) / (x1 - beta);
                    x2 = x2 * sc;
                    x3 = x3 * sc;
                }
                if (kk > m)
                {
                    w.setA(kk, kk - 1, beta);
                    w.setA(kk + 1, kk - 1, T(0));
                    if (kk < i - 1) w.setA(kk + 2, kk - 1, T(0));
                }
                else if (m > l)
                {
                    w.setA(kk, kk - 1, w.A(kk, kk - 1) * (T(1) - t1));
                }
                T t2 = t1 * x2;
                if (nr == 3)
                {
                    T t3 = t1 * x3;
                    for (int j = kk; j <= i2; j++)
                    {
                        T a0 = w.A(kk, j), a1 = w.A(kk + 1, j), a2 = w.A(kk + 2, j);
                        T sum = a0 + x2 * a1 + x3 * a2;
                        w.setA(kk, j, a0 - sum * t1);
                        w.setA(kk + 1, j, a1 - sum * t2);
                        w.setA(kk + 2, j, a2 - sum * t3);
                    }
                    for (int j = i1; j <= min(kk + 3, i); j++)
                    {
                        T a0 = w.A(j, kk), a1 = w.A(j, kk + 1), a2 = w.A(j, kk + 2);
                        T sum = a0 + x2 * a1 + x3 * a2;
                        w.setA(j, kk, a0 - sum * t1);
                        w.setA(j, kk + 1, a1 - sum * t2);
                        w.setA(j, kk + 2, a2 - sum * t3);
                    }
                    for (int j = ilo; j <= ihi; j++)
                    {
                        T z0 = w.Z(j, kk), z1 = w.Z(j, kk + 1), z2 = w.Z(j, kk + 2);
                        T sum = z0 + x2 * z1 + x3 * z2;
                        w.setZ(j, kk, z0 - sum * t1);
                        w.setZ(j, kk + 1, z1 - sum * t2);
                        w.setZ(j, kk + 2, z2 - sum * t3);
                    }
                }
                else
                {
                    for (int j = kk; j <= i2; j++)
                    {
                        T a0 = w.A(kk, j), a1 = w.A(kk + 1, j);
                        T sum = a0 + x2 * a1;
                        w.setA(kk, j, a0 - sum * t1);
                        w.setA(kk + 1, j, a1 - sum * t2);
                    }
                    for (int j = i1; j <= i; j++)
                    {
                        T a0 = w.A(j, kk), a1 = w.A(j, kk + 1);
                        T sum = a0 + x2 * a1;
                        w.setA(j, kk, a0 - sum * t1);
                        w.setA(j, kk + 1, a1 - sum * t2);
                    }
                    for (int j = ilo; j <= ihi; j++)
                    {
                        T z0 = w.Z(j, kk), z1 = w.Z(j, kk + 1);
                        T sum = z0 + x2 * z1;
                        w.setZ(j, kk, z0 - sum * t1);
                        w.setZ(j, kk + 1, z1 - sum * t2);
                    }
                }
            }
        }
        if (!split) return;

        if (l == i)
        {
            st<T>(w.d, i - 1, w.A(i, i));
            st<T>(w.e, i - 1, T(0));
        }
        else
        {
            T a = w.A(i - 1, i - 1), b = w.A(i - 1, i), c = w.A(i, i - 1), dd = w.A(i, i);
            T rt1r, rt1i, rt2r, rt2i, cs, sn;
            lanv2<T>(a, b, c, dd, rt1r, rt1i, rt2r, rt2i, cs, sn);
            w.setA(i - 1, i - 1, a);
            w.setA(i - 1, i, b);
            w.setA(i, i - 1, c);
            w.setA(i, i, dd);
            st<T>(w.d, i - 2, rt1r);
            st<T>(w.e, i - 2, rt1i);
            st<T>(w.d, i - 1, rt2r);
            st<T>(w.e, i - 1, rt2i);
            for (int j = i + 1; j <= i2; j++)
            {
                T x = w.A(i - 1, j), y = w.A(i, j);
                w.setA(i - 1, j, cs * x + sn * y);
                w.setA(i, j, cs * y - sn * x);
            }
            for (int j = i1; j <= i - 2; j++)
            {
                T x = w.A(j, i - 1), y = w.A(j, i);
                w.setA(j, i - 1, cs * x + sn * y);
                w.setA(j, i, cs * y - sn * x);
            }
            for (int j = ilo; j <= ihi; j++)
            {
                T x = w.Z(j, i - 1), y = w.Z(j, i);
                w.setZ(j, i - 1, cs * x + sn * y);
                w.setZ(j, i, cs * y - sn * x);
            }
        }
        kdefl = 0;
        i = l - 1;
    }
}

// JAMA's hqr2 back substitution: the eigenvectors of the quasi-triangular H, then back-transformed by V
void backSubstitute<T : __BuiltinFloatingPointType>(inout Work<T> w, T norm)
{
    int nn = w.nn;
    int n;
    int low = 0;
    int high = nn - 1;
    T eps = ulpOf<T>();
    T p = T(0), q = T(0), r = T(0), s = T(0), z = T(0), t, x, y, ww;
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
            // dtrevc's start on lanv2's standard block [p b; c p]: (1, i |q| / b), or (-|q| / c, i)
            if (abs(w.H(n, n - 1)) > abs(w.H(n - 1, n)))
            {
                T below = w.H(n, n - 1);
                T corner = w.H(n, n);
                w.setH(n - 1, n - 1, q / below);
                w.setH(n - 1, n, -(corner - p) / below);
                w.setH(n, n - 1, T(0));
                w.setH(n, n, T(1));
            }
            else
            {
                T above = w.H(n - 1, n);
                w.setH(n - 1, n - 1, T(1));
                w.setH(n - 1, n, T(0));
                w.setH(n, n - 1, T(0));
                w.setH(n, n, -q / above);
            }
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


// LAPACK's dgebak: undo gebal's scaling and permutations on the eigenvector rows
void gebak<T : __BuiltinFloatingPointType>(inout Work<T> w, int ilo, int ihi)
{
    int n = w.nn;
    for (int i = ilo; i <= ihi; i++)
    {
        T s = ld<T>(w.scale, i - 1);
        for (int j = 1; j <= n; j++) w.setZ(i, j, w.Z(i, j) * s);
    }
    for (int ii = 1; ii <= n; ii++)
    {
        int i = ii;
        if (i >= ilo && i <= ihi) continue;
        if (i < ilo) i = ilo - ii;
        int k = int(ld<T>(w.scale, i - 1).toFloat());
        if (k == i) continue;
        for (int j = 1; j <= n; j++) w.swapZ(i, j, k, j);
    }
}

// LAPACK's dgeev path (gebal, gehrd, hseqr), so eigenvalues come out in the CPU's order
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
    w.tau = (T*)pcs.ort + int64_t(idx) * (2 * n + 2);
    w.scale = w.tau + n;
    w.scratch = w.scale + n;
    w.nn = n;
    for (int i = 0; i < n; i++)
    {
        st<T>(w.d, i, T(0));
        st<T>(w.e, i, T(0));
        st<T>(w.tau, i, T(0));
        st<T>(w.scale, i, T(1));
    }
    int2 range = gebal<T>(w);
    gehd2<T>(w, range.x, range.y);
    T norm = T(0);
    for (int i = 0; i < n; i++)
        for (int j = max(i - 1, 0); j < n; j++) norm = norm + abs(w.H(i, j));
    lahqr<T>(w, range.x, range.y);
    backSubstitute<T>(w, norm);
    gebak<T>(w, range.x, range.y);
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void eig_main<T : __BuiltinFloatingPointType>(
    uint3 gi : SV_DispatchThreadID,
    uniform EigPushConstants pcs)
{
    eigKernel<T>(uint64_t(gi.x), pcs);
}
