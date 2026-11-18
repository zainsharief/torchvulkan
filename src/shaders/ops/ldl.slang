module ldl;

import core.indexing;

[[vk::constant_id(0)]] public const uint MODE = 0;
[[vk::constant_id(1)]] public const uint WG_x = 0;
[[vk::constant_id(2)]] public const uint COMPLEX = 0;
[[vk::constant_id(3)]] public const uint HERMITIAN = 0;

public struct LdlPushConstants
{
    public uint64_t a;
    public uint64_t pivots;
    public uint64_t info;
    public uint64_t b;
    public uint64_t scratch;
    public uint64_t count;
    public uint n;
    public uint k;
}

// one element; real matrices keep im = 0 and every operation reduces to the real one
struct Cx<T : __BuiltinFloatingPointType>
{
    T re;
    T im;
}

Cx<T> cx<T : __BuiltinFloatingPointType>(T re, T im) { Cx<T> c; c.re = re; c.im = im; return c; }

Cx<T> cadd<T : __BuiltinFloatingPointType>(Cx<T> a, Cx<T> b) { return COMPLEX != 0 ? cx<T>(a.re + b.re, a.im + b.im) : cx<T>(a.re + b.re, T(0)); }
Cx<T> csub<T : __BuiltinFloatingPointType>(Cx<T> a, Cx<T> b) { return COMPLEX != 0 ? cx<T>(a.re - b.re, a.im - b.im) : cx<T>(a.re - b.re, T(0)); }
Cx<T> cneg<T : __BuiltinFloatingPointType>(Cx<T> a) { return cx<T>(-a.re, -a.im); }
Cx<T> conj<T : __BuiltinFloatingPointType>(Cx<T> a) { return cx<T>(a.re, -a.im); }
Cx<T> creal<T : __BuiltinFloatingPointType>(Cx<T> a) { return cx<T>(a.re, T(0)); }
Cx<T> cscale<T : __BuiltinFloatingPointType>(Cx<T> a, T s) { return cx<T>(a.re * s, a.im * s); }
bool cnonzero<T : __BuiltinFloatingPointType>(Cx<T> a) { return a.re != T(0) || a.im != T(0); }

Cx<T> cmul<T : __BuiltinFloatingPointType>(Cx<T> a, Cx<T> b)
{
    if (COMPLEX == 0) return cx<T>(a.re * b.re, T(0));
    return cx<T>(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

// Smith's division
Cx<T> cdiv<T : __BuiltinFloatingPointType>(Cx<T> a, Cx<T> b)
{
    if (COMPLEX == 0) return cx<T>(a.re / b.re, T(0));
    if (abs(b.re) >= abs(b.im))
    {
        T r = b.im / b.re;
        T d = b.re + b.im * r;
        return cx<T>((a.re + a.im * r) / d, (a.im - a.re * r) / d);
    }
    T r = b.re / b.im;
    T d = b.im + b.re * r;
    return cx<T>((a.re * r + a.im) / d, (a.im * r - a.re) / d);
}

// LAPACK's CABS1, |x| for a real matrix
T cabs1<T : __BuiltinFloatingPointType>(Cx<T> a) { return COMPLEX != 0 ? abs(a.re) + abs(a.im) : abs(a.re); }

// LAPACK's dlapy2
T lapy2<T : __BuiltinFloatingPointType>(T x, T y)
{
    T w = max(abs(x), abs(y));
    T z = min(abs(x), abs(y));
    if (z == T(0)) return w;
    T q = z / w;
    return w * sqrt(T(1) + q * q);
}

// LAPACK's A(i, j), 1-based, on a row-major matrix
struct Matrix<T : __BuiltinFloatingPointType>
{
    T* data;
    int n;
    int stride;

    int64_t at(int i, int j) { return (int64_t(i - 1) * stride + (j - 1)) * (COMPLEX != 0 ? 2 : 1); }
    Cx<T> get(int i, int j)
    {
        int64_t e = at(i, j);
        return cx<T>(data[e], COMPLEX != 0 ? data[e + 1] : T(0));
    }
    [mutating] void set(int i, int j, Cx<T> v)
    {
        int64_t e = at(i, j);
        data[e] = v.re;
        if (COMPLEX != 0) data[e + 1] = v.im;
    }

    // staged through scratch: MoltenVK's translation can re-read a loaded value at its use, after the swap
    [mutating] void swap(int i1, int j1, int i2, int j2, T* scratch)
    {
        Cx<T> first = get(i1, j1);
        Cx<T> second = get(i2, j2);
        scratch[0] = first.re;
        scratch[1] = first.im;
        scratch[2] = second.re;
        scratch[3] = second.im;
        set(i1, j1, cx<T>(scratch[2], scratch[3]));
        set(i2, j2, cx<T>(scratch[0], scratch[1]));
    }
}

// LAPACK's idamax / izamax, 1-based
int idamax<T : __BuiltinFloatingPointType>(Matrix<T> a, int count, int i, int j, bool along_row)
{
    int best = 1;
    T best_value = T(-1);
    for (int t = 0; t < count; t++)
    {
        T v = cabs1<T>(along_row ? a.get(i, j + t) : a.get(i + t, j));
        if (v > best_value || (isnan(v) && !isnan(best_value)))
        {
            best = t + 1;
            best_value = v;
        }
    }
    return best;
}

// the size of a diagonal entry: zhetf2 reads only its real part
T diagonalSize<T : __BuiltinFloatingPointType>(Cx<T> a) { return HERMITIAN != 0 ? abs(a.re) : cabs1<T>(a); }

// LAPACK's dsytf2 / zsytf2 / zhetf2 with UPLO = 'L' (Bunch-Kaufman)
void factor<T : __BuiltinFloatingPointType>(int64_t batch, LdlPushConstants pcs)
{
    int n = int(pcs.n);
    int64_t width = COMPLEX != 0 ? 2 : 1;
    var a = Matrix<T>((T*)pcs.a + batch * n * n * width, n, n);
    int* ipiv = (int*)pcs.pivots + batch * n;
    T* scratch = (T*)pcs.scratch + batch * 4;
    T alpha = (T(1) + sqrt(T(17))) / T(8);
    bool hermitian = COMPLEX != 0 && HERMITIAN != 0;
    int info = 0;

    int k = 1;
    while (k <= n)
    {
        int kstep = 1;
        int kp = k;
        T absakk = diagonalSize<T>(a.get(k, k));
        int imax = k;
        T colmax = T(0);
        if (k < n)
        {
            imax = k + idamax<T>(a, n - k, k + 1, k, false);
            colmax = cabs1<T>(a.get(imax, k));
        }
        if (max(absakk, colmax) == T(0) || isnan(absakk))
        {
            if (info == 0) info = k;
            kp = k;
            if (hermitian) a.set(k, k, creal<T>(a.get(k, k)));
        }
        else
        {
            if (absakk >= alpha * colmax)
            {
                kp = k;
            }
            else
            {
                int jmax = k - 1 + idamax<T>(a, imax - k, imax, k, true);
                T rowmax = cabs1<T>(a.get(imax, jmax));
                if (imax < n)
                {
                    jmax = imax + idamax<T>(a, n - imax, imax + 1, imax, false);
                    rowmax = max(rowmax, cabs1<T>(a.get(jmax, imax)));
                }
                if (absakk >= alpha * colmax * (colmax / rowmax)) kp = k;
                else if (diagonalSize<T>(a.get(imax, imax)) >= alpha * rowmax) kp = imax;
                else
                {
                    kp = imax;
                    kstep = 2;
                }
            }

            int kk = k + kstep - 1;
            if (kp != kk)
            {
                for (int i = kp + 1; i <= n; i++) a.swap(i, kk, i, kp, scratch);
                if (hermitian)
                {
                    for (int j = kk + 1; j < kp; j++)
                    {
                        Cx<T> t = conj<T>(a.get(j, kk));
                        a.set(j, kk, conj<T>(a.get(kp, j)));
                        a.set(kp, j, t);
                    }
                    a.set(kp, kk, conj<T>(a.get(kp, kk)));
                    Cx<T> r1 = creal<T>(a.get(kk, kk));
                    a.set(kk, kk, creal<T>(a.get(kp, kp)));
                    a.set(kp, kp, r1);
                    if (kstep == 2)
                    {
                        a.set(k, k, creal<T>(a.get(k, k)));
                        a.swap(k + 1, k, kp, k, scratch);
                    }
                }
                else
                {
                    for (int t = 0; t < kp - kk - 1; t++) a.swap(kk + 1 + t, kk, kp, kk + 1 + t, scratch);
                    a.swap(kk, kk, kp, kp, scratch);
                    if (kstep == 2) a.swap(k + 1, k, kp, k, scratch);
                }
            }
            else if (hermitian)
            {
                a.set(k, k, creal<T>(a.get(k, k)));
                if (kstep == 2) a.set(k + 1, k + 1, creal<T>(a.get(k + 1, k + 1)));
            }

            if (kstep == 1)
            {
                if (k < n)
                {
                    // zher / zsyr with alpha = -r1, then a scale by r1 (real for zhetf2)
                    Cx<T> d11 = hermitian ? cx<T>(T(1) / a.get(k, k).re, T(0)) : cdiv<T>(cx<T>(T(1), T(0)), a.get(k, k));
                    for (int j = k + 1; j <= n; j++)
                    {
                        Cx<T> xj = a.get(j, k);
                        if (cnonzero<T>(xj))
                        {
                            Cx<T> temp = hermitian ? cscale<T>(conj<T>(xj), -d11.re) : cmul<T>(cneg<T>(d11), xj);
                            if (hermitian) a.set(j, j, cx<T>(a.get(j, j).re + cmul<T>(xj, temp).re, T(0)));
                            for (int i = hermitian ? j + 1 : j; i <= n; i++) a.set(i, j, cadd<T>(a.get(i, j), cmul<T>(a.get(i, k), temp)));
                        }
                        else if (hermitian)
                        {
                            a.set(j, j, creal<T>(a.get(j, j)));
                        }
                    }
                    for (int i = k + 1; i <= n; i++) a.set(i, k, hermitian ? cscale<T>(a.get(i, k), d11.re) : cmul<T>(a.get(i, k), d11));
                }
            }
            else if (hermitian)
            {
                if (k < n - 1)
                {
                    Cx<T> sub = a.get(k + 1, k);
                    T d = lapy2<T>(sub.re, sub.im);
                    T d11 = a.get(k + 1, k + 1).re / d;
                    T d22 = a.get(k, k).re / d;
                    T tt = T(1) / (d11 * d22 - T(1));
                    Cx<T> d21 = cx<T>(sub.re / d, sub.im / d);
                    d = tt / d;
                    for (int j = k + 2; j <= n; j++)
                    {
                        Cx<T> wk = cscale<T>(csub<T>(cscale<T>(a.get(j, k), d11), cmul<T>(d21, a.get(j, k + 1))), d);
                        Cx<T> wkp1 = cscale<T>(csub<T>(cscale<T>(a.get(j, k + 1), d22), cmul<T>(conj<T>(d21), a.get(j, k))), d);
                        for (int i = j; i <= n; i++)
                        {
                            a.set(i, j, csub<T>(csub<T>(a.get(i, j), cmul<T>(a.get(i, k), conj<T>(wk))), cmul<T>(a.get(i, k + 1), conj<T>(wkp1))));
                        }
                        a.set(j, k, wk);
                        a.set(j, k + 1, wkp1);
                        a.set(j, j, creal<T>(a.get(j, j)));
                    }
                }
            }
            else
            {
                if (k < n - 1)
                {
                    Cx<T> one = cx<T>(T(1), T(0));
                    Cx<T> d21 = a.get(k + 1, k);
                    Cx<T> d11 = cdiv<T>(a.get(k + 1, k + 1), d21);
                    Cx<T> d22 = cdiv<T>(a.get(k, k), d21);
                    Cx<T> t = cdiv<T>(one, csub<T>(cmul<T>(d11, d22), one));
                    d21 = cdiv<T>(t, d21);
                    for (int j = k + 2; j <= n; j++)
                    {
                        Cx<T> wk = cmul<T>(d21, csub<T>(cmul<T>(d11, a.get(j, k)), a.get(j, k + 1)));
                        Cx<T> wkp1 = cmul<T>(d21, csub<T>(cmul<T>(d22, a.get(j, k + 1)), a.get(j, k)));
                        for (int i = j; i <= n; i++) a.set(i, j, csub<T>(csub<T>(a.get(i, j), cmul<T>(a.get(i, k), wk)), cmul<T>(a.get(i, k + 1), wkp1)));
                        a.set(j, k, wk);
                        a.set(j, k + 1, wkp1);
                    }
                }
            }
        }
        if (kstep == 1)
        {
            ipiv[k - 1] = kp;
        }
        else
        {
            ipiv[k - 1] = -kp;
            ipiv[k] = -kp;
        }
        k += kstep;
    }
    ((int*)pcs.info)[batch] = info;
}

// LAPACK's dsytrs / zsytrs / zhetrs with UPLO = 'L'
void solve<T : __BuiltinFloatingPointType>(int64_t batch, int column, LdlPushConstants pcs)
{
    int n = int(pcs.n);
    int k_rhs = int(pcs.k);
    int64_t width = COMPLEX != 0 ? 2 : 1;
    var a = Matrix<T>((T*)pcs.a + batch * n * n * width, n, n);
    var b = Matrix<T>((T*)pcs.b + (batch * n * k_rhs + column) * width, n, k_rhs);
    int* ipiv = (int*)pcs.pivots + batch * n;
    T* scratch = (T*)pcs.scratch + (batch * k_rhs + column) * 4;
    bool hermitian = COMPLEX != 0 && HERMITIAN != 0;
    Cx<T> one = cx<T>(T(1), T(0));

    int k = 1;
    while (k <= n)
    {
        if (ipiv[k - 1] > 0)
        {
            int kp = ipiv[k - 1];
            if (kp != k) b.swap(k, 1, kp, 1, scratch);
            for (int i = k + 1; i <= n; i++) b.set(i, 1, csub<T>(b.get(i, 1), cmul<T>(a.get(i, k), b.get(k, 1))));
            if (hermitian) b.set(k, 1, cscale<T>(b.get(k, 1), T(1) / a.get(k, k).re));
            else b.set(k, 1, cmul<T>(b.get(k, 1), cdiv<T>(one, a.get(k, k))));
            k += 1;
        }
        else
        {
            int kp = -ipiv[k - 1];
            if (kp != k + 1) b.swap(k + 1, 1, kp, 1, scratch);
            if (k < n - 1)
            {
                for (int i = k + 2; i <= n; i++) b.set(i, 1, csub<T>(b.get(i, 1), cmul<T>(a.get(i, k), b.get(k, 1))));
                for (int i = k + 2; i <= n; i++) b.set(i, 1, csub<T>(b.get(i, 1), cmul<T>(a.get(i, k + 1), b.get(k + 1, 1))));
            }
            Cx<T> akm1k = a.get(k + 1, k);
            Cx<T> upper = hermitian ? conj<T>(akm1k) : akm1k;
            Cx<T> akm1 = cdiv<T>(a.get(k, k), upper);
            Cx<T> ak = cdiv<T>(a.get(k + 1, k + 1), akm1k);
            Cx<T> denom = csub<T>(cmul<T>(akm1, ak), one);
            Cx<T> bkm1 = cdiv<T>(b.get(k, 1), upper);
            Cx<T> bk = cdiv<T>(b.get(k + 1, 1), akm1k);
            b.set(k, 1, cdiv<T>(csub<T>(cmul<T>(ak, bkm1), bk), denom));
            b.set(k + 1, 1, cdiv<T>(csub<T>(cmul<T>(akm1, bk), bkm1), denom));
            k += 2;
        }
    }

    k = n;
    while (k >= 1)
    {
        int last = ipiv[k - 1] > 0 ? k : k - 1;
        if (k < n)
        {
            // dgemv 'T' (zgemv 'C' between two conjugations for zhetrs)
            for (int row = k; row >= last; row--)
            {
                Cx<T> s = cx<T>(T(0), T(0));
                for (int i = k + 1; i <= n; i++)
                {
                    Cx<T> l = a.get(i, row);
                    s = cadd<T>(s, cmul<T>(b.get(i, 1), hermitian ? conj<T>(l) : l));
                }
                b.set(row, 1, csub<T>(b.get(row, 1), s));
            }
        }
        int kp = ipiv[k - 1] > 0 ? ipiv[k - 1] : -ipiv[k - 1];
        if (kp != k) b.swap(k, 1, kp, 1, scratch);
        k = last - 1;
    }
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void ldl_main<T : __BuiltinFloatingPointType>(
    uint3 gi : SV_DispatchThreadID,
    uniform LdlPushConstants pcs)
{
    uint64_t idx = uint64_t(flat_thread(gi, WG_x));
    if (idx >= pcs.count) return;
    if (MODE == 0) factor<T>(int64_t(idx), pcs);
    else solve<T>(int64_t(idx) / int64_t(pcs.k), int(int64_t(idx) % int64_t(pcs.k)), pcs);
}
