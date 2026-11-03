module householder;

[[vk::constant_id(0)]] public const uint MODE = 0;
[[vk::constant_id(1)]] public const uint LEFT = 1;
[[vk::constant_id(2)]] public const uint TRANSPOSE = 0;
[[vk::constant_id(3)]] public const uint WG_x = 0;

public struct HouseholderPushConstants
{
    public uint64_t a;
    public uint64_t tau;
    public uint64_t c;
    public uint m;
    public uint n;
    public uint k;
    public uint cm;
    public uint cn;
}

// LAPACK's dgeqr2
void geqrf<T : __BuiltinFloatingPointType>(uint b, uint lane, HouseholderPushConstants pcs)
{
    int m = int(pcs.m);
    int n = int(pcs.n);
    int k = int(pcs.k);
    T* a = (T*)pcs.a + int64_t(b) * m * n;
    T* tau = (T*)pcs.tau + int64_t(b) * k;

    for (int j = 0; j < k; j++)
    {
        if (lane == 0)
        {
            T alpha = a[j * n + j];
            T xnorm = T(0);
            for (int i = j + 1; i < m; i++) xnorm += a[i * n + j] * a[i * n + j];
            xnorm = sqrt(xnorm);
            T t = T(0);
            if (xnorm != T(0))
            {
                T norm = sqrt(alpha * alpha + xnorm * xnorm);
                T beta = (alpha >= T(0)) ? -norm : norm;
                // tau is stored before a[j][j] is overwritten: MoltenVK's translation can re-read alpha at its use
                t = (beta - alpha) / beta;
                tau[j] = t;
                T scale = T(1) / (alpha - beta);
                for (int i = j + 1; i < m; i++) a[i * n + j] *= scale;
                a[j * n + j] = beta;
            }
            else
            {
                tau[j] = t;
            }
        }
        AllMemoryBarrierWithGroupSync();

        // tau travels through memory: MoltenVK mistranslates a groupshared float
        T t = tau[j];
        if (t != T(0))
        {
            for (int c = j + 1 + int(lane); c < n; c += int(WG_x))
            {
                T w = a[j * n + c];
                for (int i = j + 1; i < m; i++) w += a[i * n + j] * a[i * n + c];
                T temp = -t * w;
                a[j * n + c] += temp;
                for (int i = j + 1; i < m; i++) a[i * n + c] += a[i * n + j] * temp;
            }
        }
        AllMemoryBarrierWithGroupSync();
    }
}

// LAPACK's dorg2r
void orgqr<T : __BuiltinFloatingPointType>(uint b, uint lane, HouseholderPushConstants pcs)
{
    int m = int(pcs.m);
    int n = int(pcs.n);
    int k = int(pcs.k);
    T* a = (T*)pcs.a + int64_t(b) * m * n;
    T* tau = (T*)pcs.tau + int64_t(b) * k;

    for (int idx = int(lane); idx < m * (n - k); idx += int(WG_x))
    {
        int i = idx / (n - k);
        int c = k + idx % (n - k);
        a[i * n + c] = (i == c) ? T(1) : T(0);
    }
    AllMemoryBarrierWithGroupSync();

    for (int j = k - 1; j >= 0; j--)
    {
        // lavapipe loads tau[-1] after the last iteration
        T t = tau[max(j, 0)];
        if (j < n - 1)
        {
            if (lane == 0) a[j * n + j] = T(1);
            AllMemoryBarrierWithGroupSync();
            if (t != T(0))
            {
                for (int c = j + 1 + int(lane); c < n; c += int(WG_x))
                {
                    T w = T(0);
                    for (int i = j; i < m; i++) w += a[i * n + j] * a[i * n + c];
                    T temp = -t * w;
                    for (int i = j; i < m; i++) a[i * n + c] += a[i * n + j] * temp;
                }
            }
            AllMemoryBarrierWithGroupSync();
        }
        for (int i = j + 1 + int(lane); i < m; i += int(WG_x)) a[i * n + j] *= -t;
        if (lane == 0) a[j * n + j] = T(1) - t;
        for (int i = int(lane); i < j; i += int(WG_x)) a[i * n + j] = T(0);
        AllMemoryBarrierWithGroupSync();
    }
}

// LAPACK's dorm2r
void ormqr<T : __BuiltinFloatingPointType>(uint b, uint lane, HouseholderPushConstants pcs)
{
    int n = int(pcs.n);
    int k = int(pcs.k);
    int cm = int(pcs.cm);
    int cn = int(pcs.cn);
    int rows = (LEFT != 0) ? cm : cn;
    T* a = (T*)pcs.a + int64_t(b) * pcs.m * n;
    T* tau = (T*)pcs.tau + int64_t(b) * k;
    T* c = (T*)pcs.c + int64_t(b) * cm * cn;

    // Q C applies H(k) first and Q^T C applies H(1) first; from the right the order flips
    bool forward = (LEFT != 0) == (TRANSPOSE != 0);
    for (int step = 0; step < k; step++)
    {
        int j = forward ? step : k - 1 - step;
        T t = tau[j];
        if (t != T(0))
        {
            int lines = (LEFT != 0) ? cn : cm;
            for (int line = int(lane); line < lines; line += int(WG_x))
            {
                T w = T(0);
                for (int i = j; i < rows; i++)
                {
                    T v = (i == j) ? T(1) : a[i * n + j];
                    T entry = (LEFT != 0) ? c[i * cn + line] : c[line * cn + i];
                    w += v * entry;
                }
                T temp = -t * w;
                for (int i = j; i < rows; i++)
                {
                    T v = (i == j) ? T(1) : a[i * n + j];
                    if (LEFT != 0) c[i * cn + line] += v * temp;
                    else c[line * cn + i] += v * temp;
                }
            }
        }
        AllMemoryBarrierWithGroupSync();
    }
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void householder_main<T : __BuiltinFloatingPointType>(
    uint3 group : SV_GroupID,
    uint3 thread : SV_GroupThreadID,
    uniform HouseholderPushConstants pcs)
{
    if (MODE == 0) geqrf<T>(group.x, thread.x, pcs);
    else if (MODE == 1) orgqr<T>(group.x, thread.x, pcs);
    else ormqr<T>(group.x, thread.x, pcs);
}
