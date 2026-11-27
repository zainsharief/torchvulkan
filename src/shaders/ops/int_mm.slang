module int_mm;

__exported import core.dtype;

// torch._int_mm: int8 [M, K] x int8 [K, N] -> int32 [M, N], exact int32 accumulation.
// One 16x16 workgroup computes a 64x64 output tile, 4x4 outputs per thread, stepping K by 32
// through shared memory. Strides are arbitrary, so a transposed weight (weight.t()) works;
// tile loads walk the contiguous dimension of each operand so they stay coalesced.

public struct IntMMPushConstants
{
    public uint64_t a;
    public uint64_t b;
    public uint64_t out;
    public int64_t a_stride_m;
    public int64_t a_stride_k;
    public int64_t b_stride_k;
    public int64_t b_stride_n;
    public int64_t out_stride_m;
    public uint M;
    public uint N;
    public uint K;
}

static const uint TM = 64;
static const uint TN = 64;
static const uint TK = 32;
static const uint THREADS = 256;

groupshared int tileA[TK * (TM + 1)];
groupshared int tileB[TK * (TN + 1)];

[shader("compute")]
[numthreads(16, 16, 1)]
void int_mm_main<T : ITorchVulkan>(
    uint3 li : SV_GroupThreadID,
    uint3 wg : SV_GroupID,
    uniform IntMMPushConstants pcs)
{
    T* A = (T*)pcs.a;
    T* B = (T*)pcs.b;
    int32_t* C = (int32_t*)pcs.out;
    uint tid = li.y * 16 + li.x;
    uint m0 = wg.y * TM;
    uint n0 = wg.x * TN;

    int acc[4][4];
    [unroll] for (uint i = 0; i < 4; i++)
        [unroll] for (uint j = 0; j < 4; j++)
            acc[i][j] = 0;

    for (uint k0 = 0; k0 < pcs.K; k0 += TK)
    {
        for (uint e = tid; e < TM * TK; e += THREADS)
        {
            uint r, c;  // r: row in the tile (m), c: column (k)
            if (pcs.a_stride_k == 1) { r = e / TK; c = e % TK; } else { c = e / TM; r = e % TM; }
            uint m = m0 + r, k = k0 + c;
            int v = 0;
            if (m < pcs.M && k < pcs.K)
                v = castValue<T, int32_t>(A[int64_t(m) * pcs.a_stride_m + int64_t(k) * pcs.a_stride_k]);
            tileA[c * (TM + 1) + r] = v;
        }
        for (uint e = tid; e < TK * TN; e += THREADS)
        {
            uint r, c;  // r: row in the tile (k), c: column (n)
            if (pcs.b_stride_k == 1) { c = e / TK; r = e % TK; } else { r = e / TN; c = e % TN; }
            uint k = k0 + r, n = n0 + c;
            int v = 0;
            if (k < pcs.K && n < pcs.N)
                v = castValue<T, int32_t>(B[int64_t(k) * pcs.b_stride_k + int64_t(n) * pcs.b_stride_n]);
            tileB[r * (TN + 1) + c] = v;
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint kk = 0; kk < TK; kk++)
        {
            int a[4];
            int b[4];
            [unroll] for (uint i = 0; i < 4; i++)
                a[i] = tileA[kk * (TM + 1) + li.y * 4 + i];
            [unroll] for (uint j = 0; j < 4; j++)
                b[j] = tileB[kk * (TN + 1) + li.x * 4 + j];
            [unroll] for (uint i = 0; i < 4; i++)
                [unroll] for (uint j = 0; j < 4; j++)
                    acc[i][j] += a[i] * b[j];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    [unroll] for (uint i = 0; i < 4; i++)
    {
        uint m = m0 + li.y * 4 + i;
        if (m >= pcs.M) continue;
        [unroll] for (uint j = 0; j < 4; j++)
        {
            uint n = n0 + li.x * 4 + j;
            if (n < pcs.N)
                C[int64_t(m) * pcs.out_stride_m + int64_t(n)] = acc[i][j];
        }
    }
}
