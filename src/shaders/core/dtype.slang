module dtype;

/*
The dtypes kernels are written against.

A kernel takes `T : ITorchVulkan` and gets:

    T.ext_add(a, b), T.ext_exp(a), T.ext_lt(a, b), ...     every elementwise op
    castValue<T, U>(v)                                     conversion to any dtype
    T.Accum, v.toAccum(), T.fromAccum(acc)                 the type to accumulate in

The ops (IDtypeOps) are written once per family: builtin floats, builtin
integers and bfloat16. Each dtype also has a short ITorchVulkan conformance
that sets its Accum and conversions.

Integers implement the float-only ops through float32, so a kernel needs only
one entry point. The host promotes integers to float before those ops.

SPIR-V has no 64-bit exp/log/sin/pow, so float64 computes those ops in float32.

To add a dtype, give it an IDtypeOps and an ITorchVulkan conformance.
*/

public interface ITorchVulkan : IDtypeOps, ICastable
{
    associatedtype Accum : __BuiltinArithmeticType & ITorchVulkan;
    Accum toAccum();
    static This fromAccum(Accum v);
}

public interface IDtypeOps
{
    static This ext_pow(This a, This b);
    static This ext_max(This a, This b);
    static This ext_min(This a, This b);
    static This ext_zero();
    static This ext_one();
    static This ext_lowest();
    static This ext_highest();
    static This ext_neg(This a);
    static This ext_threshold_backward(This grad, This self, This threshold);
    static This ext_add(This a, This b);
    static This ext_sub(This a, This b);
    static This ext_mul(This a, This b);
    static This ext_div(This a, This b);
    static This ext_floor(This a);
    static This ext_ceil(This a);
    static This ext_trunc(This a);
    static This ext_round(This a);
    static This ext_frac(This a);
    static This ext_abs(This a);
    static This ext_sign(This a);
    static This ext_fmax(This a, This b);
    static This ext_fmin(This a, This b);
    static This ext_fmod(This a, This b);
    static This ext_remainder(This a, This b);
    static This ext_floor_divide(This a, This b);
    static bool ext_eq(This a, This b);
    static bool ext_ne(This a, This b);
    static bool ext_lt(This a, This b);
    static bool ext_le(This a, This b);
    static bool ext_gt(This a, This b);
    static bool ext_ge(This a, This b);
    static bool ext_nonzero(This a);
    static bool ext_isnan(This a);
    static This ext_atan2(This a, This b);
    static This ext_exp(This a);
    static This ext_log(This a);
    static This ext_sqrt(This a);
    static This ext_reciprocal(This a);
    static This ext_sin(This a);
    static This ext_cos(This a);
    static This ext_tan(This a);
    static This ext_asin(This a);
    static This ext_acos(This a);
    static This ext_atan(This a);
    static This ext_sinh(This a);
    static This ext_cosh(This a);
    static This ext_tanh(This a);
    static This ext_exp2(This a);
    static This ext_log2(This a);
    static This ext_log10(This a);
    static This ext_expm1(This a);
    static This ext_log1p(This a);
    static This ext_rsqrt(This a);
    static This ext_sigmoid(This a);
    static This ext_asinh(This a);
    static This ext_acosh(This a);
    static This ext_atanh(This a);
    static This ext_deg2rad(This a);
    static This ext_rad2deg(This a);
    static This ext_erf(This a);
    static This ext_erfc(This a);
    static This ext_erfcx(This a);
    static This ext_log_ndtr(This a);
    static This ext_erfinv(This a);
    static This ext_sinc(This a);
    static This ext_entr(This a);
    static This ext_lgamma(This a);
    static This ext_digamma(This a);
    static This ext_i0(This a);
    static This ext_i1(This a);
    static This ext_i0e(This a);
    static This ext_i1e(This a);
    static This ext_hypot(This a, This b);
    static This ext_xlogy(This a, This b);
    static This ext_logaddexp(This a, This b);
    static This ext_logaddexp2(This a, This b);
    static This ext_gelu(This a);
}

int64_t _saturate_f32(float v, int64_t lo, int64_t hi)
{
    if ((asuint(v) & 0x7fffffffu) > 0x7f800000u) return 0;
    if (v <= float(lo)) return lo;
    if (v >= float(hi)) return hi;
    return int64_t(v);
}

int64_t _saturate_f64(double v, int64_t lo, int64_t hi)
{
    if (v != v) return 0;
    if (v <= double(lo)) return lo;
    if (v >= double(hi)) return hi;
    return int64_t(v);
}

uint64_t _saturate_f32_u64(float v)
{
    if ((asuint(v) & 0x7fffffffu) > 0x7f800000u || v <= 0.0f) return 0;
    if (v >= 18446744073709551616.0f) return 0xffffffffffffffffull;
    return uint64_t(v);
}

uint64_t _saturate_f64_u64(double v)
{
    if (v != v || v <= 0.0) return 0;
    if (v >= 18446744073709551616.0) return 0xffffffffffffffffull;
    return uint64_t(v);
}

static const int64_t _INT64_MIN = int64_t(0x8000000000000000ull);
static const int64_t _INT64_MAX = 0x7fffffffffffffffll;

public extension int64_t : ITorchVulkan
{
    public typealias Accum = int64_t;
    public int64_t toAccum() { return this; }
    public static int64_t fromAccum(int64_t v) { return v; }

    public typealias Wide = int64_t;
    public int64_t widen() { return this; }
    public static int64_t castFromI64(int64_t v) { return v; }
    public static int64_t castFromU64(uint64_t v) { return int64_t(v); }
    public static int64_t castFromF32(float v) { return _saturate_f32(v, _INT64_MIN, _INT64_MAX); }
    public static int64_t castFromF64(double v) { return _saturate_f64(v, _INT64_MIN, _INT64_MAX); }
}

public extension int32_t : ITorchVulkan
{
    public typealias Accum = int32_t;
    public int32_t toAccum() { return this; }
    public static int32_t fromAccum(int32_t v) { return v; }

    public typealias Wide = int64_t;
    public int64_t widen() { return int64_t(this); }
    public static int32_t castFromI64(int64_t v) { return int32_t(v); }
    public static int32_t castFromU64(uint64_t v) { return int32_t(v); }
    public static int32_t castFromF32(float v) { return int32_t(_saturate_f32(v, -2147483648ll, 2147483647ll)); }
    public static int32_t castFromF64(double v) { return int32_t(_saturate_f64(v, -2147483648ll, 2147483647ll)); }
}

public extension int16_t : ITorchVulkan
{
    public typealias Accum = int16_t;
    public int16_t toAccum() { return this; }
    public static int16_t fromAccum(int16_t v) { return v; }

    public typealias Wide = int64_t;
    public int64_t widen() { return int64_t(this); }
    public static int16_t castFromI64(int64_t v) { return int16_t(v); }
    public static int16_t castFromU64(uint64_t v) { return int16_t(v); }
    public static int16_t castFromF32(float v) { return int16_t(_saturate_f32(v, -32768ll, 32767ll)); }
    public static int16_t castFromF64(double v) { return int16_t(_saturate_f64(v, -32768ll, 32767ll)); }
}

public extension int8_t : ITorchVulkan
{
    public typealias Accum = int8_t;
    public int8_t toAccum() { return this; }
    public static int8_t fromAccum(int8_t v) { return v; }

    public typealias Wide = int64_t;
    public int64_t widen() { return int64_t(this); }
    public static int8_t castFromI64(int64_t v) { return int8_t(v); }
    public static int8_t castFromU64(uint64_t v) { return int8_t(v); }
    public static int8_t castFromF32(float v) { return int8_t(_saturate_f32(v, -128ll, 127ll)); }
    public static int8_t castFromF64(double v) { return int8_t(_saturate_f64(v, -128ll, 127ll)); }
}

public extension uint64_t : ITorchVulkan
{
    public typealias Accum = uint64_t;
    public uint64_t toAccum() { return this; }
    public static uint64_t fromAccum(uint64_t v) { return v; }

    public typealias Wide = uint64_t;
    public uint64_t widen() { return this; }
    public static uint64_t castFromI64(int64_t v) { return uint64_t(v); }
    public static uint64_t castFromU64(uint64_t v) { return v; }
    public static uint64_t castFromF32(float v) { return _saturate_f32_u64(v); }
    public static uint64_t castFromF64(double v) { return _saturate_f64_u64(v); }
}

public extension uint32_t : ITorchVulkan
{
    public typealias Accum = uint32_t;
    public uint32_t toAccum() { return this; }
    public static uint32_t fromAccum(uint32_t v) { return v; }

    public typealias Wide = uint64_t;
    public uint64_t widen() { return uint64_t(this); }
    public static uint32_t castFromI64(int64_t v) { return uint32_t(v); }
    public static uint32_t castFromU64(uint64_t v) { return uint32_t(v); }
    public static uint32_t castFromF32(float v) { return uint32_t(_saturate_f32(v, 0ll, 4294967295ll)); }
    public static uint32_t castFromF64(double v) { return uint32_t(_saturate_f64(v, 0ll, 4294967295ll)); }
}

public extension uint16_t : ITorchVulkan
{
    public typealias Accum = uint16_t;
    public uint16_t toAccum() { return this; }
    public static uint16_t fromAccum(uint16_t v) { return v; }

    public typealias Wide = uint64_t;
    public uint64_t widen() { return uint64_t(this); }
    public static uint16_t castFromI64(int64_t v) { return uint16_t(v); }
    public static uint16_t castFromU64(uint64_t v) { return uint16_t(v); }
    public static uint16_t castFromF32(float v) { return uint16_t(_saturate_f32(v, 0ll, 65535ll)); }
    public static uint16_t castFromF64(double v) { return uint16_t(_saturate_f64(v, 0ll, 65535ll)); }
}

public extension uint8_t : ITorchVulkan
{
    public typealias Accum = uint8_t;
    public uint8_t toAccum() { return this; }
    public static uint8_t fromAccum(uint8_t v) { return v; }

    public typealias Wide = uint64_t;
    public uint64_t widen() { return uint64_t(this); }
    public static uint8_t castFromI64(int64_t v) { return uint8_t(v); }
    public static uint8_t castFromU64(uint64_t v) { return uint8_t(v); }
    public static uint8_t castFromF32(float v) { return uint8_t(_saturate_f32(v, 0ll, 255ll)); }
    public static uint8_t castFromF64(double v) { return uint8_t(_saturate_f64(v, 0ll, 255ll)); }
}

public extension double : ITorchVulkan
{
    public typealias Accum = double;
    public double toAccum() { return this; }
    public static double fromAccum(double v) { return v; }

    public typealias Wide = double;
    public double widen() { return this; }
    public static double castFromI64(int64_t v) { return double(v); }
    public static double castFromU64(uint64_t v) { return double(v); }
    public static double castFromF32(float v) { return double(v); }
    public static double castFromF64(double v) { return v; }
}

public extension float : ITorchVulkan
{
    public typealias Accum = float;
    public float toAccum() { return this; }
    public static float fromAccum(float v) { return v; }

    public typealias Wide = float;
    public float widen() { return this; }
    public static float castFromI64(int64_t v) { return float(v); }
    public static float castFromU64(uint64_t v) { return float(v); }
    public static float castFromF32(float v) { return v; }
    public static float castFromF64(double v) { return float(v); }
}

public extension half : ITorchVulkan
{
    // accumulate in float32
    public typealias Accum = float;
    public float toAccum() { return float(this); }
    public static half fromAccum(float v) { return half(v); }

    public typealias Wide = float;
    public float widen() { return float(this); }
    public static half castFromI64(int64_t v) { return half(v); }
    public static half castFromU64(uint64_t v) { return half(v); }
    public static half castFromF32(float v) { return half(v); }
    public static half castFromF64(double v) { return half(v); }
}

public extension BFloat16 : ITorchVulkan
{
    public typealias Accum = float;
    public float toAccum() { return float(this); }
    public static BFloat16 fromAccum(float v) { return BFloat16(v); }

    // bfloat16 converts natively only to and from float32, so everything else goes through it
    public typealias Wide = float;
    public float widen() { return float(this); }
    public static BFloat16 castFromI64(int64_t v) { return BFloat16(float(v)); }
    public static BFloat16 castFromU64(uint64_t v) { return BFloat16(float(v)); }
    public static BFloat16 castFromF32(float v) { return BFloat16(v); }
    public static BFloat16 castFromF64(double v) { return BFloat16(float(v)); }
}

// 2/pi from bit 31 on (bit j weighs 2^(31 - j)), for the exact argument reduction below
static const uint _TWO_OVER_PI[8] = { 0x00000000u, 0xA2F9836Eu, 0x4E441529u, 0xFC2757D1u, 0xF534DDC0u, 0xDB629599u, 0x3C439041u, 0xFE5163ABu };

// Payne-Hanek for |x| >= 2^18 (native sin/cos lose all accuracy from 2^22 on Metal): the 24-bit mantissa times the 96
// bits of 2/pi its exponent selects (higher bits only add multiples of 4) gives the quadrant and the remainder in Q64;
// returns the remainder in [-pi/4, pi/4]
float _reduce_half_pi(float x, out uint quadrant)
{
    uint b = asuint(x);
    uint64_t m = uint64_t((b & 0x7FFFFFu) | 0x800000u);
    uint j0 = ((b >> 23) & 0xFFu) - 120u;
    uint idx = j0 >> 5, sh = j0 & 31u;
    uint w0 = _TWO_OVER_PI[idx], w1 = _TWO_OVER_PI[idx + 1], w2 = _TWO_OVER_PI[idx + 2], w3 = _TWO_OVER_PI[idx + 3];
    if (sh != 0u) {
        w0 = (w0 << sh) | (w1 >> (32u - sh));
        w1 = (w1 << sh) | (w2 >> (32u - sh));
        w2 = (w2 << sh) | (w3 >> (32u - sh));
    }
    uint64_t p0 = m * uint64_t(w2), p1 = m * uint64_t(w1), p2 = m * uint64_t(w0);
    uint64_t lo = p0 + (p1 << 32);
    uint hi = uint((p1 >> 32) + p2 + (lo < p0 ? 1ull : 0ull));
    quadrant = hi >> 30;
    uint64_t f = (uint64_t(hi & 0x3FFFFFFFu) << 34) | (lo >> 30);
    if ((f >> 63) != 0ull) quadrant = (quadrant + 1u) & 3u;
    return float(int64_t(f)) * 5.421010862427522e-20 * 1.5707963267948966;
}

uint64_t _mulhi64(uint64_t a, uint64_t b)
{
    uint64_t a0 = a & 0xFFFFFFFFull, a1 = a >> 32, b0 = b & 0xFFFFFFFFull, b1 = b >> 32;
    uint64_t p01 = a0 * b1, p10 = a1 * b0;
    uint64_t mid = ((a0 * b0) >> 32) + (p01 & 0xFFFFFFFFull) + (p10 & 0xFFFFFFFFull);
    return a1 * b1 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

// pow for finite x > 0 to ~1 ulp (native pow is 16 ulp on Metal and exp2(y log2 x) elsewhere): y * log2(x) in 64-bit
// fixed point (log2 of the mantissa by the atanh series in Q64; integers, so fast math can't reassociate it), then
// 2^n * 2^f with |f| <= 1/2 by a float polynomial
float _accurate_pow(float x, float y)
{
    if (!(abs(y * log2(x)) < 512.0)) return pow(x, y);
    uint bx = asuint(x);
    uint64_t m = uint64_t(bx & 0x7FFFFFu);
    int e = int((bx >> 23) & 0xFFu) - 127;
    if (e == -127) {
        int shift = 23 - int(firstbithigh(uint(m)));
        m <<= shift;
        e = -126 - shift;
    } else {
        m |= 0x800000ull;
    }
    uint64_t d = 0x800000ull;
    if (m > 11863283ull) { d = 0x1000000ull; e += 1; }  // mantissa in [sqrt(1/2), sqrt(2))
    // |s| = |m - 1| / (m + 1) in Q64: the float quotient, corrected by the integer residual
    bool negative_log = m < d;
    uint64_t num = negative_log ? d - m : m - d, den = m + d;
    uint s0 = asuint(float(num) / float(den));
    uint64_t s = 0;
    if (num != 0) {
        s = uint64_t((s0 & 0x7FFFFFu) | 0x800000u) << uint(int(s0 >> 23) - 86);
        // in Q48 (s has >= 16 trailing zeros), so the residual of an inexact (fast-math) quotient can't overflow
        int64_t residual = int64_t((num << 48) - (s >> 16) * den);
        s += uint64_t(int64_t(float(residual) / float(den))) << 16;
    }
    uint64_t z = _mulhi64(s, s);
    // atanh(s) = s + s z (1/3 + z (1/5 + ... z / 13)), |s| <= 0.172
    uint64_t acc = 0x13B13B13B13B13B1ull;
    acc = 0x1745D1745D1745D1ull + _mulhi64(z, acc);
    acc = 0x1C71C71C71C71C71ull + _mulhi64(z, acc);
    acc = 0x2492492492492492ull + _mulhi64(z, acc);
    acc = 0x3333333333333333ull + _mulhi64(z, acc);
    acc = 0x5555555555555555ull + _mulhi64(z, acc);
    uint64_t atanh_s = s + _mulhi64(s, _mulhi64(z, acc));
    // |log2 m| = atanh(s) * 2 log2(e), the fraction 2 log2(e) - 2 in Q64
    uint64_t log2_m = (atanh_s << 1) + _mulhi64(atanh_s, 0xE2A8ECA5705FC2EEull);
    // log2 x in Q55, times y = Y * 2^ey into Q32
    int64_t log2_x = (int64_t(e) << 55) + (negative_log ? -int64_t(log2_m >> 9) : int64_t(log2_m >> 9));
    uint64_t magnitude = uint64_t(log2_x < 0 ? -log2_x : log2_x);
    uint by = asuint(y);
    int ey_field = int((by >> 23) & 0xFFu);
    uint64_t big_y = uint64_t(by & 0x7FFFFFu) | (ey_field != 0 ? 0x800000ull : 0ull);
    int shift = max(ey_field, 1) - 173;
    uint64_t lo = big_y * magnitude, hi = _mulhi64(big_y, magnitude);
    uint64_t t = 0;
    if (shift >= 0) { if (shift < 64) t = lo << uint(shift); }
    else if (shift > -64) { t = (lo >> uint(-shift)) | (hi << uint(64 + shift)); }
    else if (shift > -128) { t = hi >> uint(-shift - 64); }
    int64_t t_q32 = ((by >> 31) != 0) != (log2_x < 0) ? -int64_t(t) : int64_t(t);
    int64_t n = (t_q32 + int64_t(0x80000000ull)) >> 32;
    float f = float(t_q32 - (n << 32)) * 2.3283064365386963e-10;
    float p = 1.5252733646775596e-05;
    p = p * f + 1.540352968731895e-04;
    p = p * f + 1.3333557872101665e-03;
    p = p * f + 9.618128649890423e-03;
    p = p * f + 5.550410971045494e-02;
    p = p * f + 2.4022650718688965e-01;
    p = p * f + 6.931471824645996e-01;
    p = p * f + 1.0;
    // 2^n through the exponent bits (a product of two scale factors can be reassociated into an overflow)
    int exponent = int((asuint(p) >> 23) & 0xFFu) + int(n);
    if (exponent > 254) return asfloat(0x7F800000u);
    if (exponent >= 1) return asfloat(asuint(p) + (uint(int(n)) << 23));
    if (exponent < -24) return 0.0;
    return p * asfloat(uint(int(n) + 191) << 23) * 5.421010862427522e-20;  // subnormal: 2^(n + 64) * 2^-64
}

// exp(x) = 2^n exp(r), r = x - n ln 2 in two parts (fdlibm's split): the GPU's exp is exp2(x log2 e), and rounding that
// product costs about |x| ulp
float _exp_f32(float x)
{
    if (!(abs(x) < 104.0f)) return exp(x);
    float n = round(x * 1.44269504088896341f);
    float r = (x - n * 6.9313812256e-01f) - n * 9.0580006145e-06f;
    int ni = int(n);
    int h = ni / 2;
    // 2^n in two halves, so 2^128 can't overflow before exp(r) < 1 scales it back
    return exp(r) * asfloat(uint(h + 127) << 23) * asfloat(uint(ni - h + 127) << 23);
}

float _with_sign_of(float magnitude, float x) { return asfloat((asuint(magnitude) & 0x7fffffffu) | (asuint(x) & 0x80000000u)); }

// fdlibm's sinh and cosh: expm1 near zero, where (e^x - e^-x) / 2 cancels
float _sinh_f32(float x)
{
    float ax = abs(x);
    float r;
    if (ax < 22.0f) {
        float t = float.ext_expm1(ax);
        r = ax < 1.0f ? 0.5f * (2.0f * t - t * t / (t + 1.0f)) : 0.5f * (t + t / (t + 1.0f));
    } else if (ax < 88.0f) {
        r = 0.5f * _exp_f32(ax);
    } else {
        float w = _exp_f32(0.5f * ax);
        r = (0.5f * w) * w;
    }
    return _with_sign_of(r, x);
}

float _cosh_f32(float x)
{
    float ax = abs(x);
    if (ax < 0.34657359f) {
        float t = float.ext_expm1(ax);
        float w = 1.0f + t;
        return 1.0f + (t * t) / (w + w);
    }
    if (ax < 22.0f) {
        float t = _exp_f32(ax);
        return 0.5f * t + 0.5f / t;
    }
    if (ax < 88.0f) return 0.5f * _exp_f32(ax);
    float w = _exp_f32(0.5f * ax);
    return (0.5f * w) * w;
}

// sqrt(x^2 + y^2) of finite x, y >= 0 scaled by a power of 2 into range, then Borges' fma correction of the square root
// (nearly always correctly rounded; m sqrt(1 + r^2) is an ulp off often enough to flip isclose at a boundary)
float _hypot_f32(float x, float y)
{
    float m = max(x, y);
    float scale = 1.0f;
    if (m > 1.0e18f) scale = 1.4210854715202004e-14f;        // 2^-46
    else if (m < 1.0e-18f) scale = 70368744177664.0f;        // 2^46
    float a = x * scale;
    float b = y * scale;
    float h = sqrt(fma(a, a, b * b));
    float h_sq = h * h;
    float a_sq = a * a;
    float e = fma(-b, b, h_sq - a_sq) + fma(h, h, -h_sq) - fma(a, a, -a_sq);
    h = h - e / (2.0f * h);
    return h / scale;
}

public extension<T : __BuiltinFloatingPointType> T : IDtypeOps
{
    // C99 Annex F special cases; signed zeros and infinities from bit patterns (fast math assumes neither).
    // half and double compute in float32
    public static T ext_pow(T a, T b)
    {
        if (b == T(0.0) || a == T(1.0)) return T(1.0);
        if (isnan(a) || isnan(b)) return _nan();
        if (b == T(1.0)) return a;
        if (b == T(2.0)) return a * a;
        T inf = T(asfloat(0x7F800000u));
        if (isinf(b)) {
            if (abs(a) == T(1.0)) return T(1.0);
            return (abs(a) < T(1.0)) == (b < T(0.0)) ? inf : T(0.0);
        }
        bool odd = frac(abs(b) * T(0.5)) == T(0.5);
        bool negative = (asuint(a.toFloat()) >> 31) != 0;
        if (a == T(0.0) || isinf(a)) {
            // |a|^b is 0 or inf; an odd b keeps a's sign
            bool big = (a == T(0.0)) == (b < T(0.0));
            return T(asfloat((big ? 0x7F800000u : 0u) | (negative && odd ? 0x80000000u : 0u)));
        }
        if (!negative) return T(_accurate_pow(a.toFloat(), b.toFloat()));
        if (frac(b) != T(0.0)) return _nan();
        T res = T(_accurate_pow(-a.toFloat(), b.toFloat()));
        return odd ? -res : res;
    }

    // max/min propagate NaN (a + b is NaN if either is), fmax/fmin suppress it
    public static T ext_max(T a, T b) { if (isnan(a) || isnan(b)) return a + b; return max(a, b); }
    public static T ext_min(T a, T b) { if (isnan(a) || isnan(b)) return a + b; return min(a, b); }
    public static T ext_zero() { return T(0); }
    public static T ext_one() { return T(1); }

    public static T ext_lowest() { T zero = T(0); return T(-1.0) / zero; }
    public static T ext_highest() { T zero = T(0); return T(1.0) / zero; }

    public static T ext_neg(T a) { return -a; }
    public static T ext_threshold_backward(T grad, T self, T threshold) { return (self > threshold) ? grad : T(0); }
    public static T ext_add(T a, T b) { return a + b; }
    public static T ext_sub(T a, T b) { return a - b; }
    public static T ext_mul(T a, T b) { return a * b; }

    // a GPU may divide as a * (1 / b), and 1 / b flushes to zero once |b| passes 2^126: past 2^125, quarter both first. b is
    // quartered through its exponent bits, which a fast-math compiler can't fold back into a / b.
    public static T ext_div(T a, T b)
    {
        if (sizeof(T) == 4 && abs(b) > T(4.2535295865117308e37) && !isinf(b)) {
            float quarter_b = asfloat(asuint(b.toFloat()) - 0x01000000u);
            return T((a.toFloat() * 0.25f) / quarter_b);
        }
        return a / b;
    }

    public static T ext_floor(T a) { return floor(a); }
    public static T ext_ceil(T a) { return ceil(a); }
    public static T ext_trunc(T a) { return trunc(a); }

    public static T ext_round(T a)
    {
        T r = floor(a);
        T d = a - r;
        if (d < T(0.5)) return r;
        if (d > T(0.5)) return r + T(1);
        return (frac(r * T(0.5)) == T(0)) ? r : r + T(1);  // round half to even
    }

    public static T ext_frac(T a) { return a - trunc(a); }
    public static T ext_abs(T a) { return abs(a); }
    public static T ext_sign(T a) { if (isnan(a)) return a; return (a > T(0)) ? T(1) : ((a < T(0)) ? T(-1) : a); }

    public static T ext_fmax(T a, T b) { if (isnan(a)) return b; if (isnan(b)) return a; return max(a, b); }
    public static T ext_fmin(T a, T b) { if (isnan(a)) return b; if (isnan(b)) return a; return min(a, b); }
    // exact at any quotient size: t is the largest b * 2^k <= |a|, then |a| keeps t <= x < 2t,
    // so every subtraction is exact (Sterbenz). a - b * trunc(a / b) rounds once the quotient is large
    public static T ext_fmod(T a, T b)
    {
        if (sizeof(T) < 4) return T(float.ext_fmod(a.toFloat(), b.toFloat()));
        if (isnan(a) || isnan(b) || isinf(a) || b == T(0)) return T(0.0) / T(0.0);  // NaN

        T x = abs(a);
        T y = abs(b);
        if (isinf(y) || x < y) return a;

        T t = y;
        [loop] while (t * T(2) <= x) t = t * T(2);
        [loop] while (t >= y)
        {
            if (x >= t) x = x - t;
            t = t * T(0.5);
        }
        return (a < T(0)) ? -x : x;
    }

    // as torch computes it: fmod, then moved into the divisor's sign
    public static T ext_remainder(T a, T b)
    {
        T r = ext_fmod(a, b);
        if (r != T(0) && ((r < T(0)) != (b < T(0)))) r = r + b;
        return r;
    }

    // as torch computes it: the quotient of a - fmod(a, b) rounded to the nearest whole number,
    // with a zero quotient keeping the sign of a / b
    public static T ext_floor_divide(T a, T b)
    {
        if (sizeof(T) < 4) return T(float.ext_floor_divide(a.toFloat(), b.toFloat()));
        if (b == T(0)) return a / b;

        T mod = ext_fmod(a, b);
        T div = ext_div(a - mod, b);
        if (mod != T(0) && ((b < T(0)) != (mod < T(0)))) div = div - T(1);

        if (div != T(0))
        {
            T floordiv = floor(div);
            if (div - floordiv > T(0.5)) floordiv = floordiv + T(1);
            return floordiv;
        }
        T q = ext_div(a, b);
        bool negative = q < T(0) || (q == T(0) && T(1) / q < T(0));
        return negative ? -T(0) : T(0);
    }

    public static bool ext_eq(T a, T b) { return a == b; }
    public static bool ext_ne(T a, T b) { return a != b; }
    public static bool ext_lt(T a, T b) { return a < b; }
    public static bool ext_le(T a, T b) { return a <= b; }
    public static bool ext_gt(T a, T b) { return a > b; }
    public static bool ext_ge(T a, T b) { return a >= b; }
    public static bool ext_nonzero(T a) { return a != T(0); }
    public static bool ext_isnan(T a) { return isnan(a); }

    public static T ext_atan2(T a, T b)
    {
        // both zero: +-pi when b is -0, +-0 when it is +0, signed like a (C's atan2)
        if (a == T(0) && b == T(0)) {
            bool a_negative = (asuint(a.toFloat()) >> 31) != 0, b_negative = (asuint(b.toFloat()) >> 31) != 0;
            T magnitude = b_negative ? T(3.14159265358979) : T(0);
            return a_negative ? -magnitude : magnitude;
        }
        if (sizeof(T) != 4) return T(atan2(a.toFloat(), b.toFloat()));
        return atan2(a, b);
    }

    public static T ext_exp(T a) { if (sizeof(T) >= 4) return T(_exp_f32(a.toFloat())); return exp(a); }

    public static T ext_log(T a)
    {
        if (a < T(0.0)) return T(0.0) / T(0.0);   // NaN
        if (a == T(0.0)) return T(-1.0) / T(0.0); // -inf
        if (sizeof(T) == 8) return T(log(a.toFloat()));
        return log(a);
    }

    public static T ext_sqrt(T a) { return sqrt(a); }
    public static T ext_reciprocal(T a) { return ext_div(T(1.0), a); }
    // the GPU's sin and cos give 0 for inf and NaN: those are NaN, tested and built on the bits so fast math keeps them
    static bool _nonfinite(T a) { return (asuint(a.toFloat()) & 0x7F800000u) == 0x7F800000u; }
    static T _nan() { return T(asfloat(0x7FC00000u)); }
    public static T ext_sin(T a)
    {
        if (_nonfinite(a)) return _nan();
        float x = a.toFloat();
        if (abs(x) < 262144.0) return sizeof(T) == 8 ? T(sin(x)) : sin(a);
        uint q;
        float t = _reduce_half_pi(x, q);
        float r = (q & 1u) != 0 ? cos(t) : sin(t);
        return T(((q & 2u) != 0) != (x < 0.0) ? -r : r);
    }
    public static T ext_cos(T a)
    {
        if (_nonfinite(a)) return _nan();
        float x = a.toFloat();
        if (abs(x) < 262144.0) return sizeof(T) == 8 ? T(cos(x)) : cos(a);
        uint q;
        float t = _reduce_half_pi(x, q);
        float r = (q & 1u) != 0 ? sin(t) : cos(t);
        return T(((q + 1u) & 2u) != 0 ? -r : r);
    }
    public static T ext_tan(T a)
    {
        if (_nonfinite(a)) return _nan();
        float x = a.toFloat();
        if (abs(x) < 262144.0) return sizeof(T) == 8 ? T(tan(x)) : tan(a);
        uint q;
        float t = _reduce_half_pi(x, q);
        float r = (q & 1u) != 0 ? -cos(t) / sin(t) : tan(t);
        return T(x < 0.0 ? -r : r);
    }
    public static T ext_asin(T a) { if (sizeof(T) == 8) return T(asin(a.toFloat())); return asin(a); }
    public static T ext_acos(T a) { if (sizeof(T) == 8) return T(acos(a.toFloat())); return acos(a); }
    public static T ext_atan(T a) { if (sizeof(T) == 8) return T(atan(a.toFloat())); return atan(a); }
    public static T ext_sinh(T a) { return T(_sinh_f32(a.toFloat())); }
    public static T ext_cosh(T a) { return T(_cosh_f32(a.toFloat())); }
    public static T ext_tanh(T a)
    {
        float x = a.toFloat();
        // +-1 past 20 (already so in float): the builtin's exponentials overflow to inf / inf, in half from 6
        if (abs(x) > 20.0f) return T(x < 0.0f ? -1.0f : 1.0f);
        if (sizeof(T) != 4) return T(tanh(x));
        return tanh(a);
    }
    public static T ext_exp2(T a) { if (sizeof(T) == 8) return T(exp2(a.toFloat())); return exp2(a); }

    public static T ext_log2(T a)
    {
        if (a < T(0.0)) return T(0.0) / T(0.0);   // NaN
        if (a == T(0.0)) return T(-1.0) / T(0.0); // -inf
        if (sizeof(T) == 8) return T(log2(a.toFloat()));
        return log2(a);
    }

    public static T ext_log10(T a)
    {
        if (a < T(0.0)) return T(0.0) / T(0.0);   // NaN
        if (a == T(0.0)) return T(-1.0) / T(0.0); // -inf
        if (sizeof(T) == 8) return T(log10(a.toFloat()));
        return log10(a);
    }

    // a Taylor series near zero, where exp(a) - 1 cancels to nothing
    public static T ext_expm1(T a)
    {
        if (sizeof(T) < 4) return T(float.ext_expm1(a.toFloat()));
        if (abs(a) >= T(0.5)) return ext_exp(a) - T(1);
        T p = T(1.0 / 362880.0);
        p = T(1.0 / 40320.0) + a * p;
        p = T(1.0 / 5040.0) + a * p;
        p = T(1.0 / 720.0) + a * p;
        p = T(1.0 / 120.0) + a * p;
        p = T(1.0 / 24.0) + a * p;
        p = T(1.0 / 6.0) + a * p;
        p = T(0.5) + a * p;
        p = T(1) + a * p;
        return a * p;
    }

    // log1p(a) = 2 atanh(s) with s = a / (2 + a), a short odd series near zero, where log(1 + a) loses a
    public static T ext_log1p(T a)
    {
        if (sizeof(T) < 4) return T(float.ext_log1p(a.toFloat()));
        if (a < T(-0.25) || a > T(0.4)) return ext_log(T(1) + a);
        T s = a / (T(2) + a);
        T s2 = s * s;
        T p = T(1.0 / 15.0);
        p = T(1.0 / 13.0) + s2 * p;
        p = T(1.0 / 11.0) + s2 * p;
        p = T(1.0 / 9.0) + s2 * p;
        p = T(1.0 / 7.0) + s2 * p;
        p = T(1.0 / 5.0) + s2 * p;
        p = T(1.0 / 3.0) + s2 * p;
        p = T(1) + s2 * p;
        return T(2) * s * p;
    }
    public static T ext_rsqrt(T a) { return T(1) / ext_sqrt(a); }
    public static T ext_sigmoid(T a) { return T(1) / (T(1) + ext_exp(-a)); }
    // fdlibm's forms, in float: the builtins overflow past |x| ~ 1.8e19 (x * x) and lose small arguments to log(1 + y)
    public static T ext_asinh(T a)
    {
        float x = a.toFloat();
        float ax = abs(x);
        if (isnan(x) || isinf(x)) return T(x);
        float r;
        if (ax > 268435456.0f) r = float.ext_log(ax) + 0.693147180559945309f;  // 2^28: log(2|x|)
        else if (ax > 2.0f) r = float.ext_log(2.0f * ax + 1.0f / (sqrt(x * x + 1.0f) + ax));
        else r = float.ext_log1p(ax + x * x / (1.0f + sqrt(1.0f + x * x)));
        return T(x < 0.0f ? -r : r);
    }
    public static T ext_acosh(T a)
    {
        float x = a.toFloat();
        if (isnan(x) || x < 1.0f) return _nan();
        if (isinf(x)) return T(x);
        if (x > 268435456.0f) return T(float.ext_log(x) + 0.693147180559945309f);
        if (x > 2.0f) return T(float.ext_log(2.0f * x - 1.0f / (x + sqrt(x * x - 1.0f))));
        float t = x - 1.0f;
        return T(float.ext_log1p(t + sqrt(2.0f * t + t * t)));
    }
    public static T ext_atanh(T a)
    {
        float x = a.toFloat();
        float ax = abs(x);
        if (isnan(x) || ax > 1.0f) return _nan();
        if (ax == 1.0f) return T(x < 0.0f ? -asfloat(0x7F800000u) : asfloat(0x7F800000u));
        float r = ax < 0.5f ? 0.5f * float.ext_log1p(2.0f * ax + 2.0f * ax * ax / (1.0f - ax))
                            : 0.5f * float.ext_log1p(2.0f * ax / (1.0f - ax));
        return T(x < 0.0f ? -r : r);
    }
    // multiplied out: radians()/degrees() have no 64-bit form in SPIR-V
    public static T ext_deg2rad(T a) { return a * T(0.017453292519943295); }
    public static T ext_rad2deg(T a) { return a * T(57.29577951308232); }

    // the exponent of erfc(z) = t * exp(-z^2 + P(t)), t = 1 / (1 + z / 2), for z >= 0: Numerical Recipes'
    // erfcc Chebyshev fit, with fractional error under 1.2e-7 everywhere
    static float _erfc_poly(float t)
    {
        return -1.26551223f + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f + t * (-0.18628806f +
            t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f))))))));
    }

    // erfc(z) for z >= 0
    static float _erfc_pos(float z)
    {
        float t = 1.0f / (1.0f + 0.5f * z);
        return t * exp(-z * z + _erfc_poly(t));
    }

    // erf: its Maclaurin series below 0.5, where 1 - erfc would cancel, and 1 - erfc above
    public static T ext_erf(T a)
    {
        float x = a.toFloat();
        float ax = abs(x);
        if (ax < 0.5f)
        {
            float x2 = x * x;
            float p = -1.0f / 1320.0f;
            p = 1.0f / 216.0f + x2 * p;
            p = -1.0f / 42.0f + x2 * p;
            p = 1.0f / 10.0f + x2 * p;
            p = -1.0f / 3.0f + x2 * p;
            p = 1.0f + x2 * p;
            return T(1.1283791670955126f * x * p);  // 2 / sqrt(pi)
        }
        float y = 1.0f - _erfc_pos(ax);
        return T(x < 0.0f ? -y : y);
    }

    public static T ext_erfc(T a)
    {
        float x = a.toFloat();
        float e = _erfc_pos(abs(x));
        return T(x >= 0.0f ? e : 2.0f - e);
    }

    // erfcx(x) = exp(x^2) erfc(x): for x >= 0 the exp(-x^2) inside erfc cancels, so it never overflows
    public static T ext_erfcx(T a)
    {
        float x = a.toFloat();
        float z = abs(x);
        float t = 1.0f / (1.0f + 0.5f * z);
        float e = t * exp(_erfc_poly(t));
        return T(x >= 0.0f ? e : 2.0f * exp(x * x) - e);
    }

    // log(ndtr(x)), ndtr(x) = erfc(-x / sqrt 2) / 2: in log space for x < 0, so the far left tail stays finite
    public static T ext_log_ndtr(T a)
    {
        float x = a.toFloat();
        float w = -x * 0.7071067811865476f;
        if (w >= 0.0f)
        {
            float t = 1.0f / (1.0f + 0.5f * w);
            return T(-0.6931471805599453f + log(t) - w * w + _erfc_poly(t));  // log(1/2) + log(erfc(w))
        }
        return T(float.ext_log1p(-0.5f * _erfc_pos(-w)));
    }

    // erfinv: Giles (2010) two-branch polynomial approximation (branch split at w = 5)
    public static T ext_erfinv(T a)
    {
        float x = a.toFloat();
        if (x >= 1.0f || x <= -1.0f)
        {
            float zero = 0.0f;
            if (x == 1.0f) return T(1.0f / zero);
            if (x == -1.0f) return T(-1.0f / zero);
            return T(0.0f / zero);
        }

        float w = -log((1.0f - x) * (1.0f + x));
        float p;
        if (w < 5.0f)
        {
            w -= 2.5f;
            p = 2.81022636e-08f;
            p = 3.43273939e-07f + p * w;
            p = -3.5233877e-06f + p * w;
            p = -4.39150654e-06f + p * w;
            p = 0.00021858087f + p * w;
            p = -0.00125372503f + p * w;
            p = -0.00417768164f + p * w;
            p = 0.246640727f + p * w;
            p = 1.50140941f + p * w;
        }
        else
        {
            w = sqrt(w) - 3.0f;
            p = -0.000200214257f;
            p = 0.000100950558f + p * w;
            p = 0.00134934322f + p * w;
            p = -0.00367342844f + p * w;
            p = 0.00573950773f + p * w;
            p = -0.0076224613f + p * w;
            p = 0.00943887047f + p * w;
            p = 1.00167406f + p * w;
            p = 2.83297682f + p * w;
        }
        return T(p * x);
    }

    public static T ext_sinc(T a)
    {
        float x = a.toFloat();
        if (isinf(x)) return _nan();
        float px = 3.14159265358979f * x;  // 3.14159265358979 = pi
        // normalized sinc = sin(pi*x)/(pi*x), through ext_sin: the builtin sin loses large arguments
        return T(px == 0.0f ? 1.0f : float.ext_sin(px) / px);
    }

    public static T ext_entr(T a)
    {
        float x = a.toFloat();
        if (isnan(x)) return a;
        if (x > 0.0f) return T(-x * log(x));
        if (x == 0.0f) return T(0.0f);
        float zero = 0.0f;
        return T(-1.0f / zero);
    }

    // log-gamma: Lanczos approximation (g = 7, 9-term series), reflection formula for z < 0.5
    static float _lgamma_lanczos(float z0)
    {
        const float PI = 3.14159265358979f;
        bool reflect = z0 < 0.5f;
        float z = (reflect ? (1.0f - z0) : z0) - 1.0f;
        float t = z + 7.5f;
        float a = 0.99999999999980993f
            + 676.5203681218851f / (z + 1.0f) - 1259.1392167224028f / (z + 2.0f)
            + 771.32342877765313f / (z + 3.0f) - 176.61502916214059f / (z + 4.0f)
            + 12.507343278686905f / (z + 5.0f) - 0.13857109526572012f / (z + 6.0f)
            + 9.9843695780195716e-6f / (z + 7.0f) + 1.5056327351493116e-7f / (z + 8.0f);
        float lg = 0.5f * log(2.0f * PI) + (z + 0.5f) * log(t) - t + log(a);
        // |sin(pi x)| has period 1, so x - round(x) (exact) keeps the precision pi * x loses for large x
        if (reflect) return log(PI / abs(sin(PI * (z0 - round(z0))))) - lg;
        return lg;
    }

    public static T ext_lgamma(T a)
    {
        float x = a.toFloat();
        if ((asuint(x) & 0x7fffffffu) == 0x7f800000u) return T(asfloat(0x7f800000u));  // lgamma(+-inf) = +inf
        if (x <= 0.0f && x == floor(x)) { float zero = 0.0f; return T(1.0f / zero); }
        return T(_lgamma_lanczos(x));
    }

    // digamma: asymptotic (Stirling) series with Bernoulli terms 1/12, 1/120, 1/252, shifted to
    // x >= 6 by the recurrence psi(x) = psi(x+1) - 1/x, plus reflection for x < 0.5
    public static T ext_digamma(T a)
    {
        float x0 = a.toFloat();
        const float PI = 3.14159265358979f;
        // +-0 gives -+inf (copysign(inf, -x), as the CPU); other non-positive integers are poles
        if (x0 == 0.0f) return T(asfloat((asuint(x0) & 0x80000000u) != 0u ? 0x7f800000u : 0xff800000u));
        if (x0 < 0.0f && x0 == floor(x0)) return T(asfloat(0x7fc00000u));

        bool reflect = x0 < 0.5f;
        float x = reflect ? (1.0f - x0) : x0;
        float result = 0.0f;
        [loop] for (int i = 0; i < 8; i++) { if (x >= 6.0f) break; result -= 1.0f / x; x += 1.0f; }

        float inv = 1.0f / x;
        float inv2 = inv * inv;
        result += log(x) - 0.5f * inv - inv2 * (1.0f / 12.0f - inv2 * (1.0f / 120.0f - inv2 / 252.0f));
        // tan(pi x) has period 1: as the CPU, take it of the exact fractional part
        if (reflect) result -= PI / tan(PI * (x0 - round(x0)));
        return T(result);
    }

    // modified Bessel I0 scaled by exp(-|x|): Abramowitz & Stegun 9.8.1/9.8.2 polynomial approximation
    // (split at |x| = 3.75). Above the split the formula carries exp(|x|) as a factor, so the scaled
    // form drops it rather than multiplying an overflowed I0 by exp(-|x|).
    static float _i0e(float ax)
    {
        if (ax < 3.75f)
        {
            float t = (ax / 3.75f);
            t = t * t;
            return exp(-ax) * (1.0f + t * (3.5156229f + t * (3.0899424f + t * (1.2067492f + t * (0.2659732f + t * (0.0360768f + t * 0.0045813f))))));
        }
        float t = 3.75f / ax;
        return (0.39894228f + t * (0.01328592f + t * (0.00225319f + t * (-0.00157565f + t * (0.00916281f + t * (-0.02057706f + t * (0.02635537f + t * (-0.01647633f + t * 0.00392377f)))))))) / sqrt(ax);
    }

    // modified Bessel I1 scaled by exp(-|x|), for |x|: Abramowitz & Stegun 9.8.3/9.8.4, as _i0e
    static float _i1e(float ax)
    {
        if (ax < 3.75f)
        {
            float t = (ax / 3.75f);
            t = t * t;
            return exp(-ax) * ax * (0.5f + t * (0.87890594f + t * (0.51498869f + t * (0.15084934f + t * (0.02658733f + t * (0.00301532f + t * 0.00032411f))))));
        }
        float t = 3.75f / ax;
        return (0.39894228f + t * (-0.03988024f + t * (-0.00362018f + t * (0.00163801f + t * (-0.01031555f + t * (0.02282967f + t * (-0.02895312f + t * (0.01787654f + t * (-0.00420059f))))))))) / sqrt(ax);
    }

    public static T ext_i0(T a)
    {
        float ax = abs(a.toFloat());
        if (ax < 3.75f) return T(_i0e(ax) * exp(ax));
        return T(exp(ax) * _i0e(ax));
    }

    public static T ext_i1(T a)
    {
        float x = a.toFloat();
        float r = exp(abs(x)) * _i1e(abs(x));
        return T(x < 0.0f ? -r : r);
    }

    public static T ext_i0e(T a) { return T(_i0e(abs(a.toFloat()))); }

    public static T ext_i1e(T a)
    {
        float x = a.toFloat();
        float r = _i1e(abs(x));
        return T(x < 0.0f ? -r : r);
    }

    // scaled, so x * x + y * y can't overflow or underflow; infinity wins even over NaN, as in std::hypot
    public static T ext_hypot(T a, T b)
    {
        if (sizeof(T) < 4) return T(float.ext_hypot(a.toFloat(), b.toFloat()));
        T x = abs(a);
        T y = abs(b);
        if (isinf(x) || isinf(y)) return T(1.0) / T(0.0);
        if (isnan(x) || isnan(y)) return x + y;
        T m = max(x, y);
        if (m == T(0)) return T(0);
        if (sizeof(T) == 4) return T(_hypot_f32(x.toFloat(), y.toFloat()));
        // scaled by 2^-32 near the top of the range: GPUs divide via 1 / m, which flushes to zero once it's subnormal
        T scale = m > T(1.0e30) ? T(2.3283064365386963e-10) : T(1);
        T r = (min(x, y) * scale) / (m * scale);
        return m * sqrt(T(1) + r * r);
    }
    public static T ext_xlogy(T a, T b) { if (isnan(a)) return a; if (a == T(0)) return T(0); return a * ext_log(b); }
    public static T ext_logaddexp(T a, T b) { if (isnan(a) || isnan(b)) return a + b; T m = max(a, b); T n = min(a, b); if (isinf(m)) return m; return m + ext_log1p(ext_exp(n - m)); }
    public static T ext_logaddexp2(T a, T b) { if (isnan(a) || isnan(b)) return a + b; T m = max(a, b); T n = min(a, b); if (isinf(m)) return m; return m + ext_log2(T(1) + ext_exp2(n - m)); }

    // the exact (erf) form, as torch computes gelu with approximate='none'
    public static T ext_gelu(T a) { return T(0.5) * a * (T(1.0) + ext_erf(a * T(0.7071067811865476))); }
}

public extension<T : __BuiltinIntegerType> T : IDtypeOps
{
    static bool _is_signed() { return T(-1) < T(0); }

    // the sign bit for a signed T, and 0 for an unsigned one
    static T _sign_bit() { return _is_signed() ? T(T(1) << (sizeof(T) * 8 - 1)) : T(0); }

    // integer pow by squaring. A negative exponent cannot produce a fraction, so
    // it collapses to zero except where the base is its own reciprocal (+-1).
    public static T ext_pow(T a, T b)
    {
        // keep the exponent in T: a generic integer can't be converted to int32_t
        T e = b;
        if (e < T(0))
        {
            if (a == T(1)) return T(1);
            if (a == T(-1)) return ((e & T(1)) != T(0)) ? T(-1) : T(1);
            return T(0);
        }

        T res = T(1);
        T a_val = a;
        while (e > T(0))
        {
            if ((e & T(1)) != T(0)) res *= a_val;
            a_val *= a_val;
            e >>= 1;
        }
        return res;
    }

    public static T ext_max(T a, T b) { return max(a, b); }
    public static T ext_min(T a, T b) { return min(a, b); }
    public static T ext_zero() { return T(0); }
    public static T ext_one() { return T(1); }

    public static T ext_lowest() { return _sign_bit(); }
    public static T ext_highest() { return ~_sign_bit(); }

    public static T ext_neg(T a) { return -a; }
    public static T ext_threshold_backward(T grad, T self, T threshold) { return (self > threshold) ? grad : T(0); }
    public static T ext_add(T a, T b) { return a + b; }
    public static T ext_sub(T a, T b) { return a - b; }
    public static T ext_mul(T a, T b) { return a * b; }
    public static T ext_div(T a, T b) { return a / b; }

    // integers are already whole, so rounding is the identity and frac is zero
    public static T ext_floor(T a) { return a; }
    public static T ext_ceil(T a) { return a; }
    public static T ext_trunc(T a) { return a; }
    public static T ext_round(T a) { return a; }
    public static T ext_frac(T a) { return T(0); }

    // written out rather than calling abs(), which is not defined for unsigned types
    public static T ext_abs(T a) { return (a < T(0)) ? -a : a; }
    public static T ext_sign(T a) { return (a > T(0)) ? T(1) : ((a < T(0)) ? T(-1) : T(0)); }

    public static T ext_fmax(T a, T b) { return max(a, b); }
    public static T ext_fmin(T a, T b) { return min(a, b); }
    public static T ext_fmod(T a, T b) { if (b == T(0)) return T(0); return a % b; }

    // remainder takes the sign of the divisor where fmod takes the dividend's;
    // for unsigned T neither operand is ever negative, so the fixup never fires
    public static T ext_remainder(T a, T b)
    {
        if (b == T(0)) return T(0);
        T r = a % b;
        if (r != T(0) && ((r < T(0)) != (b < T(0)))) r += b;
        return r;
    }

    // truncation rounds toward zero, so a nonzero remainder with differing signs is one too high
    public static T ext_floor_divide(T a, T b)
    {
        if (b == T(0)) return T(0);
        T q = a / b;
        if ((a < T(0)) != (b < T(0)) && a % b != T(0)) q -= T(1);
        return q;
    }

    public static bool ext_eq(T a, T b) { return a == b; }
    public static bool ext_ne(T a, T b) { return a != b; }
    public static bool ext_lt(T a, T b) { return a < b; }
    public static bool ext_le(T a, T b) { return a <= b; }
    public static bool ext_gt(T a, T b) { return a > b; }
    public static bool ext_ge(T a, T b) { return a >= b; }
    public static bool ext_nonzero(T a) { return a != T(0); }
    public static bool ext_isnan(T a) { return false; }  // integers have no NaN

    // float-only ops, computed in float32 (the host never sends integers here)

    // __realCast picks the signed or unsigned conversion; int64_t bridges back
    static float _to_f32(T a) { return __realCast<float>(a); }
    static T _from_f32(float v) { return T(int64_t(v)); }

    public static T ext_atan2(T a, T b) { return _from_f32(float.ext_atan2(_to_f32(a), _to_f32(b))); }
    public static T ext_exp(T a) { return _from_f32(float.ext_exp(_to_f32(a))); }
    public static T ext_log(T a) { return _from_f32(float.ext_log(_to_f32(a))); }
    public static T ext_sqrt(T a) { return _from_f32(float.ext_sqrt(_to_f32(a))); }
    public static T ext_reciprocal(T a) { return _from_f32(float.ext_reciprocal(_to_f32(a))); }
    public static T ext_sin(T a) { return _from_f32(float.ext_sin(_to_f32(a))); }
    public static T ext_cos(T a) { return _from_f32(float.ext_cos(_to_f32(a))); }
    public static T ext_tan(T a) { return _from_f32(float.ext_tan(_to_f32(a))); }
    public static T ext_asin(T a) { return _from_f32(float.ext_asin(_to_f32(a))); }
    public static T ext_acos(T a) { return _from_f32(float.ext_acos(_to_f32(a))); }
    public static T ext_atan(T a) { return _from_f32(float.ext_atan(_to_f32(a))); }
    public static T ext_sinh(T a) { return _from_f32(float.ext_sinh(_to_f32(a))); }
    public static T ext_cosh(T a) { return _from_f32(float.ext_cosh(_to_f32(a))); }
    public static T ext_tanh(T a) { return _from_f32(float.ext_tanh(_to_f32(a))); }
    public static T ext_exp2(T a) { return _from_f32(float.ext_exp2(_to_f32(a))); }
    public static T ext_log2(T a) { return _from_f32(float.ext_log2(_to_f32(a))); }
    public static T ext_log10(T a) { return _from_f32(float.ext_log10(_to_f32(a))); }
    public static T ext_expm1(T a) { return _from_f32(float.ext_expm1(_to_f32(a))); }
    public static T ext_log1p(T a) { return _from_f32(float.ext_log1p(_to_f32(a))); }
    public static T ext_rsqrt(T a) { return _from_f32(float.ext_rsqrt(_to_f32(a))); }
    public static T ext_sigmoid(T a) { return _from_f32(float.ext_sigmoid(_to_f32(a))); }
    public static T ext_asinh(T a) { return _from_f32(float.ext_asinh(_to_f32(a))); }
    public static T ext_acosh(T a) { return _from_f32(float.ext_acosh(_to_f32(a))); }
    public static T ext_atanh(T a) { return _from_f32(float.ext_atanh(_to_f32(a))); }
    public static T ext_deg2rad(T a) { return _from_f32(float.ext_deg2rad(_to_f32(a))); }
    public static T ext_rad2deg(T a) { return _from_f32(float.ext_rad2deg(_to_f32(a))); }
    public static T ext_erf(T a) { return _from_f32(float.ext_erf(_to_f32(a))); }
    public static T ext_erfc(T a) { return _from_f32(float.ext_erfc(_to_f32(a))); }
    public static T ext_erfcx(T a) { return _from_f32(float.ext_erfcx(_to_f32(a))); }
    public static T ext_log_ndtr(T a) { return _from_f32(float.ext_log_ndtr(_to_f32(a))); }
    public static T ext_erfinv(T a) { return _from_f32(float.ext_erfinv(_to_f32(a))); }
    public static T ext_sinc(T a) { return _from_f32(float.ext_sinc(_to_f32(a))); }
    public static T ext_entr(T a) { return _from_f32(float.ext_entr(_to_f32(a))); }
    public static T ext_lgamma(T a) { return _from_f32(float.ext_lgamma(_to_f32(a))); }
    public static T ext_digamma(T a) { return _from_f32(float.ext_digamma(_to_f32(a))); }
    public static T ext_i0(T a) { return _from_f32(float.ext_i0(_to_f32(a))); }
    public static T ext_i1(T a) { return _from_f32(float.ext_i1(_to_f32(a))); }
    public static T ext_i0e(T a) { return _from_f32(float.ext_i0e(_to_f32(a))); }
    public static T ext_i1e(T a) { return _from_f32(float.ext_i1e(_to_f32(a))); }
    public static T ext_hypot(T a, T b) { return _from_f32(float.ext_hypot(_to_f32(a), _to_f32(b))); }
    public static T ext_xlogy(T a, T b) { return _from_f32(float.ext_xlogy(_to_f32(a), _to_f32(b))); }
    public static T ext_logaddexp(T a, T b) { return _from_f32(float.ext_logaddexp(_to_f32(a), _to_f32(b))); }
    public static T ext_logaddexp2(T a, T b) { return _from_f32(float.ext_logaddexp2(_to_f32(a), _to_f32(b))); }
    public static T ext_gelu(T a) { return _from_f32(float.ext_gelu(_to_f32(a))); }
}

public extension BFloat16 : IDtypeOps
{
    public static BFloat16 ext_pow(BFloat16 a, BFloat16 b) { return BFloat16(float.ext_pow(float(a), float(b))); }
    public static BFloat16 ext_max(BFloat16 a, BFloat16 b) { return BFloat16(float.ext_max(float(a), float(b))); }
    public static BFloat16 ext_min(BFloat16 a, BFloat16 b) { return BFloat16(float.ext_min(float(a), float(b))); }
    public static BFloat16 ext_zero() { return BFloat16(0.0f); }
    public static BFloat16 ext_one() { return BFloat16(1.0f); }
    public static BFloat16 ext_lowest() { return BFloat16(float.ext_lowest()); }
    public static BFloat16 ext_highest() { return BFloat16(float.ext_highest()); }
    public static BFloat16 ext_neg(BFloat16 a) { return BFloat16(-float(a)); }
    public static BFloat16 ext_threshold_backward(BFloat16 grad, BFloat16 self, BFloat16 threshold) { return BFloat16(float.ext_threshold_backward(float(grad), float(self), float(threshold))); }
    public static BFloat16 ext_add(BFloat16 a, BFloat16 b) { return BFloat16(float(a) + float(b)); }
    public static BFloat16 ext_sub(BFloat16 a, BFloat16 b) { return BFloat16(float(a) - float(b)); }
    public static BFloat16 ext_mul(BFloat16 a, BFloat16 b) { return BFloat16(float(a) * float(b)); }
    public static BFloat16 ext_div(BFloat16 a, BFloat16 b) { return BFloat16(float(a) / float(b)); }
    public static BFloat16 ext_floor(BFloat16 a) { return BFloat16(floor(float(a))); }
    public static BFloat16 ext_ceil(BFloat16 a) { return BFloat16(ceil(float(a))); }
    public static BFloat16 ext_trunc(BFloat16 a) { return BFloat16(trunc(float(a))); }
    public static BFloat16 ext_round(BFloat16 a) { return BFloat16(float.ext_round(float(a))); }
    public static BFloat16 ext_frac(BFloat16 a) { return BFloat16(float.ext_frac(float(a))); }
    public static BFloat16 ext_abs(BFloat16 a) { return BFloat16(abs(float(a))); }
    public static BFloat16 ext_sign(BFloat16 a) { return BFloat16(float.ext_sign(float(a))); }
    public static BFloat16 ext_fmax(BFloat16 a, BFloat16 b) { return BFloat16(float.ext_fmax(float(a), float(b))); }
    public static BFloat16 ext_fmin(BFloat16 a, BFloat16 b) { return BFloat16(float.ext_fmin(float(a), float(b))); }
    public static BFloat16 ext_fmod(BFloat16 a, BFloat16 b) { return BFloat16(float.ext_fmod(float(a), float(b))); }
    public static BFloat16 ext_remainder(BFloat16 a, BFloat16 b) { return BFloat16(float.ext_remainder(float(a), float(b))); }
    public static BFloat16 ext_floor_divide(BFloat16 a, BFloat16 b) { return BFloat16(float.ext_floor_divide(float(a), float(b))); }
    public static bool ext_eq(BFloat16 a, BFloat16 b) { return float(a) == float(b); }
    public static bool ext_ne(BFloat16 a, BFloat16 b) { return float(a) != float(b); }
    public static bool ext_lt(BFloat16 a, BFloat16 b) { return float(a) < float(b); }
    public static bool ext_le(BFloat16 a, BFloat16 b) { return float(a) <= float(b); }
    public static bool ext_gt(BFloat16 a, BFloat16 b) { return float(a) > float(b); }
    public static bool ext_ge(BFloat16 a, BFloat16 b) { return float(a) >= float(b); }
    public static bool ext_nonzero(BFloat16 a) { return float(a) != 0.0f; }
    public static bool ext_isnan(BFloat16 a) { return isnan(float(a)); }
    public static BFloat16 ext_atan2(BFloat16 a, BFloat16 b) { return BFloat16(float.ext_atan2(float(a), float(b))); }
    public static BFloat16 ext_exp(BFloat16 a) { return BFloat16(float.ext_exp(float(a))); }
    public static BFloat16 ext_log(BFloat16 a) { return BFloat16(float.ext_log(float(a))); }
    public static BFloat16 ext_sqrt(BFloat16 a) { return BFloat16(float.ext_sqrt(float(a))); }
    public static BFloat16 ext_reciprocal(BFloat16 a) { return BFloat16(float.ext_reciprocal(float(a))); }
    public static BFloat16 ext_sin(BFloat16 a) { return BFloat16(float.ext_sin(float(a))); }
    public static BFloat16 ext_cos(BFloat16 a) { return BFloat16(float.ext_cos(float(a))); }
    public static BFloat16 ext_tan(BFloat16 a) { return BFloat16(float.ext_tan(float(a))); }
    public static BFloat16 ext_asin(BFloat16 a) { return BFloat16(float.ext_asin(float(a))); }
    public static BFloat16 ext_acos(BFloat16 a) { return BFloat16(float.ext_acos(float(a))); }
    public static BFloat16 ext_atan(BFloat16 a) { return BFloat16(float.ext_atan(float(a))); }
    public static BFloat16 ext_sinh(BFloat16 a) { return BFloat16(float.ext_sinh(float(a))); }
    public static BFloat16 ext_cosh(BFloat16 a) { return BFloat16(float.ext_cosh(float(a))); }
    public static BFloat16 ext_tanh(BFloat16 a) { return BFloat16(float.ext_tanh(float(a))); }
    public static BFloat16 ext_exp2(BFloat16 a) { return BFloat16(float.ext_exp2(float(a))); }
    public static BFloat16 ext_log2(BFloat16 a) { return BFloat16(float.ext_log2(float(a))); }
    public static BFloat16 ext_log10(BFloat16 a) { return BFloat16(float.ext_log10(float(a))); }
    public static BFloat16 ext_expm1(BFloat16 a) { return BFloat16(float.ext_expm1(float(a))); }
    public static BFloat16 ext_log1p(BFloat16 a) { return BFloat16(float.ext_log1p(float(a))); }
    public static BFloat16 ext_rsqrt(BFloat16 a) { return BFloat16(float.ext_rsqrt(float(a))); }
    public static BFloat16 ext_sigmoid(BFloat16 a) { return BFloat16(float.ext_sigmoid(float(a))); }
    public static BFloat16 ext_asinh(BFloat16 a) { return BFloat16(float.ext_asinh(float(a))); }
    public static BFloat16 ext_acosh(BFloat16 a) { return BFloat16(float.ext_acosh(float(a))); }
    public static BFloat16 ext_atanh(BFloat16 a) { return BFloat16(float.ext_atanh(float(a))); }
    public static BFloat16 ext_deg2rad(BFloat16 a) { return BFloat16(float.ext_deg2rad(float(a))); }
    public static BFloat16 ext_rad2deg(BFloat16 a) { return BFloat16(float.ext_rad2deg(float(a))); }
    public static BFloat16 ext_erf(BFloat16 a) { return BFloat16(float.ext_erf(float(a))); }
    public static BFloat16 ext_erfc(BFloat16 a) { return BFloat16(float.ext_erfc(float(a))); }
    public static BFloat16 ext_erfcx(BFloat16 a) { return BFloat16(float.ext_erfcx(float(a))); }
    public static BFloat16 ext_log_ndtr(BFloat16 a) { return BFloat16(float.ext_log_ndtr(float(a))); }
    public static BFloat16 ext_erfinv(BFloat16 a) { return BFloat16(float.ext_erfinv(float(a))); }
    public static BFloat16 ext_sinc(BFloat16 a) { return BFloat16(float.ext_sinc(float(a))); }
    public static BFloat16 ext_entr(BFloat16 a) { return BFloat16(float.ext_entr(float(a))); }
    public static BFloat16 ext_lgamma(BFloat16 a) { return BFloat16(float.ext_lgamma(float(a))); }
    public static BFloat16 ext_digamma(BFloat16 a) { return BFloat16(float.ext_digamma(float(a))); }
    public static BFloat16 ext_i0(BFloat16 a) { return BFloat16(float.ext_i0(float(a))); }
    public static BFloat16 ext_i1(BFloat16 a) { return BFloat16(float.ext_i1(float(a))); }
    public static BFloat16 ext_i0e(BFloat16 a) { return BFloat16(float.ext_i0e(float(a))); }
    public static BFloat16 ext_i1e(BFloat16 a) { return BFloat16(float.ext_i1e(float(a))); }
    public static BFloat16 ext_hypot(BFloat16 a, BFloat16 b) { return BFloat16(float.ext_hypot(float(a), float(b))); }
    public static BFloat16 ext_xlogy(BFloat16 a, BFloat16 b) { return BFloat16(float.ext_xlogy(float(a), float(b))); }
    public static BFloat16 ext_logaddexp(BFloat16 a, BFloat16 b) { return BFloat16(float.ext_logaddexp(float(a), float(b))); }
    public static BFloat16 ext_logaddexp2(BFloat16 a, BFloat16 b) { return BFloat16(float.ext_logaddexp2(float(a), float(b))); }
    public static BFloat16 ext_gelu(BFloat16 a) { return BFloat16(float.ext_gelu(float(a))); }
}

// conversion between any two dtypes
__include "cast.slang";
