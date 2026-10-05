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
    static This ext_erfinv(This a);
    static This ext_sinc(This a);
    static This ext_entr(This a);
    static This ext_lgamma(This a);
    static This ext_digamma(This a);
    static This ext_i0(This a);
    static This ext_i1(This a);
    static This ext_hypot(This a, This b);
    static This ext_xlogy(This a, This b);
    static This ext_logaddexp(This a, This b);
    static This ext_logaddexp2(This a, This b);
    static This ext_gelu(This a);
}

public extension int64_t : ITorchVulkan
{
    public typealias Accum = int64_t;
    public int64_t toAccum() { return this; }
    public static int64_t fromAccum(int64_t v) { return v; }

    public typealias Wide = int64_t;
    public int64_t widen() { return this; }
    public static int64_t castFromI64(int64_t v) { return v; }
    public static int64_t castFromU64(uint64_t v) { return int64_t(v); }
    public static int64_t castFromF32(float v) { return int64_t(v); }
    public static int64_t castFromF64(double v) { return int64_t(v); }
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
    public static int32_t castFromF32(float v) { return int32_t(v); }
    public static int32_t castFromF64(double v) { return int32_t(v); }
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
    public static int16_t castFromF32(float v) { return int16_t(v); }
    public static int16_t castFromF64(double v) { return int16_t(v); }
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
    public static int8_t castFromF32(float v) { return int8_t(v); }
    public static int8_t castFromF64(double v) { return int8_t(v); }
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
    public static uint64_t castFromF32(float v) { return uint64_t(v); }
    public static uint64_t castFromF64(double v) { return uint64_t(v); }
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
    public static uint32_t castFromF32(float v) { return uint32_t(v); }
    public static uint32_t castFromF64(double v) { return uint32_t(v); }
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
    public static uint16_t castFromF32(float v) { return uint16_t(v); }
    public static uint16_t castFromF64(double v) { return uint16_t(v); }
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
    public static uint8_t castFromF32(float v) { return uint8_t(v); }
    public static uint8_t castFromF64(double v) { return uint8_t(v); }
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

public extension<T : __BuiltinFloatingPointType> T : IDtypeOps
{
    // half and double have no native pow, so both route through float32
    public static T _safe_pow(T base, T exp)
    {
        if (sizeof(T) != 4) return T(pow(base.toFloat(), exp.toFloat()));
        return pow(base, exp);
    }

    public static T ext_pow(T a, T b)
    {
        if (b == T(0.0)) return T(1.0);
        else if (b == T(1.0)) return a;
        else if (b == T(2.0)) return a * a;
        else if (a == T(1.0)) return T(1.0);
        else if (isnan(a) || isnan(b)) return T(0.0) / T(0.0);
        else if (a == T(0.0))
        {
            if (b < T(0.0)) return T(1.0) / T(0.0);
            return T(0.0);
        }
        else if (a > T(0.0)) { return _safe_pow(a, b); }
        else if (isinf(b)) { return _safe_pow(abs(a), b); }
        else if (frac(b) != T(0.0)) { return T(0.0) / T(0.0); /* NaN */ }

        T res = _safe_pow(abs(a), b);
        if (fmod(abs(b), T(2.0)) == T(1.0)) { return -res; }
        return res;
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
    public static T ext_div(T a, T b) { return a / b; }

    public static T ext_floor(T a) { return floor(a); }
    public static T ext_ceil(T a) { return ceil(a); }
    public static T ext_trunc(T a) { return trunc(a); }

    public static T ext_round(T a)
    {
        T r = floor(a);
        T d = a - r;
        if (d < T(0.5)) return r;
        if (d > T(0.5)) return r + T(1);
        return (fmod(r, T(2)) == T(0)) ? r : r + T(1);  // round half to even
    }

    public static T ext_frac(T a) { return a - trunc(a); }
    public static T ext_abs(T a) { return abs(a); }
    public static T ext_sign(T a) { if (isnan(a)) return a; return (a > T(0)) ? T(1) : ((a < T(0)) ? T(-1) : a); }

    public static T ext_fmax(T a, T b) { if (isnan(a)) return b; if (isnan(b)) return a; return max(a, b); }
    public static T ext_fmin(T a, T b) { if (isnan(a)) return b; if (isnan(b)) return a; return min(a, b); }
    public static T ext_fmod(T a, T b) { if (sizeof(T) < 4) { float x = a.toFloat(); float y = b.toFloat(); return T(x - y * trunc(x / y)); } return a - b * trunc(a / b); }
    public static T ext_remainder(T a, T b) { if (sizeof(T) < 4) { float x = a.toFloat(); float y = b.toFloat(); return T(x - y * floor(x / y)); } return a - b * floor(a / b); }

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
        if (a == T(0) && b == T(0)) return T(0);
        if (sizeof(T) != 4) return T(atan2(a.toFloat(), b.toFloat()));
        return atan2(a, b);
    }

    public static T ext_exp(T a) { if (sizeof(T) == 8) return T(exp(a.toFloat())); return exp(a); }

    public static T ext_log(T a)
    {
        if (a < T(0.0)) return T(0.0) / T(0.0);   // NaN
        if (a == T(0.0)) return T(-1.0) / T(0.0); // -inf
        if (sizeof(T) == 8) return T(log(a.toFloat()));
        return log(a);
    }

    public static T ext_sqrt(T a) { return sqrt(a); }
    public static T ext_reciprocal(T a) { return T(1.0) / a; }
    public static T ext_sin(T a) { if (sizeof(T) == 8) return T(sin(a.toFloat())); return sin(a); }
    public static T ext_cos(T a) { if (sizeof(T) == 8) return T(cos(a.toFloat())); return cos(a); }
    public static T ext_tan(T a) { if (sizeof(T) == 8) return T(tan(a.toFloat())); return tan(a); }
    public static T ext_asin(T a) { if (sizeof(T) == 8) return T(asin(a.toFloat())); return asin(a); }
    public static T ext_acos(T a) { if (sizeof(T) == 8) return T(acos(a.toFloat())); return acos(a); }
    public static T ext_atan(T a) { if (sizeof(T) == 8) return T(atan(a.toFloat())); return atan(a); }
    public static T ext_sinh(T a) { if (sizeof(T) == 8) return T(sinh(a.toFloat())); return sinh(a); }
    public static T ext_cosh(T a) { if (sizeof(T) == 8) return T(cosh(a.toFloat())); return cosh(a); }
    public static T ext_tanh(T a) { if (sizeof(T) == 8) return T(tanh(a.toFloat())); return tanh(a); }
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

    public static T ext_expm1(T a) { return ext_exp(a) - T(1); }
    public static T ext_log1p(T a) { return ext_log(T(1) + a); }
    public static T ext_rsqrt(T a) { return T(1) / ext_sqrt(a); }
    public static T ext_sigmoid(T a) { return T(1) / (T(1) + ext_exp(-a)); }
    public static T ext_asinh(T a) { if (sizeof(T) == 8) return T(asinh(a.toFloat())); return asinh(a); }
    public static T ext_acosh(T a) { if (sizeof(T) == 8) return T(acosh(a.toFloat())); return acosh(a); }
    public static T ext_atanh(T a) { if (sizeof(T) == 8) return T(atanh(a.toFloat())); return atanh(a); }
    // multiplied out: radians()/degrees() have no 64-bit form in SPIR-V
    public static T ext_deg2rad(T a) { return a * T(0.017453292519943295); }
    public static T ext_rad2deg(T a) { return a * T(57.29577951308232); }

    // erf: Abramowitz & Stegun 7.1.26 rational-times-exp approximation (|err| < 1.5e-7)
    public static T ext_erf(T a)
    {
        float x = a.toFloat();
        float ax = abs(x);
        float t = 1.0f / (1.0f + 0.3275911f * ax);
        float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-ax * ax);
        return T(x < 0.0f ? -y : y);
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
        float px = 3.14159265358979f * a.toFloat();  // 3.14159265358979 = pi
        return T(px == 0.0f ? 1.0f : sin(px) / px);  // normalized sinc = sin(pi*x)/(pi*x)
    }

    public static T ext_entr(T a)
    {
        float x = a.toFloat();
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
        if (reflect) return log(PI / abs(sin(PI * z0))) - lg;
        return lg;
    }

    public static T ext_lgamma(T a)
    {
        float x = a.toFloat();
        if (x <= 0.0f && x == floor(x)) { float zero = 0.0f; return T(1.0f / zero); }
        return T(_lgamma_lanczos(x));
    }

    // digamma: asymptotic (Stirling) series with Bernoulli terms 1/12, 1/120, 1/252, shifted to
    // x >= 6 by the recurrence psi(x) = psi(x+1) - 1/x, plus reflection for x < 0.5
    public static T ext_digamma(T a)
    {
        float x0 = a.toFloat();
        const float PI = 3.14159265358979f;
        if (x0 <= 0.0f && x0 == floor(x0)) { float zero = 0.0f; return T(x0 == 0.0f ? (-1.0f / zero) : (0.0f / zero)); }

        bool reflect = x0 < 0.5f;
        float x = reflect ? (1.0f - x0) : x0;
        float result = 0.0f;
        [loop] for (int i = 0; i < 8; i++) { if (x >= 6.0f) break; result -= 1.0f / x; x += 1.0f; }

        float inv = 1.0f / x;
        float inv2 = inv * inv;
        result += log(x) - 0.5f * inv - inv2 * (1.0f / 12.0f - inv2 * (1.0f / 120.0f - inv2 / 252.0f));
        if (reflect) result -= PI / tan(PI * x0);
        return T(result);
    }

    // modified Bessel I0: Abramowitz & Stegun 9.8.1/9.8.2 polynomial approximation (split at |x| = 3.75)
    public static T ext_i0(T a)
    {
        float ax = abs(a.toFloat());
        if (ax < 3.75f)
        {
            float t = (ax / 3.75f);
            t = t * t;
            return T(1.0f + t * (3.5156229f + t * (3.0899424f + t * (1.2067492f + t * (0.2659732f + t * (0.0360768f + t * 0.0045813f))))));
        }
        float t = 3.75f / ax;
        return T((exp(ax) / sqrt(ax)) * (0.39894228f + t * (0.01328592f + t * (0.00225319f + t * (-0.00157565f + t * (0.00916281f + t * (-0.02057706f + t * (0.02635537f + t * (-0.01647633f + t * 0.00392377f)))))))));
    }

    // modified Bessel I1: Abramowitz & Stegun 9.8.3/9.8.4 polynomial approximation (split at |x| = 3.75)
    public static T ext_i1(T a)
    {
        float x = a.toFloat();
        float ax = abs(x);
        float r;
        if (ax < 3.75f)
        {
            float t = (ax / 3.75f);
            t = t * t;
            r = ax * (0.5f + t * (0.87890594f + t * (0.51498869f + t * (0.15084934f + t * (0.02658733f + t * (0.00301532f + t * 0.00032411f))))));
        }
        else
        {
            float t = 3.75f / ax;
            r = (exp(ax) / sqrt(ax)) * (0.39894228f + t * (-0.03988024f + t * (-0.00362018f + t * (0.00163801f + t * (-0.01031555f + t * (0.02282967f + t * (-0.02895312f + t * (0.01787654f + t * (-0.00420059f)))))))));
        }
        return T(x < 0.0f ? -r : r);
    }

    public static T ext_hypot(T a, T b) { float x = a.toFloat(); float y = b.toFloat(); return T(sqrt(x * x + y * y)); }
    public static T ext_xlogy(T a, T b) { if (isnan(a)) return a; if (a == T(0)) return T(0); return a * ext_log(b); }
    public static T ext_logaddexp(T a, T b) { T m = max(a, b); T n = min(a, b); if (isinf(m)) return m; return m + ext_log1p(ext_exp(n - m)); }
    public static T ext_logaddexp2(T a, T b) { T m = max(a, b); T n = min(a, b); if (isinf(m)) return m; return m + ext_log2(T(1) + ext_exp2(n - m)); }

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
    public static T ext_erfinv(T a) { return _from_f32(float.ext_erfinv(_to_f32(a))); }
    public static T ext_sinc(T a) { return _from_f32(float.ext_sinc(_to_f32(a))); }
    public static T ext_entr(T a) { return _from_f32(float.ext_entr(_to_f32(a))); }
    public static T ext_lgamma(T a) { return _from_f32(float.ext_lgamma(_to_f32(a))); }
    public static T ext_digamma(T a) { return _from_f32(float.ext_digamma(_to_f32(a))); }
    public static T ext_i0(T a) { return _from_f32(float.ext_i0(_to_f32(a))); }
    public static T ext_i1(T a) { return _from_f32(float.ext_i1(_to_f32(a))); }
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
    public static BFloat16 ext_erfinv(BFloat16 a) { return BFloat16(float.ext_erfinv(float(a))); }
    public static BFloat16 ext_sinc(BFloat16 a) { return BFloat16(float.ext_sinc(float(a))); }
    public static BFloat16 ext_entr(BFloat16 a) { return BFloat16(float.ext_entr(float(a))); }
    public static BFloat16 ext_lgamma(BFloat16 a) { return BFloat16(float.ext_lgamma(float(a))); }
    public static BFloat16 ext_digamma(BFloat16 a) { return BFloat16(float.ext_digamma(float(a))); }
    public static BFloat16 ext_i0(BFloat16 a) { return BFloat16(float.ext_i0(float(a))); }
    public static BFloat16 ext_i1(BFloat16 a) { return BFloat16(float.ext_i1(float(a))); }
    public static BFloat16 ext_hypot(BFloat16 a, BFloat16 b) { return BFloat16(float.ext_hypot(float(a), float(b))); }
    public static BFloat16 ext_xlogy(BFloat16 a, BFloat16 b) { return BFloat16(float.ext_xlogy(float(a), float(b))); }
    public static BFloat16 ext_logaddexp(BFloat16 a, BFloat16 b) { return BFloat16(float.ext_logaddexp(float(a), float(b))); }
    public static BFloat16 ext_logaddexp2(BFloat16 a, BFloat16 b) { return BFloat16(float.ext_logaddexp2(float(a), float(b))); }
    public static BFloat16 ext_gelu(BFloat16 a) { return BFloat16(float.ext_gelu(float(a))); }
}

// conversion between any two dtypes
__include "cast.slang";
