module dtype;

/*
The dtypes kernels are written against.

A kernel takes `T : ITorchVulkan` and gets:

    T.ext_add(a, b), T.ext_exp(a), T.ext_lt(a, b), ...     every elementwise op
    castValue<T, U>(v)                                     conversion to any dtype
    T.Accum, v.toAccum(), T.fromAccum(acc)                 the type to accumulate in

The ops (IDtypeOps) are written once per family: float32 and float64, builtin
integers, float16 and bfloat16. Each dtype also has a short ITorchVulkan
conformance that sets its Accum and conversions.

Integers implement the float-only ops through float32, so a kernel needs only
one entry point. The host promotes integers to float before those ops.

SPIR-V has no 64-bit exp/log/sin/pow, so float64 computes those ops in float32.

Each dtype has a file in src/shaders/dtype/ with its conformance, load and store.
Kernels only reach memory through T.load, T.store and T.storeVec, so a dtype the
device can't run natively runs as an emulated type of its own (dtype/emulated.slang):
it computes in float32 or int32 and keeps its elements in whole 32-bit words.
float64 without shaderFloat64 is Float64Emulated (dtype/float64.slang).

To add a dtype, give it a file there with an IDtypeOps and an ITorchVulkan
conformance, or a codec for EmulatedFloat or EmulatedInt.
*/

public interface ITorchVulkan : IDtypeOps, ICastable
{
    associatedtype Accum : IArithmetic & ITorchVulkan;
    Accum toAccum();
    static This fromAccum(Accum v);
    // v already holds a value of this dtype (a load widened to Accum): converts without rounding
    static This fromAccumExact(Accum v);

    // How a scalar of this dtype travels in push constants (PushConstantBuilder::push_scalar).
    // 8- and 16-bit types widen to 32 bits: storagePushConstant8/16 are optional features, and
    // AMD's Windows driver for RDNA2 has neither, so a T-typed push constant fails to build there.
    associatedtype PushScalar;
    static This fromPushScalar(PushScalar v);

    // a tensor's elements, which kernels only reach through these, so an emulated dtype can keep its own layout
    static This load(uint64_t address, uint64_t index);
    static void store(uint64_t address, uint64_t index, This value);
    // the first `count` of `values` to index, index + 1, ...
    static void storeVec<let N : int>(uint64_t address, uint64_t index, This values[N], uint count);
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
    static This ext_gelu_tanh(This a);
    static This ext_gelu_backward(This grad, This self);
    static This ext_gelu_tanh_backward(This grad, This self);
}

public extension<T : __BuiltinFloatingPointType> T : IDtypeOps
{
    // double has no native pow, so it routes through float32
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
        if (frac(abs(b) * T(0.5)) != T(0.0)) { return -res; }  // b is an odd integer
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
        if (b == T(0)) return a / b;

        T mod = ext_fmod(a, b);
        T div = (a - mod) / b;
        if (mod != T(0) && ((b < T(0)) != (mod < T(0)))) div = div - T(1);

        if (div != T(0))
        {
            T floordiv = floor(div);
            if (div - floordiv > T(0.5)) floordiv = floordiv + T(1);
            return floordiv;
        }
        T q = a / b;
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
    // Vulkan defines tanh's precision through (e^x - e^-x) / (e^x + e^-x), so drivers may overflow to
    // inf / inf = NaN for large |x| (AMD's does from about 89). tanh rounds to exactly +-1 in float32
    // beyond |x| ~ 9, so return the sign there (NaN still propagates: the comparison is false). Near
    // zero that formula also loses relative accuracy (Intel Gen9's tanh(1e-4) is 0.1 % off), so below
    // 1/16 use the series x - x^3/3 + 2x^5/15, whose next term is under 4e-9 relative there.
    public static T ext_tanh(T a)
    {
        if (sizeof(T) == 8) return T(float.ext_tanh(a.toFloat()));
        if (abs(a) > T(15)) return a > T(0) ? T(1) : T(-1);
        if (abs(a) < T(0.0625))
        {
            T a2 = a * a;
            return a + a * a2 * (T(-1.0 / 3.0) + a2 * T(2.0 / 15.0));
        }
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
    public static T ext_asinh(T a) { if (sizeof(T) == 8) return T(asinh(a.toFloat())); return asinh(a); }
    public static T ext_acosh(T a) { if (sizeof(T) == 8) return T(acosh(a.toFloat())); return acosh(a); }
    // Vulkan leaves atanh undefined for |x| >= 1, and AMD's returns +-1.7e38 at +-1 instead of +-inf.
    // The infinity and NaN are built from their bits: Vulkan doesn't define division by zero either.
    public static T ext_atanh(T a)
    {
        if (abs(a) == T(1)) return T(asfloat(a > T(0) ? 0x7f800000u : 0xff800000u));
        if (abs(a) > T(1)) return T(asfloat(0x7fc00000u));
        if (sizeof(T) == 8) return T(atanh(a.toFloat()));
        return atanh(a);
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
        if (x0 <= 0.0f && x0 == floor(x0)) return T(x0 == 0.0f ? asfloat(0xff800000u) : asfloat(0x7fc00000u));

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
        T x = abs(a);
        T y = abs(b);
        if (isinf(x) || isinf(y)) return T(1.0) / T(0.0);
        if (isnan(x) || isnan(y)) return x + y;
        T m = max(x, y);
        if (m == T(0)) return T(0);
        // scaled by 2^-32 near the top of the range: GPUs divide via 1 / m, which flushes to zero once it's subnormal
        T scale = m > T(1.0e30) ? T(2.3283064365386963e-10) : T(1);
        T r = (min(x, y) * scale) / (m * scale);
        return m * sqrt(T(1) + r * r);
    }
    public static T ext_xlogy(T a, T b) { if (isnan(a)) return a; if (a == T(0)) return T(0); return a * ext_log(b); }
    public static T ext_logaddexp(T a, T b) { T m = max(a, b); T n = min(a, b); if (isinf(m)) return m; return m + ext_log1p(ext_exp(n - m)); }
    public static T ext_logaddexp2(T a, T b) { T m = max(a, b); T n = min(a, b); if (isinf(m)) return m; return m + ext_log2(T(1) + ext_exp2(n - m)); }

    // the exact (erf) form, as torch computes gelu with approximate='none'
    public static T ext_gelu(T a) { return T(0.5) * a * (T(1.0) + ext_erf(a * T(0.7071067811865476))); }
    public static T ext_gelu_tanh(T a) { return T(0.5) * a * (T(1.0) + ext_tanh(T(0.7978845608028654) * (a + T(0.044715) * a * a * a))); }
    public static T ext_gelu_backward(T grad, T a)
    {
        T cdf = T(0.5) * (T(1.0) + ext_erf(a * T(0.7071067811865476)));
        T pdf = ext_exp(T(-0.5) * a * a) * T(0.3989422804014327);
        return grad * (cdf + a * pdf);
    }
    public static T ext_gelu_tanh_backward(T grad, T a)
    {
        T x2 = a * a;
        T t = ext_tanh(T(0.7978845608028654) * (a + T(0.044715) * x2 * a));
        T dinner = T(0.7978845608028654) * (T(1.0) + T(3.0 * 0.044715) * x2);
        return grad * (T(0.5) * (T(1.0) + t) + T(0.5) * a * (T(1.0) - t * t) * dinner);
    }
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
    static T truncRemainder(T a, T b) { return a - (a / b) * b; }

    public static T ext_fmod(T a, T b) { if (b == T(0)) return T(0); return truncRemainder(a, b); }

    // remainder takes the sign of the divisor where fmod takes the dividend's;
    // for unsigned T neither operand is ever negative, so the fixup never fires
    public static T ext_remainder(T a, T b)
    {
        if (b == T(0)) return T(0);
        T r = truncRemainder(a, b);
        if (r != T(0) && ((r < T(0)) != (b < T(0)))) r += b;
        return r;
    }

    // truncation rounds toward zero, so a nonzero remainder with differing signs is one too high
    public static T ext_floor_divide(T a, T b)
    {
        if (b == T(0)) return T(0);
        T q = a / b;
        if ((a < T(0)) != (b < T(0)) && q * b != a) q -= T(1);
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
    public static T ext_gelu_tanh(T a) { return _from_f32(float.ext_gelu_tanh(_to_f32(a))); }
    public static T ext_gelu_backward(T grad, T a) { return _from_f32(float.ext_gelu_backward(_to_f32(grad), _to_f32(a))); }
    public static T ext_gelu_tanh_backward(T grad, T a) { return _from_f32(float.ext_gelu_tanh_backward(_to_f32(grad), _to_f32(a))); }
}

// each dtype's conformance, load and store
__include "../dtype/int64.slang";
__include "../dtype/int32.slang";
__include "../dtype/int16.slang";
__include "../dtype/int8.slang";
__include "../dtype/uint64.slang";
__include "../dtype/uint32.slang";
__include "../dtype/uint16.slang";
__include "../dtype/uint8.slang";
__include "../dtype/bool.slang";
__include "../dtype/float64.slang";
__include "../dtype/float32.slang";
__include "../dtype/float16.slang";
__include "../dtype/bfloat16.slang";
__include "../dtype/float8_e4m3fn.slang";
__include "../dtype/float8_e5m2.slang";
__include "../dtype/float8_e4m3fnuz.slang";
__include "../dtype/float8_e5m2fnuz.slang";
__include "../dtype/float8_e8m0fnu.slang";
__include "../dtype/emulated.slang";

// conversion between any two dtypes
__include "cast.slang";
