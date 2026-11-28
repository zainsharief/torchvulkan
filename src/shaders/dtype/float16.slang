implementing dtype;

// Conversions to float16 round to nearest even, as on the CPU and CUDA. They are done in software
// because Vulkan leaves the rounding mode of a float conversion to the driver, and AMD's rounds
// toward zero (half of all values come out one step low).

public half halfFromFloat(float f)
{
    return bit_cast<half>(uint16_t(halfBitsFromFloat(f)));
}

// float32 -> float16 bits, round to nearest even (Fabian Giesen's float_to_half_fast3_rtne)
public uint halfBitsFromFloat(float f)
{
    uint x = asuint(f);
    uint sign = (x >> 16) & 0x8000;
    uint ax = x & 0x7fffffff;
    uint h;
    if (ax >= 0x7f800000) h = ax > 0x7f800000 ? 0x7e00 : 0x7c00;  // nan, inf
    else if (ax >= 0x477ff000) h = 0x7c00;                         // rounds to inf
    else if (ax < 0x38800000)
    {
        // a float16 subnormal or zero, counted in steps of 2^-24 and rounded in integers (a float add
        // would round however the driver does); rounding up out of the subnormals gives 0x400, 2^-14
        uint e = ax >> 23;
        if (e < 102) h = 0;  // below 2^-25
        else
        {
            uint m = (ax & 0x7fffff) | 0x800000;
            uint s = 126 - e;
            h = m >> s;
            uint rest = m & ((1u << s) - 1);
            uint halfway = 1u << (s - 1);
            if (rest > halfway || (rest == halfway && (h & 1) != 0)) h += 1;
        }
    }
    else
    {
        uint odd = (ax >> 13) & 1;
        h = (ax + 0xc8000fff + odd) >> 13;  // rebias the exponent and round half to even
    }
    return sign | h;
}

// float16 sqrt, correctly rounded on any driver. Vulkan lets float32 sqrt be off by a few ulp (Intel
// Gen9's is), which flips the float16 rounding of some results. A midpoint between two neighbouring
// float16 values has at most 12 significant bits, so its square is exact in float32, and comparing
// it with x picks the correctly rounded neighbour (ties to even).
public half halfSqrt(float x)
{
    half h = halfFromFloat(sqrt(x));
    if (!(x > 0.0f) || isinf(x)) return h;  // zero, negative, inf, NaN: nothing to correct
    uint16_t b = bit_cast<uint16_t>(h);
    float lo = 0.5f * (float(bit_cast<half>(uint16_t(b - 1))) + float(h));
    float hi = 0.5f * (float(h) + float(bit_cast<half>(uint16_t(b + 1))));
    if (x < lo * lo || (x == lo * lo && (b & 1) != 0)) return bit_cast<half>(uint16_t(b - 1));
    if (x > hi * hi || (x == hi * hi && (b & 1) != 0)) return bit_cast<half>(uint16_t(b + 1));
    return h;
}

public extension half : ITorchVulkan
{
    public static half load(uint64_t address, uint64_t index) { return ((half*)address)[index]; }
    public static void store(uint64_t address, uint64_t index, half value) { ((half*)address)[index] = value; }
    public static void storeVec<let N : int>(uint64_t address, uint64_t index, half values[N], uint count)
    {
        [ForceUnroll] for (int k = 0; k < N; k++) if (uint(k) < count) ((half*)address)[index + uint64_t(k)] = values[k];
    }

    // accumulate in float32
    public typealias Accum = float;
    public float toAccum() { return float(this); }
    public static half fromAccum(float v) { return halfFromFloat(v); }
    public static half fromAccumExact(float v) { return half(v); }
    public typealias PushScalar = float;
    public static half fromPushScalar(float v) { return halfFromFloat(v); }

    public typealias Wide = float;
    public float widen() { return float(this); }
    // (every integer that does not round to inf is exact in float32)
    public static half castFromI64(int64_t v) { return halfFromFloat(float(v)); }
    public static half castFromU64(uint64_t v) { return halfFromFloat(float(v)); }
    public static half castFromF32(float v) { return halfFromFloat(v); }
    // through float32, as PyTorch's c10::Half(double) does
    public static half castFromF64(double v) { return halfFromFloat(float(v)); }
}

// float16 computes in float32 and rounds once, as the CPU and CUDA do: Vulkan lets native float16
// division, sqrt and transcendentals be a few ulp off. Results that are exact in float16 (min/max,
// rounding, fmod) take the native conversion; add, sub, mul and compares are exact natively.
public extension half : IDtypeOps
{
    public static half ext_pow(half a, half b) { return halfFromFloat(float.ext_pow(float(a), float(b))); }
    public static half ext_max(half a, half b) { return half(float.ext_max(float(a), float(b))); }
    public static half ext_min(half a, half b) { return half(float.ext_min(float(a), float(b))); }
    public static half ext_zero() { return half(0.0f); }
    public static half ext_one() { return half(1.0f); }
    public static half ext_lowest() { return half(float.ext_lowest()); }
    public static half ext_highest() { return half(float.ext_highest()); }
    public static half ext_neg(half a) { return -a; }
    public static half ext_threshold_backward(half grad, half self, half threshold) { return (self > threshold) ? grad : half(0.0f); }
    public static half ext_add(half a, half b) { return a + b; }
    public static half ext_sub(half a, half b) { return a - b; }
    public static half ext_mul(half a, half b) { return a * b; }
    public static half ext_div(half a, half b) { return halfFromFloat(float(a) / float(b)); }
    public static half ext_floor(half a) { return floor(a); }
    public static half ext_ceil(half a) { return ceil(a); }
    public static half ext_trunc(half a) { return trunc(a); }
    public static half ext_round(half a) { return half(float.ext_round(float(a))); }
    public static half ext_frac(half a) { return half(float.ext_frac(float(a))); }
    public static half ext_abs(half a) { return abs(a); }
    public static half ext_sign(half a) { return half(float.ext_sign(float(a))); }
    public static half ext_fmax(half a, half b) { return half(float.ext_fmax(float(a), float(b))); }
    public static half ext_fmin(half a, half b) { return half(float.ext_fmin(float(a), float(b))); }
    public static half ext_fmod(half a, half b) { return half(float.ext_fmod(float(a), float(b))); }
    public static half ext_remainder(half a, half b) { return halfFromFloat(float.ext_remainder(float(a), float(b))); }
    public static half ext_floor_divide(half a, half b) { return halfFromFloat(float.ext_floor_divide(float(a), float(b))); }
    public static bool ext_eq(half a, half b) { return a == b; }
    public static bool ext_ne(half a, half b) { return a != b; }
    public static bool ext_lt(half a, half b) { return a < b; }
    public static bool ext_le(half a, half b) { return a <= b; }
    public static bool ext_gt(half a, half b) { return a > b; }
    public static bool ext_ge(half a, half b) { return a >= b; }
    public static bool ext_nonzero(half a) { return (bit_cast<uint16_t>(a) & uint16_t(0x7FFF)) != 0; }
    public static bool ext_isnan(half a) { return isnan(a); }
    public static half ext_atan2(half a, half b) { return halfFromFloat(float.ext_atan2(float(a), float(b))); }
    public static half ext_exp(half a) { return halfFromFloat(float.ext_exp(float(a))); }
    public static half ext_log(half a) { return halfFromFloat(float.ext_log(float(a))); }
    public static half ext_sqrt(half a) { return halfSqrt(float(a)); }
    public static half ext_reciprocal(half a) { return halfFromFloat(float.ext_reciprocal(float(a))); }
    public static half ext_sin(half a) { return halfFromFloat(float.ext_sin(float(a))); }
    public static half ext_cos(half a) { return halfFromFloat(float.ext_cos(float(a))); }
    public static half ext_tan(half a) { return halfFromFloat(float.ext_tan(float(a))); }
    public static half ext_asin(half a) { return halfFromFloat(float.ext_asin(float(a))); }
    public static half ext_acos(half a) { return halfFromFloat(float.ext_acos(float(a))); }
    public static half ext_atan(half a) { return halfFromFloat(float.ext_atan(float(a))); }
    public static half ext_sinh(half a) { return halfFromFloat(float.ext_sinh(float(a))); }
    public static half ext_cosh(half a) { return halfFromFloat(float.ext_cosh(float(a))); }
    public static half ext_tanh(half a) { return halfFromFloat(float.ext_tanh(float(a))); }
    public static half ext_exp2(half a) { return halfFromFloat(float.ext_exp2(float(a))); }
    public static half ext_log2(half a) { return halfFromFloat(float.ext_log2(float(a))); }
    public static half ext_log10(half a) { return halfFromFloat(float.ext_log10(float(a))); }
    public static half ext_expm1(half a) { return halfFromFloat(float.ext_expm1(float(a))); }
    public static half ext_log1p(half a) { return halfFromFloat(float.ext_log1p(float(a))); }
    public static half ext_rsqrt(half a) { return halfFromFloat(float.ext_rsqrt(float(a))); }
    public static half ext_sigmoid(half a) { return halfFromFloat(float.ext_sigmoid(float(a))); }
    public static half ext_asinh(half a) { return halfFromFloat(float.ext_asinh(float(a))); }
    public static half ext_acosh(half a) { return halfFromFloat(float.ext_acosh(float(a))); }
    public static half ext_atanh(half a) { return halfFromFloat(float.ext_atanh(float(a))); }
    public static half ext_deg2rad(half a) { return halfFromFloat(float.ext_deg2rad(float(a))); }
    public static half ext_rad2deg(half a) { return halfFromFloat(float.ext_rad2deg(float(a))); }
    public static half ext_erf(half a) { return halfFromFloat(float.ext_erf(float(a))); }
    public static half ext_erfc(half a) { return halfFromFloat(float.ext_erfc(float(a))); }
    public static half ext_erfcx(half a) { return halfFromFloat(float.ext_erfcx(float(a))); }
    public static half ext_log_ndtr(half a) { return halfFromFloat(float.ext_log_ndtr(float(a))); }
    public static half ext_erfinv(half a) { return halfFromFloat(float.ext_erfinv(float(a))); }
    public static half ext_sinc(half a) { return halfFromFloat(float.ext_sinc(float(a))); }
    public static half ext_entr(half a) { return halfFromFloat(float.ext_entr(float(a))); }
    public static half ext_lgamma(half a) { return halfFromFloat(float.ext_lgamma(float(a))); }
    public static half ext_digamma(half a) { return halfFromFloat(float.ext_digamma(float(a))); }
    public static half ext_i0(half a) { return halfFromFloat(float.ext_i0(float(a))); }
    public static half ext_i1(half a) { return halfFromFloat(float.ext_i1(float(a))); }
    public static half ext_i0e(half a) { return halfFromFloat(float.ext_i0e(float(a))); }
    public static half ext_i1e(half a) { return halfFromFloat(float.ext_i1e(float(a))); }
    public static half ext_hypot(half a, half b) { return halfFromFloat(float.ext_hypot(float(a), float(b))); }
    public static half ext_xlogy(half a, half b) { return halfFromFloat(float.ext_xlogy(float(a), float(b))); }
    public static half ext_logaddexp(half a, half b) { return halfFromFloat(float.ext_logaddexp(float(a), float(b))); }
    public static half ext_logaddexp2(half a, half b) { return halfFromFloat(float.ext_logaddexp2(float(a), float(b))); }
    public static half ext_gelu(half a) { return halfFromFloat(float.ext_gelu(float(a))); }
    public static half ext_gelu_tanh(half a) { return halfFromFloat(float.ext_gelu_tanh(float(a))); }
    public static half ext_gelu_backward(half grad, half a) { return halfFromFloat(float.ext_gelu_backward(float(grad), float(a))); }
    public static half ext_gelu_tanh_backward(half grad, half a) { return halfFromFloat(float.ext_gelu_tanh_backward(float(grad), float(a))); }
}

// float16 on devices without shaderFloat16: computes in float32, kept as 16-bit codes in whole words
public struct Float16Codec : IFloatCodec
{
    public static uint bytes() { return 2; }

    public static float decode(uint bits)
    {
        uint sign = (bits & 0x8000u) << 16;
        if (((bits >> 10) & 0x1Fu) == 0x1F) return asfloat(sign | 0x7F800000u | ((bits & 0x3FFu) << 13));  // inf, NaN
        return asfloat(sign | decodeMinifloat(bits & 0x7FFFu, 10, 15));
    }

    public static uint encode(float value) { return halfBitsFromFloat(value); }

    // as halfSqrt, without the 16-bit type
    public static float sqrt(float x)
    {
        uint b = halfBitsFromFloat(float.ext_sqrt(x));
        float h = decode(b);
        if (!(x > 0.0f) || isinf(x)) return h;
        float lo = 0.5f * (decode(b - 1) + h);
        float hi = 0.5f * (h + decode(b + 1));
        if (x < lo * lo || (x == lo * lo && (b & 1) != 0)) return decode(b - 1);
        if (x > hi * hi || (x == hi * hi && (b & 1) != 0)) return decode(b + 1);
        return h;
    }
}

public typealias Float16Emulated = EmulatedFloat<Float16Codec>;
