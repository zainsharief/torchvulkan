implementing dtype;

public extension BFloat16 : ITorchVulkan
{
    public static BFloat16 load(uint64_t address, uint64_t index) { return ((BFloat16*)address)[index]; }
    public static void store(uint64_t address, uint64_t index, BFloat16 value) { ((BFloat16*)address)[index] = value; }
    public static void storeVec<let N : int>(uint64_t address, uint64_t index, BFloat16 values[N], uint count)
    {
        [ForceUnroll] for (int k = 0; k < N; k++) if (uint(k) < count) ((BFloat16*)address)[index + uint64_t(k)] = values[k];
    }

    public typealias Accum = float;
    public float toAccum() { return float(this); }
    public static BFloat16 fromAccum(float v) { return BFloat16(v); }
    public static BFloat16 fromAccumExact(float v) { return BFloat16(v); }
    public typealias PushScalar = float;
    public static BFloat16 fromPushScalar(float v) { return BFloat16(v); }

    // bfloat16 converts natively only to and from float32, so everything else goes through it
    public typealias Wide = float;
    public float widen() { return float(this); }
    public static BFloat16 castFromI64(int64_t v) { return BFloat16(float(v)); }
    public static BFloat16 castFromU64(uint64_t v) { return BFloat16(float(v)); }
    public static BFloat16 castFromF32(float v) { return BFloat16(v); }
    public static BFloat16 castFromF64(double v) { return BFloat16(float(v)); }
    public static BFloat16 castFromEmulatedF64(Float64Emulated v) { return BFloat16(v.toF32()); }
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
    public static BFloat16 ext_gelu_tanh(BFloat16 a) { return BFloat16(float.ext_gelu_tanh(float(a))); }
    public static BFloat16 ext_gelu_backward(BFloat16 grad, BFloat16 a) { return BFloat16(float.ext_gelu_backward(float(grad), float(a))); }
    public static BFloat16 ext_gelu_tanh_backward(BFloat16 grad, BFloat16 a) { return BFloat16(float.ext_gelu_tanh_backward(float(grad), float(a))); }
}

// bfloat16 without VK_KHR_shader_bfloat16: computes in float32, kept as 16-bit codes in whole words
public struct BFloat16Codec : IFloatCodec
{
    public static uint bytes() { return 2; }
    public static float decode(uint bits) { return asfloat(bits << 16); }

    // as c10: round to nearest even, every NaN to 0x7FC0
    public static uint encode(float value)
    {
        uint x = asuint(value);
        if ((x & 0x7FFFFFFFu) > 0x7F800000u) return 0x7FC0;
        return (x + 0x7FFFu + ((x >> 16) & 1)) >> 16;
    }

    public static float sqrt(float value) { return decode(encode(float.ext_sqrt(value))); }
}

public typealias BFloat16Emulated = EmulatedFloat<BFloat16Codec>;
