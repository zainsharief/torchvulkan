implementing dtype;

/*
What a dtype the device can't run natively is built from. A dtype's file gives a codec (its conversion to
and from float32 or int32) and names its type, e.g. Float16Emulated = EmulatedFloat<Float16Codec>.

A value holds its float32 or int32, already rounded to the dtype, and every op rounds its result back,
as the native types do. Elements are only ever touched as whole 32-bit words, so no 8- or 16-bit type
or capability reaches the SPIR-V: loads are atomic word loads, a word a thread writes whole is stored
outright, and a partly written word is updated with compare-and-swap, so neighbouring elements written by
other threads or dispatches survive.

Loads stay inline: SPIRV-Cross (MoltenVK) forwards a call's result past a store to a loop counter its
arguments read, so an accumulating loop that passed an index to a load function read one element ahead.
*/

public interface IFloatCodec
{
    static uint bytes();
    static float decode(uint bits);
    static uint encode(float value);
    // sqrt, correctly rounded to the dtype
    static float sqrt(float value);
}

public interface IIntCodec
{
    static uint bytes();
    static int decode(uint bits);
    static uint encode(int value);
    static int lowest();
    static int highest();
}

// ---- whole 32-bit words ----

uint loadWordBits(uint64_t address, uint64_t index, uint bytes)
{
    uint64_t offset = (address & 3ull) + index * bytes;
    Atomic<uint>* words = (Atomic<uint>*)(address & ~3ull);
    uint word = words[offset >> 2].load();
    return (word >> (uint(offset & 3ull) * 8)) & ((1u << (bytes * 8)) - 1);
}

[noinline]
void storeWordBits(Atomic<uint>* word, uint bits, uint mask)
{
    if (mask == 0xFFFFFFFFu)
    {
        word[0].store(bits);
        return;
    }
    uint old = word[0].load();
    [loop] for (;;)
    {
        uint seen = word[0].compareExchange(old, (old & ~mask) | (bits & mask));
        if (seen == old) break;
        old = seen;
    }
}

// up to N 8- or 16-bit elements starting at index, which span at most three words
void storeWordsBits<let N : int>(uint64_t address, uint64_t index, uint bits[N], uint count, uint bytes)
{
    uint64_t first = (address & 3ull) + index * bytes;
    Atomic<uint>* words = (Atomic<uint>*)(address & ~3ull) + (first >> 2);
    uint elementMask = (1u << (bytes * 8)) - 1;

    uint word[3] = { 0, 0, 0 };
    uint mask[3] = { 0, 0, 0 };
    [ForceUnroll] for (int k = 0; k < N; k++)
    {
        if (uint(k) < count)
        {
            uint byte = uint(first & 3ull) + uint(k) * bytes;
            uint shift = (byte & 3) * 8;
            word[byte >> 2] |= (bits[k] & elementMask) << shift;
            mask[byte >> 2] |= elementMask << shift;
        }
    }
    [ForceUnroll] for (int w = 0; w < 3; w++)
    {
        if (mask[w] != 0) storeWordBits(words + w, word[w], mask[w]);
    }
}

// ---- small floats: a code with M mantissa bits and exponent bias B, in integer arithmetic ----

// float32 bits of an unsigned code (no sign, no special values)
uint decodeMinifloat(uint code, uint M, int B)
{
    uint e = code >> M;
    uint m = code & ((1u << M) - 1);
    if (e == 0)
    {
        if (m == 0) return 0;
        uint top = uint(firstbithigh(m));  // normalise the subnormal
        return (uint(int(top) - int(M) + 1 - B + 127) << 23) | (((m << (M - top)) & ((1u << M) - 1)) << (23 - M));
    }
    return (uint(int(e) - B + 127) << 23) | (m << (23 - M));
}

// |x| (as float32 bits, below the format's overflow threshold) rounded to nearest even; may carry into the
// next exponent, as c10's conversions do
uint encodeMinifloat(uint ax, uint M, int B)
{
    uint minNormal = uint(127 - B + 1) << 23;
    if (ax < minNormal)
    {
        // a subnormal of the format, counted in its smallest steps and rounded in integers
        uint e = ax >> 23;
        if (e == 0) return 0;  // float32 subnormals are far below half a step
        uint m = (ax & 0x7FFFFFu) | 0x800000u;
        int s = (127 - B + 1) - int(e) + 23 - int(M);  // m * 2^-s steps
        if (s > 25) return 0;
        uint q = m >> uint(s);
        uint rest = m & ((1u << uint(s)) - 1);
        uint halfway = 1u << uint(s - 1);
        if (rest > halfway || (rest == halfway && (q & 1) != 0)) q += 1;
        return q;
    }
    uint odd = (ax >> (23 - M)) & 1;
    uint biased = ax + (uint(B - 127) << 23) + ((1u << (22 - M)) - 1) + odd;
    return biased >> (23 - M);
}

public struct EmulatedFloat<C : IFloatCodec> : ITorchVulkan
{
    public float v;

    public __init(float roundedValue) { v = roundedValue; }
    public static This rounded(float value) { return This(C.decode(C.encode(value))); }

    public static This load(uint64_t address, uint64_t index) { return This(C.decode(loadWordBits(address, index, C.bytes()))); }
    public static void store(uint64_t address, uint64_t index, This value)
    {
        This values[1] = { value };
        storeVec<1>(address, index, values, 1);
    }
    public static void storeVec<let N : int>(uint64_t address, uint64_t index, This values[N], uint count)
    {
        uint bits[N];
        [ForceUnroll] for (int k = 0; k < N; k++) bits[k] = C.encode(values[k].v);
        storeWordsBits<N>(address, index, bits, count, C.bytes());
    }

    public typealias Accum = float;
    public float toAccum() { return v; }
    public static This fromAccum(float a) { return rounded(a); }
    public static This fromAccumExact(float a) { return This(a); }
    public typealias PushScalar = float;
    public static This fromPushScalar(float a) { return rounded(a); }

    // every value converts through float32, as c10 converts these dtypes
    public typealias Wide = float;
    public float widen() { return v; }
    public static This castFromI64(int64_t a) { return rounded(float(a)); }
    public static This castFromU64(uint64_t a) { return rounded(float(a)); }
    public static This castFromF32(float a) { return rounded(a); }
    public static This castFromF64(double a) { return rounded(float(a)); }
    public static This castFromEmulatedF64(Float64Emulated a) { return rounded(a.toF32()); }

    public static This ext_pow(This a, This b) { return rounded(float.ext_pow(a.v, b.v)); }
    public static This ext_max(This a, This b) { return rounded(float.ext_max(a.v, b.v)); }
    public static This ext_min(This a, This b) { return rounded(float.ext_min(a.v, b.v)); }
    public static This ext_zero() { return This(float.ext_zero()); }
    public static This ext_one() { return This(float.ext_one()); }
    // unrounded, as a reduction's starting value: a dtype without infinities would round it to a number
    public static This ext_lowest() { return This(float.ext_lowest()); }
    public static This ext_highest() { return This(float.ext_highest()); }
    public static This ext_neg(This a) { return rounded(float.ext_neg(a.v)); }
    public static This ext_threshold_backward(This grad, This self, This threshold) { return rounded(float.ext_threshold_backward(grad.v, self.v, threshold.v)); }
    public static This ext_add(This a, This b) { return rounded(float.ext_add(a.v, b.v)); }
    public static This ext_sub(This a, This b) { return rounded(float.ext_sub(a.v, b.v)); }
    public static This ext_mul(This a, This b) { return rounded(float.ext_mul(a.v, b.v)); }
    public static This ext_div(This a, This b) { return rounded(float.ext_div(a.v, b.v)); }
    public static This ext_floor(This a) { return rounded(float.ext_floor(a.v)); }
    public static This ext_ceil(This a) { return rounded(float.ext_ceil(a.v)); }
    public static This ext_trunc(This a) { return rounded(float.ext_trunc(a.v)); }
    public static This ext_round(This a) { return rounded(float.ext_round(a.v)); }
    public static This ext_frac(This a) { return rounded(float.ext_frac(a.v)); }
    public static This ext_abs(This a) { return rounded(float.ext_abs(a.v)); }
    public static This ext_sign(This a) { return rounded(float.ext_sign(a.v)); }
    public static This ext_fmax(This a, This b) { return rounded(float.ext_fmax(a.v, b.v)); }
    public static This ext_fmin(This a, This b) { return rounded(float.ext_fmin(a.v, b.v)); }
    public static This ext_fmod(This a, This b) { return rounded(float.ext_fmod(a.v, b.v)); }
    public static This ext_remainder(This a, This b) { return rounded(float.ext_remainder(a.v, b.v)); }
    public static This ext_floor_divide(This a, This b) { return rounded(float.ext_floor_divide(a.v, b.v)); }
    public static bool ext_eq(This a, This b) { return float.ext_eq(a.v, b.v); }
    public static bool ext_ne(This a, This b) { return float.ext_ne(a.v, b.v); }
    public static bool ext_lt(This a, This b) { return float.ext_lt(a.v, b.v); }
    public static bool ext_le(This a, This b) { return float.ext_le(a.v, b.v); }
    public static bool ext_gt(This a, This b) { return float.ext_gt(a.v, b.v); }
    public static bool ext_ge(This a, This b) { return float.ext_ge(a.v, b.v); }
    // by its bits: a subnormal float32 (float8_e8m0fnu's smallest value) is zero to GPU arithmetic
    public static bool ext_nonzero(This a) { return (asuint(a.v) & 0x7FFFFFFFu) != 0; }
    public static bool ext_isnan(This a) { return float.ext_isnan(a.v); }
    public static This ext_atan2(This a, This b) { return rounded(float.ext_atan2(a.v, b.v)); }
    public static This ext_exp(This a) { return rounded(float.ext_exp(a.v)); }
    public static This ext_log(This a) { return rounded(float.ext_log(a.v)); }
    public static This ext_sqrt(This a) { return This(C.sqrt(a.v)); }
    public static This ext_reciprocal(This a) { return rounded(float.ext_reciprocal(a.v)); }
    public static This ext_sin(This a) { return rounded(float.ext_sin(a.v)); }
    public static This ext_cos(This a) { return rounded(float.ext_cos(a.v)); }
    public static This ext_tan(This a) { return rounded(float.ext_tan(a.v)); }
    public static This ext_asin(This a) { return rounded(float.ext_asin(a.v)); }
    public static This ext_acos(This a) { return rounded(float.ext_acos(a.v)); }
    public static This ext_atan(This a) { return rounded(float.ext_atan(a.v)); }
    public static This ext_sinh(This a) { return rounded(float.ext_sinh(a.v)); }
    public static This ext_cosh(This a) { return rounded(float.ext_cosh(a.v)); }
    public static This ext_tanh(This a) { return rounded(float.ext_tanh(a.v)); }
    public static This ext_exp2(This a) { return rounded(float.ext_exp2(a.v)); }
    public static This ext_log2(This a) { return rounded(float.ext_log2(a.v)); }
    public static This ext_log10(This a) { return rounded(float.ext_log10(a.v)); }
    public static This ext_expm1(This a) { return rounded(float.ext_expm1(a.v)); }
    public static This ext_log1p(This a) { return rounded(float.ext_log1p(a.v)); }
    public static This ext_rsqrt(This a) { return rounded(float.ext_rsqrt(a.v)); }
    public static This ext_sigmoid(This a) { return rounded(float.ext_sigmoid(a.v)); }
    public static This ext_asinh(This a) { return rounded(float.ext_asinh(a.v)); }
    public static This ext_acosh(This a) { return rounded(float.ext_acosh(a.v)); }
    public static This ext_atanh(This a) { return rounded(float.ext_atanh(a.v)); }
    public static This ext_deg2rad(This a) { return rounded(float.ext_deg2rad(a.v)); }
    public static This ext_rad2deg(This a) { return rounded(float.ext_rad2deg(a.v)); }
    public static This ext_erf(This a) { return rounded(float.ext_erf(a.v)); }
    public static This ext_erfc(This a) { return rounded(float.ext_erfc(a.v)); }
    public static This ext_erfcx(This a) { return rounded(float.ext_erfcx(a.v)); }
    public static This ext_log_ndtr(This a) { return rounded(float.ext_log_ndtr(a.v)); }
    public static This ext_erfinv(This a) { return rounded(float.ext_erfinv(a.v)); }
    public static This ext_sinc(This a) { return rounded(float.ext_sinc(a.v)); }
    public static This ext_entr(This a) { return rounded(float.ext_entr(a.v)); }
    public static This ext_lgamma(This a) { return rounded(float.ext_lgamma(a.v)); }
    public static This ext_digamma(This a) { return rounded(float.ext_digamma(a.v)); }
    public static This ext_i0(This a) { return rounded(float.ext_i0(a.v)); }
    public static This ext_i1(This a) { return rounded(float.ext_i1(a.v)); }
    public static This ext_i0e(This a) { return rounded(float.ext_i0e(a.v)); }
    public static This ext_i1e(This a) { return rounded(float.ext_i1e(a.v)); }
    public static This ext_hypot(This a, This b) { return rounded(float.ext_hypot(a.v, b.v)); }
    public static This ext_xlogy(This a, This b) { return rounded(float.ext_xlogy(a.v, b.v)); }
    public static This ext_logaddexp(This a, This b) { return rounded(float.ext_logaddexp(a.v, b.v)); }
    public static This ext_logaddexp2(This a, This b) { return rounded(float.ext_logaddexp2(a.v, b.v)); }
    public static This ext_gelu(This a) { return rounded(float.ext_gelu(a.v)); }
    public static This ext_gelu_tanh(This a) { return rounded(float.ext_gelu_tanh(a.v)); }
    public static This ext_gelu_backward(This grad, This self) { return rounded(float.ext_gelu_backward(grad.v, self.v)); }
    public static This ext_gelu_tanh_backward(This grad, This self) { return rounded(float.ext_gelu_tanh_backward(grad.v, self.v)); }
}

public struct EmulatedInt<C : IIntCodec> : ITorchVulkan
{
    public int v;

    public __init(int wrappedValue) { v = wrappedValue; }
    public static This wrapped(int value) { return This(C.decode(C.encode(value))); }

    public static This load(uint64_t address, uint64_t index) { return This(C.decode(loadWordBits(address, index, C.bytes()))); }
    public static void store(uint64_t address, uint64_t index, This value)
    {
        This values[1] = { value };
        storeVec<1>(address, index, values, 1);
    }
    public static void storeVec<let N : int>(uint64_t address, uint64_t index, This values[N], uint count)
    {
        uint bits[N];
        [ForceUnroll] for (int k = 0; k < N; k++) bits[k] = C.encode(values[k].v);
        storeWordsBits<N>(address, index, bits, count, C.bytes());
    }

    // computes in int32 and wraps to the dtype after each op, as C++ promotes 8- and 16-bit integers
    public typealias Accum = int32_t;
    public int32_t toAccum() { return v; }
    public static This fromAccum(int32_t a) { return wrapped(a); }
    public static This fromAccumExact(int32_t a) { return wrapped(a); }
    public typealias PushScalar = int32_t;
    public static This fromPushScalar(int32_t a) { return wrapped(a); }

    // a float converts through int64, as torch converts it to uint8
    public typealias Wide = int64_t;
    public int64_t widen() { return int64_t(v); }
    public static This castFromI64(int64_t a) { return wrapped(int(a)); }
    public static This castFromU64(uint64_t a) { return wrapped(int(a)); }
    public static This castFromF32(float a) { return wrapped(int(int64_t(a))); }
    public static This castFromF64(double a) { return wrapped(int(int64_t(a))); }
    public static This castFromEmulatedF64(Float64Emulated a) { return wrapped(int(a.toI64())); }

    public static This ext_pow(This a, This b) { return wrapped(int32_t.ext_pow(a.v, b.v)); }
    public static This ext_max(This a, This b) { return wrapped(int32_t.ext_max(a.v, b.v)); }
    public static This ext_min(This a, This b) { return wrapped(int32_t.ext_min(a.v, b.v)); }
    public static This ext_zero() { return This(int32_t.ext_zero()); }
    public static This ext_one() { return This(int32_t.ext_one()); }
    public static This ext_lowest() { return This(C.lowest()); }
    public static This ext_highest() { return This(C.highest()); }
    public static This ext_neg(This a) { return wrapped(int32_t.ext_neg(a.v)); }
    public static This ext_threshold_backward(This grad, This self, This threshold) { return wrapped(int32_t.ext_threshold_backward(grad.v, self.v, threshold.v)); }
    public static This ext_add(This a, This b) { return wrapped(int32_t.ext_add(a.v, b.v)); }
    public static This ext_sub(This a, This b) { return wrapped(int32_t.ext_sub(a.v, b.v)); }
    public static This ext_mul(This a, This b) { return wrapped(int32_t.ext_mul(a.v, b.v)); }
    public static This ext_div(This a, This b) { return wrapped(int32_t.ext_div(a.v, b.v)); }
    public static This ext_floor(This a) { return wrapped(int32_t.ext_floor(a.v)); }
    public static This ext_ceil(This a) { return wrapped(int32_t.ext_ceil(a.v)); }
    public static This ext_trunc(This a) { return wrapped(int32_t.ext_trunc(a.v)); }
    public static This ext_round(This a) { return wrapped(int32_t.ext_round(a.v)); }
    public static This ext_frac(This a) { return wrapped(int32_t.ext_frac(a.v)); }
    public static This ext_abs(This a) { return wrapped(int32_t.ext_abs(a.v)); }
    public static This ext_sign(This a) { return wrapped(int32_t.ext_sign(a.v)); }
    public static This ext_fmax(This a, This b) { return wrapped(int32_t.ext_fmax(a.v, b.v)); }
    public static This ext_fmin(This a, This b) { return wrapped(int32_t.ext_fmin(a.v, b.v)); }
    public static This ext_fmod(This a, This b) { return wrapped(int32_t.ext_fmod(a.v, b.v)); }
    public static This ext_remainder(This a, This b) { return wrapped(int32_t.ext_remainder(a.v, b.v)); }
    public static This ext_floor_divide(This a, This b) { return wrapped(int32_t.ext_floor_divide(a.v, b.v)); }
    public static bool ext_eq(This a, This b) { return int32_t.ext_eq(a.v, b.v); }
    public static bool ext_ne(This a, This b) { return int32_t.ext_ne(a.v, b.v); }
    public static bool ext_lt(This a, This b) { return int32_t.ext_lt(a.v, b.v); }
    public static bool ext_le(This a, This b) { return int32_t.ext_le(a.v, b.v); }
    public static bool ext_gt(This a, This b) { return int32_t.ext_gt(a.v, b.v); }
    public static bool ext_ge(This a, This b) { return int32_t.ext_ge(a.v, b.v); }
    public static bool ext_nonzero(This a) { return a.v != 0; }
    public static bool ext_isnan(This a) { return int32_t.ext_isnan(a.v); }
    public static This ext_atan2(This a, This b) { return wrapped(int32_t.ext_atan2(a.v, b.v)); }
    public static This ext_exp(This a) { return wrapped(int32_t.ext_exp(a.v)); }
    public static This ext_log(This a) { return wrapped(int32_t.ext_log(a.v)); }
    public static This ext_sqrt(This a) { return wrapped(int32_t.ext_sqrt(a.v)); }
    public static This ext_reciprocal(This a) { return wrapped(int32_t.ext_reciprocal(a.v)); }
    public static This ext_sin(This a) { return wrapped(int32_t.ext_sin(a.v)); }
    public static This ext_cos(This a) { return wrapped(int32_t.ext_cos(a.v)); }
    public static This ext_tan(This a) { return wrapped(int32_t.ext_tan(a.v)); }
    public static This ext_asin(This a) { return wrapped(int32_t.ext_asin(a.v)); }
    public static This ext_acos(This a) { return wrapped(int32_t.ext_acos(a.v)); }
    public static This ext_atan(This a) { return wrapped(int32_t.ext_atan(a.v)); }
    public static This ext_sinh(This a) { return wrapped(int32_t.ext_sinh(a.v)); }
    public static This ext_cosh(This a) { return wrapped(int32_t.ext_cosh(a.v)); }
    public static This ext_tanh(This a) { return wrapped(int32_t.ext_tanh(a.v)); }
    public static This ext_exp2(This a) { return wrapped(int32_t.ext_exp2(a.v)); }
    public static This ext_log2(This a) { return wrapped(int32_t.ext_log2(a.v)); }
    public static This ext_log10(This a) { return wrapped(int32_t.ext_log10(a.v)); }
    public static This ext_expm1(This a) { return wrapped(int32_t.ext_expm1(a.v)); }
    public static This ext_log1p(This a) { return wrapped(int32_t.ext_log1p(a.v)); }
    public static This ext_rsqrt(This a) { return wrapped(int32_t.ext_rsqrt(a.v)); }
    public static This ext_sigmoid(This a) { return wrapped(int32_t.ext_sigmoid(a.v)); }
    public static This ext_asinh(This a) { return wrapped(int32_t.ext_asinh(a.v)); }
    public static This ext_acosh(This a) { return wrapped(int32_t.ext_acosh(a.v)); }
    public static This ext_atanh(This a) { return wrapped(int32_t.ext_atanh(a.v)); }
    public static This ext_deg2rad(This a) { return wrapped(int32_t.ext_deg2rad(a.v)); }
    public static This ext_rad2deg(This a) { return wrapped(int32_t.ext_rad2deg(a.v)); }
    public static This ext_erf(This a) { return wrapped(int32_t.ext_erf(a.v)); }
    public static This ext_erfc(This a) { return wrapped(int32_t.ext_erfc(a.v)); }
    public static This ext_erfcx(This a) { return wrapped(int32_t.ext_erfcx(a.v)); }
    public static This ext_log_ndtr(This a) { return wrapped(int32_t.ext_log_ndtr(a.v)); }
    public static This ext_erfinv(This a) { return wrapped(int32_t.ext_erfinv(a.v)); }
    public static This ext_sinc(This a) { return wrapped(int32_t.ext_sinc(a.v)); }
    public static This ext_entr(This a) { return wrapped(int32_t.ext_entr(a.v)); }
    public static This ext_lgamma(This a) { return wrapped(int32_t.ext_lgamma(a.v)); }
    public static This ext_digamma(This a) { return wrapped(int32_t.ext_digamma(a.v)); }
    public static This ext_i0(This a) { return wrapped(int32_t.ext_i0(a.v)); }
    public static This ext_i1(This a) { return wrapped(int32_t.ext_i1(a.v)); }
    public static This ext_i0e(This a) { return wrapped(int32_t.ext_i0e(a.v)); }
    public static This ext_i1e(This a) { return wrapped(int32_t.ext_i1e(a.v)); }
    public static This ext_hypot(This a, This b) { return wrapped(int32_t.ext_hypot(a.v, b.v)); }
    public static This ext_xlogy(This a, This b) { return wrapped(int32_t.ext_xlogy(a.v, b.v)); }
    public static This ext_logaddexp(This a, This b) { return wrapped(int32_t.ext_logaddexp(a.v, b.v)); }
    public static This ext_logaddexp2(This a, This b) { return wrapped(int32_t.ext_logaddexp2(a.v, b.v)); }
    public static This ext_gelu(This a) { return wrapped(int32_t.ext_gelu(a.v)); }
    public static This ext_gelu_tanh(This a) { return wrapped(int32_t.ext_gelu_tanh(a.v)); }
    public static This ext_gelu_backward(This grad, This self) { return wrapped(int32_t.ext_gelu_backward(grad.v, self.v)); }
    public static This ext_gelu_tanh_backward(This grad, This self) { return wrapped(int32_t.ext_gelu_tanh_backward(grad.v, self.v)); }
}

// moves one element between tensors without looking at it, for kernels that only copy (copy, index_gather)
public interface IElementMover
{
    static void move(uint64_t dst, uint64_t dst_index, uint64_t src, uint64_t src_index);
}

// any width the device can address, as a typed pointer
public struct PointerMover<T> : IElementMover
{
    public static void move(uint64_t dst, uint64_t dst_index, uint64_t src, uint64_t src_index) { ((T*)dst)[dst_index] = ((T*)src)[src_index]; }
}

// through a dtype's load and store: the emulated 8- and 16-bit integers keep every bit
public struct ElementMover<T : ITorchVulkan> : IElementMover
{
    public static void move(uint64_t dst, uint64_t dst_index, uint64_t src, uint64_t src_index) { T.store(dst, dst_index, T.load(src, src_index)); }
}
