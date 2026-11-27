implementing dtype;

public extension double : ITorchVulkan
{
    public static double load(uint64_t address, uint64_t index) { return ((double*)address)[index]; }
    public static void store(uint64_t address, uint64_t index, double value) { ((double*)address)[index] = value; }
    public static void storeVec<let N : int>(uint64_t address, uint64_t index, double values[N], uint count)
    {
        [ForceUnroll] for (int k = 0; k < N; k++) if (uint(k) < count) ((double*)address)[index + uint64_t(k)] = values[k];
    }

    public typealias Accum = double;
    public double toAccum() { return this; }
    public static double fromAccum(double v) { return v; }
    public static double fromAccumExact(double v) { return v; }
    public typealias PushScalar = double;
    public static double fromPushScalar(double v) { return v; }

    public typealias Wide = double;
    public double widen() { return this; }
    public static double castFromI64(int64_t v) { return double(v); }
    public static double castFromU64(uint64_t v) { return double(v); }
    public static double castFromF32(float v) { return double(v); }
    public static double castFromF64(double v) { return v; }
    public static double castFromEmulatedF64(Float64Emulated v) { return bit_cast<double>(v.bits); }
}

/*
float64 on devices without shaderFloat64: IEEE 754 binary64 kept in a uint64_t, computed in integer
arithmetic. Add, subtract, multiply, divide, sqrt, fmod and the conversions are exact and correctly
rounded (to nearest, ties to even), so they match the CPU. The ops native float64 computes in float32
(exp, log, sin, ...) compute in float32 here too.

Everything reduces to roundBinary: a value sign * sig * 2^exp2 (any nonzero 64-bit sig, plus a sticky bit
for anything below it) rounded into an IEEE format.
*/

static const uint64_t F64_SIGN = 0x8000000000000000ull;
static const uint64_t F64_FRAC = 0x000FFFFFFFFFFFFFull;
static const uint64_t F64_INF = 0x7FF0000000000000ull;
static const uint64_t F64_QNAN = 0x7FF8000000000000ull;

uint leadingZeros64(uint64_t x)
{
    uint hi = uint(x >> 32);
    if (hi != 0) return 31 - uint(firstbithigh(hi));
    uint lo = uint(x);
    if (lo != 0) return 63 - uint(firstbithigh(lo));
    return 64;
}

// sign * (sig + sticky) * 2^exp2 in an IEEE format with FRAC fraction bits and EXP exponent bits
// (52 / 11 is binary64, 23 / 8 binary32), rounded to nearest even
[noinline]
uint64_t roundBinary(bool sign, int exp2, uint64_t sig, bool sticky, uint FRAC, uint EXP)
{
    uint64_t signBit = sign ? (1ull << (FRAC + EXP)) : 0ull;
    int BIAS = (1 << (EXP - 1)) - 1;
    int maxExp = (1 << EXP) - 1;  // all ones: inf and NaN
    if (sig == 0) return signBit;

    uint lz = leadingZeros64(sig);
    sig <<= lz;
    exp2 -= int(lz);
    // now sig's top bit is the leading one: |value| = 1.xxx * 2^(exp2 + 63)
    int biased = exp2 + 63 + BIAS;
    if (biased >= maxExp) return signBit | (uint64_t(uint(maxExp)) << FRAC);

    if (biased <= 0)
    {
        // subnormal: scale to the format's smallest exponent, keeping what falls off as sticky
        uint shift = uint(1 - biased);
        if (shift >= 64) { sticky = sticky || sig != 0; sig = 0; }
        else { sticky = sticky || (sig << (64 - shift)) != 0; sig >>= shift; }
        biased = 0;
    }

    uint drop = 63 - FRAC;  // bits below the fraction's last place
    uint64_t kept = sig >> drop;
    uint64_t below = sig & ((1ull << drop) - 1);
    bool guard = ((below >> (drop - 1)) & 1) != 0;
    bool rest = sticky || (below & ((1ull << (drop - 1)) - 1)) != 0;
    if (guard && (rest || (kept & 1) != 0)) kept += 1;

    if (biased == 0) return signBit | kept;  // a subnormal (or, rounded up, the smallest normal)
    if (kept == (1ull << (FRAC + 1)))
    {
        kept >>= 1;
        biased += 1;
        if (biased >= maxExp) return signBit | (uint64_t(uint(maxExp)) << FRAC);
    }
    return signBit | (uint64_t(uint(biased)) << FRAC) | (kept & ((1ull << FRAC) - 1));
}

uint64_t roundF64(bool sign, int exp2, uint64_t sig, bool sticky) { return roundBinary(sign, exp2, sig, sticky, 52, 11); }

bool f64IsNaN(uint64_t a) { return (a & ~F64_SIGN) > F64_INF; }
bool f64IsInf(uint64_t a) { return (a & ~F64_SIGN) == F64_INF; }
bool f64IsZero(uint64_t a) { return (a & ~F64_SIGN) == 0; }
bool f64Sign(uint64_t a) { return (a & F64_SIGN) != 0; }
uint64_t f64Quiet(uint64_t a) { return a | 0x0008000000000000ull; }

// a finite nonzero |a| = sig * 2^exp2 with sig's leading one at bit 52
void f64Unpack(uint64_t a, out int exp2, out uint64_t sig)
{
    int e = int(uint(a >> 52) & 0x7FFu);
    sig = a & F64_FRAC;
    if (e == 0)
    {
        uint shift = leadingZeros64(sig) - 11;
        sig <<= shift;
        exp2 = 1 - 1075 - int(shift);
    }
    else
    {
        sig |= 1ull << 52;
        exp2 = e - 1075;
    }
}

[noinline]
uint64_t f64Add(uint64_t a, uint64_t b)
{
    if (f64IsNaN(a) || f64IsNaN(b)) return f64Quiet(f64IsNaN(a) ? a : b);
    if (f64IsInf(a) || f64IsInf(b))
    {
        if (f64IsInf(a) && f64IsInf(b) && f64Sign(a) != f64Sign(b)) return F64_QNAN;  // inf - inf
        return f64IsInf(a) ? a : b;
    }
    if (f64IsZero(a) || f64IsZero(b))
    {
        if (f64IsZero(a) && f64IsZero(b)) return (a & b) & F64_SIGN;  // -0 only from -0 + -0
        return f64IsZero(a) ? b : a;
    }

    int ea, eb;
    uint64_t sa, sb;
    f64Unpack(a, ea, sa);
    f64Unpack(b, eb, sb);
    bool signA = f64Sign(a);
    bool signB = f64Sign(b);
    // make a the larger magnitude
    if (eb > ea || (eb == ea && sb > sa))
    {
        int te = ea; ea = eb; eb = te;
        uint64_t ts = sa; sa = sb; sb = ts;
        bool tg = signA; signA = signB; signB = tg;
    }
    // 10 spare bits below each significand; what b loses to the alignment is jammed into its lowest bit
    sa <<= 10;
    sb <<= 10;
    uint shift = uint(ea - eb);
    if (shift > 0)
    {
        if (shift >= 64) sb = 1;
        else sb = (sb >> shift) | (((sb << (64 - shift)) != 0) ? 1ull : 0ull);
    }
    uint64_t sum = signA == signB ? sa + sb : sa - sb;
    if (sum == 0) return 0;  // x - x is +0
    return roundF64(signA, ea - 10, sum, false);
}

uint64_t f64Sub(uint64_t a, uint64_t b) { return f64Add(a, f64IsNaN(b) ? b : b ^ F64_SIGN); }

[noinline]
uint64_t f64Mul(uint64_t a, uint64_t b)
{
    if (f64IsNaN(a) || f64IsNaN(b)) return f64Quiet(f64IsNaN(a) ? a : b);
    bool sign = f64Sign(a) != f64Sign(b);
    uint64_t signBit = sign ? F64_SIGN : 0ull;
    if (f64IsInf(a) || f64IsInf(b))
    {
        if (f64IsZero(a) || f64IsZero(b)) return F64_QNAN;  // inf * 0
        return signBit | F64_INF;
    }
    if (f64IsZero(a) || f64IsZero(b)) return signBit;

    int ea, eb;
    uint64_t sa, sb;
    f64Unpack(a, ea, sa);
    f64Unpack(b, eb, sb);
    // the 106-bit product from 32-bit halves
    uint64_t a0 = sa & 0xFFFFFFFFull, a1 = sa >> 32;
    uint64_t b0 = sb & 0xFFFFFFFFull, b1 = sb >> 32;
    uint64_t mid = a0 * b1 + a1 * b0;  // both terms are below 2^53 * 2^21, so the sum can't overflow
    uint64_t lo = a0 * b0;
    uint64_t hi = a1 * b1 + (mid >> 32);
    uint64_t lo2 = lo + (mid << 32);
    if (lo2 < lo) hi += 1;
    // its top 64 bits, the rest as sticky
    uint64_t top = (hi << 22) | (lo2 >> 42);
    bool sticky = (lo2 & ((1ull << 42) - 1)) != 0;
    return roundF64(sign, ea + eb + 42, top, sticky);
}

[noinline]
uint64_t f64Div(uint64_t a, uint64_t b)
{
    if (f64IsNaN(a) || f64IsNaN(b)) return f64Quiet(f64IsNaN(a) ? a : b);
    bool sign = f64Sign(a) != f64Sign(b);
    uint64_t signBit = sign ? F64_SIGN : 0ull;
    if (f64IsInf(a)) return f64IsInf(b) ? F64_QNAN : (signBit | F64_INF);
    if (f64IsInf(b)) return signBit;
    if (f64IsZero(b)) return f64IsZero(a) ? F64_QNAN : (signBit | F64_INF);
    if (f64IsZero(a)) return signBit;

    int ea, eb;
    uint64_t sa, sb;
    f64Unpack(a, ea, sa);
    f64Unpack(b, eb, sb);
    // sa / sb is in (1/2, 2): 64 quotient bits from 2^0 down to 2^-63
    uint64_t rem = sa;
    uint64_t q = 0;
    for (int i = 0; i < 64; i++)
    {
        q <<= 1;
        if (rem >= sb)
        {
            rem -= sb;
            q |= 1;
        }
        rem <<= 1;
    }
    return roundF64(sign, ea - eb - 63, q, rem != 0);
}

[noinline]
uint64_t f64Sqrt(uint64_t a)
{
    if (f64IsNaN(a)) return f64Quiet(a);
    if (f64IsZero(a)) return a;  // sqrt(-0) is -0
    if (f64Sign(a)) return F64_QNAN;
    if (f64IsInf(a)) return a;

    int e;
    uint64_t s;
    f64Unpack(a, e, s);
    if ((e & 1) != 0)
    {
        s <<= 1;
        e -= 1;
    }
    // isqrt(s * 2^62): a 58-bit root, with the radicand's 116 bits in hi:lo
    uint64_t hi = s >> 2;
    uint64_t lo = s << 62;
    uint64_t root = 0;
    uint64_t rem = 0;
    for (int i = 57; i >= 0; i--)
    {
        uint pair = (2 * i >= 64) ? uint(hi >> uint(2 * i - 64)) & 3u : uint(lo >> uint(2 * i)) & 3u;
        rem = (rem << 2) | uint64_t(pair);
        uint64_t trial = (root << 2) | 1ull;
        root <<= 1;
        if (rem >= trial)
        {
            rem -= trial;
            root |= 1;
        }
    }
    return roundF64(false, (e - 62) / 2, root, rem != 0);
}

// the remainder of a / b with a's sign, exact: subtract b * 2^k for each k from the exponent difference down
[noinline]
uint64_t f64Fmod(uint64_t a, uint64_t b)
{
    if (f64IsNaN(a) || f64IsNaN(b) || f64IsInf(a) || f64IsZero(b)) return F64_QNAN;
    if (f64IsInf(b) || f64IsZero(a)) return a;

    int ea, eb;
    uint64_t sa, sb;
    f64Unpack(a, ea, sa);
    f64Unpack(b, eb, sb);
    if (ea < eb || (ea == eb && sa < sb)) return a;

    uint64_t r = sa;
    [loop] for (int k = ea - eb; k > 0; k--)
    {
        if (r >= sb) r -= sb;
        r <<= 1;
    }
    if (r >= sb) r -= sb;
    if (r == 0) return a & F64_SIGN;
    return roundF64(f64Sign(a), eb, r, false);
}

bool f64Eq(uint64_t a, uint64_t b)
{
    if (f64IsNaN(a) || f64IsNaN(b)) return false;
    return a == b || (f64IsZero(a) && f64IsZero(b));
}

bool f64Lt(uint64_t a, uint64_t b)
{
    if (f64IsNaN(a) || f64IsNaN(b) || (f64IsZero(a) && f64IsZero(b))) return false;
    bool sa = f64Sign(a), sb = f64Sign(b);
    if (sa != sb) return sa;
    return sa ? a > b : a < b;  // magnitudes order like their bits
}

bool f64Le(uint64_t a, uint64_t b) { return f64Lt(a, b) || f64Eq(a, b); }

uint64_t f64Trunc(uint64_t a)
{
    int e = int(uint(a >> 52) & 0x7FFu);
    if (e >= 1075) return a;  // already whole, inf or NaN
    if (e < 1023) return a & F64_SIGN;
    return a & ~(F64_FRAC >> uint(e - 1023));
}

uint64_t f64FromF32(float f)
{
    uint x = asuint(f);
    bool sign = (x >> 31) != 0;
    uint e = (x >> 23) & 0xFFu;
    uint m = x & 0x7FFFFFu;
    uint64_t signBit = sign ? F64_SIGN : 0ull;
    if (e == 0xFF) return signBit | F64_INF | (m != 0 ? 0x0008000000000000ull | (uint64_t(m) << 29) : 0ull);
    if (e == 0 && m == 0) return signBit;
    if (e == 0) return roundF64(sign, -149, uint64_t(m), false);
    return signBit | (uint64_t(e - 127 + 1023) << 52) | (uint64_t(m) << 29);
}

[noinline]
float f64ToF32(uint64_t a)
{
    bool sign = f64Sign(a);
    if (f64IsNaN(a)) return asfloat((sign ? 0x80000000u : 0u) | 0x7FC00000u | uint((a & F64_FRAC) >> 29));
    if (f64IsInf(a)) return asfloat(sign ? 0xFF800000u : 0x7F800000u);
    if (f64IsZero(a)) return asfloat(sign ? 0x80000000u : 0u);
    int e;
    uint64_t s;
    f64Unpack(a, e, s);
    return asfloat(uint(roundBinary(sign, e, s, false, 23, 8)));
}

uint64_t f64FromI64(int64_t v)
{
    if (v == 0) return 0;
    bool sign = v < 0;
    return roundF64(sign, 0, sign ? uint64_t(0) - uint64_t(v) : uint64_t(v), false);
}

uint64_t f64FromU64(uint64_t v) { return v == 0 ? 0ull : roundF64(false, 0, v, false); }

// toward zero; NaN is 0 and out of range saturates
int64_t f64ToI64(uint64_t a)
{
    if (f64IsNaN(a) || f64IsZero(a)) return 0;
    bool sign = f64Sign(a);
    if (f64IsInf(a)) return sign ? int64_t(F64_SIGN) : int64_t(~F64_SIGN);
    int e;
    uint64_t s;
    f64Unpack(a, e, s);
    if (e + 52 >= 63) return sign ? int64_t(F64_SIGN) : int64_t(~F64_SIGN);
    uint64_t mag = e >= 0 ? s << uint(e) : (e <= -64 ? 0ull : s >> uint(-e));
    return sign ? -int64_t(mag) : int64_t(mag);
}

// as f64ToI64 below 2^63 (so negative values wrap, as through int64), saturating at 2^64
uint64_t f64ToU64(uint64_t a)
{
    int e = int(uint(a >> 52) & 0x7FFu);
    if (f64IsNaN(a) || f64IsZero(a)) return 0;
    if (f64Sign(a) || e < 1086) return uint64_t(f64ToI64(a));
    if (e >= 1087) return 0xFFFFFFFFFFFFFFFFull;  // 2^64 and up, inf included
    return ((a & F64_FRAC) | (1ull << 52)) << 11;  // [2^63, 2^64)
}

public struct Float64Emulated : IArithmetic
{
    public uint64_t bits;

    public __init(int v) { bits = f64FromI64(int64_t(v)); }
    public __init(Float64Emulated v) { bits = v.bits; }

    public static Float64Emulated fromBits(uint64_t b) { Float64Emulated r; r.bits = b; return r; }
    public static Float64Emulated fromF32(float v) { return fromBits(f64FromF32(v)); }
    public float toF32() { return f64ToF32(bits); }
    public int64_t toI64() { return f64ToI64(bits); }

    public Float64Emulated add(Float64Emulated o) { return fromBits(f64Add(bits, o.bits)); }
    public Float64Emulated sub(Float64Emulated o) { return fromBits(f64Sub(bits, o.bits)); }
    public Float64Emulated mul(Float64Emulated o) { return fromBits(f64Mul(bits, o.bits)); }
    public Float64Emulated div(Float64Emulated o) { return fromBits(f64Div(bits, o.bits)); }
    public Float64Emulated mod(Float64Emulated o) { return fromBits(f64Fmod(bits, o.bits)); }
    public Float64Emulated neg() { return fromBits(bits ^ F64_SIGN); }
    public bool lessThan(Float64Emulated o) { return f64Lt(bits, o.bits); }
    public bool lessThanOrEquals(Float64Emulated o) { return f64Le(bits, o.bits); }
    public bool equals(Float64Emulated o) { return f64Eq(bits, o.bits); }

    public bool isNaN() { return f64IsNaN(bits); }
    public bool isInf() { return f64IsInf(bits); }
    public bool isNegative() { return f64Sign(bits); }
    public Float64Emulated absolute() { return fromBits(bits & ~F64_SIGN); }
    public Float64Emulated truncated() { return fromBits(f64Trunc(bits)); }

    public Float64Emulated floored()
    {
        Float64Emulated t = truncated();
        if (isNaN() || isInf() || t.bits == bits || !isNegative()) return t;
        return t.sub(Float64Emulated(1));
    }

    public Float64Emulated ceiled()
    {
        Float64Emulated t = truncated();
        if (isNaN() || isInf() || t.bits == bits || isNegative()) return t;
        return t.add(Float64Emulated(1));
    }

    // x - floor(x), as HLSL's frac
    public Float64Emulated fractional() { return sub(floored()); }
}

public extension Float64Emulated : ICastWide
{
    public D narrow<D : ICastable>() { return D.castFromEmulatedF64(this); }
}

public extension Float64Emulated : ITorchVulkan
{
    // a double's 8 bytes, as a uint64_t
    public static Float64Emulated load(uint64_t address, uint64_t index) { return fromBits(((uint64_t*)address)[index]); }
    public static void store(uint64_t address, uint64_t index, Float64Emulated value) { ((uint64_t*)address)[index] = value.bits; }
    public static void storeVec<let N : int>(uint64_t address, uint64_t index, Float64Emulated values[N], uint count)
    {
        [ForceUnroll] for (int k = 0; k < N; k++) if (uint(k) < count) ((uint64_t*)address)[index + uint64_t(k)] = values[k].bits;
    }

    public typealias Accum = Float64Emulated;
    public Float64Emulated toAccum() { return this; }
    public static Float64Emulated fromAccum(Float64Emulated v) { return v; }
    public static Float64Emulated fromAccumExact(Float64Emulated v) { return v; }
    // the host pushes a double's 8 bytes, which a uint64_t holds without declaring a 64-bit float
    public typealias PushScalar = uint64_t;
    public static Float64Emulated fromPushScalar(uint64_t v) { return fromBits(v); }

    public typealias Wide = Float64Emulated;
    public Float64Emulated widen() { return this; }
    public static Float64Emulated castFromI64(int64_t v) { return fromBits(f64FromI64(v)); }
    public static Float64Emulated castFromU64(uint64_t v) { return fromBits(f64FromU64(v)); }
    public static Float64Emulated castFromF32(float v) { return fromF32(v); }
    public static Float64Emulated castFromF64(double v) { return fromBits(bit_cast<uint64_t>(v)); }
    public static Float64Emulated castFromEmulatedF64(Float64Emulated v) { return v; }
}

Float64Emulated f64Const(uint64_t b) { return Float64Emulated.fromBits(b); }
Float64Emulated f64ViaF32(float v) { return Float64Emulated.fromF32(v); }

public extension Float64Emulated : IDtypeOps
{
    static Float64Emulated _nan() { return f64Const(F64_QNAN); }
    static Float64Emulated _pow_f32(Float64Emulated a, Float64Emulated b) { return f64ViaF32(pow(a.toF32(), b.toF32())); }

    public static Float64Emulated ext_pow(Float64Emulated a, Float64Emulated b)
    {
        Float64Emulated zero = Float64Emulated(0);
        Float64Emulated one = Float64Emulated(1);
        if (b == zero) return one;
        else if (b == one) return a;
        else if (b == Float64Emulated(2)) return a * a;
        else if (a == one) return one;
        else if (a.isNaN() || b.isNaN()) return _nan();
        else if (a == zero)
        {
            if (b < zero) return ext_highest();
            return zero;
        }
        else if (a > zero) return _pow_f32(a, b);
        else if (b.isInf()) return _pow_f32(a.absolute(), b);
        else if (b.fractional() != zero) return _nan();

        Float64Emulated res = _pow_f32(a.absolute(), b);
        if ((b.absolute() * f64Const(0x3FE0000000000000ull)).fractional() != zero) return res.neg();  // b is an odd integer
        return res;
    }

    public static Float64Emulated ext_max(Float64Emulated a, Float64Emulated b) { if (a.isNaN() || b.isNaN()) return a + b; return (a < b) ? b : a; }
    public static Float64Emulated ext_min(Float64Emulated a, Float64Emulated b) { if (a.isNaN() || b.isNaN()) return a + b; return (b < a) ? b : a; }
    public static Float64Emulated ext_zero() { return Float64Emulated(0); }
    public static Float64Emulated ext_one() { return Float64Emulated(1); }
    public static Float64Emulated ext_lowest() { return f64Const(F64_SIGN | F64_INF); }
    public static Float64Emulated ext_highest() { return f64Const(F64_INF); }

    public static Float64Emulated ext_neg(Float64Emulated a) { return a.neg(); }
    public static Float64Emulated ext_threshold_backward(Float64Emulated grad, Float64Emulated self, Float64Emulated threshold) { return (self > threshold) ? grad : Float64Emulated(0); }
    public static Float64Emulated ext_add(Float64Emulated a, Float64Emulated b) { return a + b; }
    public static Float64Emulated ext_sub(Float64Emulated a, Float64Emulated b) { return a - b; }
    public static Float64Emulated ext_mul(Float64Emulated a, Float64Emulated b) { return a * b; }
    public static Float64Emulated ext_div(Float64Emulated a, Float64Emulated b) { return a / b; }

    public static Float64Emulated ext_floor(Float64Emulated a) { return a.floored(); }
    public static Float64Emulated ext_ceil(Float64Emulated a) { return a.ceiled(); }
    public static Float64Emulated ext_trunc(Float64Emulated a) { return a.truncated(); }

    public static Float64Emulated ext_round(Float64Emulated a)
    {
        Float64Emulated half = f64Const(0x3FE0000000000000ull);
        Float64Emulated r = a.floored();
        Float64Emulated d = a - r;
        if (d < half) return r;
        if (d > half) return r + Float64Emulated(1);
        return ((r * half).fractional() == Float64Emulated(0)) ? r : r + Float64Emulated(1);  // round half to even
    }

    public static Float64Emulated ext_frac(Float64Emulated a) { return a - a.truncated(); }
    public static Float64Emulated ext_abs(Float64Emulated a) { return a.absolute(); }
    public static Float64Emulated ext_sign(Float64Emulated a)
    {
        if (a.isNaN()) return a;
        Float64Emulated zero = Float64Emulated(0);
        return (a > zero) ? Float64Emulated(1) : ((a < zero) ? Float64Emulated(-1) : a);
    }

    public static Float64Emulated ext_fmax(Float64Emulated a, Float64Emulated b) { if (a.isNaN()) return b; if (b.isNaN()) return a; return (a < b) ? b : a; }
    public static Float64Emulated ext_fmin(Float64Emulated a, Float64Emulated b) { if (a.isNaN()) return b; if (b.isNaN()) return a; return (b < a) ? b : a; }
    public static Float64Emulated ext_fmod(Float64Emulated a, Float64Emulated b) { return a % b; }

    public static Float64Emulated ext_remainder(Float64Emulated a, Float64Emulated b)
    {
        Float64Emulated zero = Float64Emulated(0);
        Float64Emulated r = a % b;
        if (r != zero && ((r < zero) != (b < zero))) r = r + b;
        return r;
    }

    public static Float64Emulated ext_floor_divide(Float64Emulated a, Float64Emulated b)
    {
        Float64Emulated zero = Float64Emulated(0);
        if (b == zero) return a / b;

        Float64Emulated mod = a % b;
        Float64Emulated div = (a - mod) / b;
        if (mod != zero && ((b < zero) != (mod < zero))) div = div - Float64Emulated(1);

        if (div != zero)
        {
            Float64Emulated floordiv = div.floored();
            if (div - floordiv > f64Const(0x3FE0000000000000ull)) floordiv = floordiv + Float64Emulated(1);
            return floordiv;
        }
        Float64Emulated q = a / b;
        bool negative = q < zero || (q == zero && q.isNegative());
        return negative ? f64Const(F64_SIGN) : zero;
    }

    public static bool ext_eq(Float64Emulated a, Float64Emulated b) { return a == b; }
    public static bool ext_ne(Float64Emulated a, Float64Emulated b) { return !(a == b); }
    public static bool ext_lt(Float64Emulated a, Float64Emulated b) { return a < b; }
    public static bool ext_le(Float64Emulated a, Float64Emulated b) { return a <= b; }
    public static bool ext_gt(Float64Emulated a, Float64Emulated b) { return b < a; }
    public static bool ext_ge(Float64Emulated a, Float64Emulated b) { return b <= a; }
    public static bool ext_nonzero(Float64Emulated a) { return !(a == Float64Emulated(0)); }
    public static bool ext_isnan(Float64Emulated a) { return a.isNaN(); }

    public static Float64Emulated ext_atan2(Float64Emulated a, Float64Emulated b)
    {
        Float64Emulated zero = Float64Emulated(0);
        if (a == zero && b == zero) return zero;
        return f64ViaF32(atan2(a.toF32(), b.toF32()));
    }

    public static Float64Emulated ext_exp(Float64Emulated a) { return f64ViaF32(exp(a.toF32())); }

    public static Float64Emulated ext_log(Float64Emulated a)
    {
        Float64Emulated zero = Float64Emulated(0);
        if (a < zero) return _nan();
        if (a == zero) return ext_lowest();
        return f64ViaF32(log(a.toF32()));
    }

    public static Float64Emulated ext_sqrt(Float64Emulated a) { return Float64Emulated.fromBits(f64Sqrt(a.bits)); }
    public static Float64Emulated ext_reciprocal(Float64Emulated a) { return Float64Emulated(1) / a; }
    public static Float64Emulated ext_sin(Float64Emulated a) { return f64ViaF32(sin(a.toF32())); }
    public static Float64Emulated ext_cos(Float64Emulated a) { return f64ViaF32(cos(a.toF32())); }
    public static Float64Emulated ext_tan(Float64Emulated a) { return f64ViaF32(tan(a.toF32())); }
    public static Float64Emulated ext_asin(Float64Emulated a) { return f64ViaF32(asin(a.toF32())); }
    public static Float64Emulated ext_acos(Float64Emulated a) { return f64ViaF32(acos(a.toF32())); }
    public static Float64Emulated ext_atan(Float64Emulated a) { return f64ViaF32(atan(a.toF32())); }
    public static Float64Emulated ext_sinh(Float64Emulated a) { return f64ViaF32(sinh(a.toF32())); }
    public static Float64Emulated ext_cosh(Float64Emulated a) { return f64ViaF32(cosh(a.toF32())); }
    public static Float64Emulated ext_tanh(Float64Emulated a) { return f64ViaF32(float.ext_tanh(a.toF32())); }
    public static Float64Emulated ext_exp2(Float64Emulated a) { return f64ViaF32(exp2(a.toF32())); }

    public static Float64Emulated ext_log2(Float64Emulated a)
    {
        Float64Emulated zero = Float64Emulated(0);
        if (a < zero) return _nan();
        if (a == zero) return ext_lowest();
        return f64ViaF32(log2(a.toF32()));
    }

    public static Float64Emulated ext_log10(Float64Emulated a)
    {
        Float64Emulated zero = Float64Emulated(0);
        if (a < zero) return _nan();
        if (a == zero) return ext_lowest();
        return f64ViaF32(log10(a.toF32()));
    }

    public static Float64Emulated ext_expm1(Float64Emulated a)
    {
        if (a.absolute() >= f64Const(0x3FE0000000000000ull)) return ext_exp(a) - Float64Emulated(1);
        Float64Emulated p = f64Const(0x3EC71DE3A556C734ull);  // 1/362880
        p = f64Const(0x3EFA01A01A01A01Aull) + a * p;     // 1/40320
        p = f64Const(0x3F2A01A01A01A01Aull) + a * p;     // 1/5040
        p = f64Const(0x3F56C16C16C16C17ull) + a * p;     // 1/720
        p = f64Const(0x3F81111111111111ull) + a * p;     // 1/120
        p = f64Const(0x3FA5555555555555ull) + a * p;     // 1/24
        p = f64Const(0x3FC5555555555555ull) + a * p;     // 1/6
        p = f64Const(0x3FE0000000000000ull) + a * p;     // 1/2
        p = Float64Emulated(1) + a * p;
        return a * p;
    }

    public static Float64Emulated ext_log1p(Float64Emulated a)
    {
        if (a < f64Const(0xBFD0000000000000ull) || a > f64Const(0x3FD999999999999Aull)) return ext_log(Float64Emulated(1) + a);
        Float64Emulated s = a / (Float64Emulated(2) + a);
        Float64Emulated s2 = s * s;
        Float64Emulated p = f64Const(0x3FB1111111111111ull);  // 1/15
        p = f64Const(0x3FB3B13B13B13B14ull) + s2 * p;    // 1/13
        p = f64Const(0x3FB745D1745D1746ull) + s2 * p;    // 1/11
        p = f64Const(0x3FBC71C71C71C71Cull) + s2 * p;    // 1/9
        p = f64Const(0x3FC2492492492492ull) + s2 * p;    // 1/7
        p = f64Const(0x3FC999999999999Aull) + s2 * p;    // 1/5
        p = f64Const(0x3FD5555555555555ull) + s2 * p;    // 1/3
        p = Float64Emulated(1) + s2 * p;
        return Float64Emulated(2) * s * p;
    }

    public static Float64Emulated ext_rsqrt(Float64Emulated a) { return Float64Emulated(1) / ext_sqrt(a); }
    public static Float64Emulated ext_sigmoid(Float64Emulated a) { return Float64Emulated(1) / (Float64Emulated(1) + ext_exp(a.neg())); }
    public static Float64Emulated ext_asinh(Float64Emulated a) { return f64ViaF32(asinh(a.toF32())); }
    public static Float64Emulated ext_acosh(Float64Emulated a) { return f64ViaF32(acosh(a.toF32())); }

    public static Float64Emulated ext_atanh(Float64Emulated a)
    {
        Float64Emulated m = a.absolute();
        if (m == Float64Emulated(1)) return a.isNegative() ? ext_lowest() : ext_highest();
        if (m > Float64Emulated(1)) return _nan();
        return f64ViaF32(atanh(a.toF32()));
    }

    public static Float64Emulated ext_deg2rad(Float64Emulated a) { return a * f64Const(0x3F91DF46A2529D39ull); }
    public static Float64Emulated ext_rad2deg(Float64Emulated a) { return a * f64Const(0x404CA5DC1A63C1F8ull); }

    public static Float64Emulated ext_erf(Float64Emulated a) { return f64ViaF32(float.ext_erf(a.toF32())); }
    public static Float64Emulated ext_erfc(Float64Emulated a) { return f64ViaF32(float.ext_erfc(a.toF32())); }
    public static Float64Emulated ext_erfcx(Float64Emulated a) { return f64ViaF32(float.ext_erfcx(a.toF32())); }
    public static Float64Emulated ext_log_ndtr(Float64Emulated a) { return f64ViaF32(float.ext_log_ndtr(a.toF32())); }
    public static Float64Emulated ext_erfinv(Float64Emulated a) { return f64ViaF32(float.ext_erfinv(a.toF32())); }
    public static Float64Emulated ext_sinc(Float64Emulated a) { return f64ViaF32(float.ext_sinc(a.toF32())); }
    public static Float64Emulated ext_entr(Float64Emulated a) { return f64ViaF32(float.ext_entr(a.toF32())); }
    public static Float64Emulated ext_lgamma(Float64Emulated a) { return f64ViaF32(float.ext_lgamma(a.toF32())); }
    public static Float64Emulated ext_digamma(Float64Emulated a) { return f64ViaF32(float.ext_digamma(a.toF32())); }
    public static Float64Emulated ext_i0(Float64Emulated a) { return f64ViaF32(float.ext_i0(a.toF32())); }
    public static Float64Emulated ext_i1(Float64Emulated a) { return f64ViaF32(float.ext_i1(a.toF32())); }
    public static Float64Emulated ext_i0e(Float64Emulated a) { return f64ViaF32(float.ext_i0e(a.toF32())); }
    public static Float64Emulated ext_i1e(Float64Emulated a) { return f64ViaF32(float.ext_i1e(a.toF32())); }

    public static Float64Emulated ext_hypot(Float64Emulated a, Float64Emulated b)
    {
        Float64Emulated x = a.absolute();
        Float64Emulated y = b.absolute();
        if (x.isInf() || y.isInf()) return ext_highest();
        if (x.isNaN() || y.isNaN()) return x + y;
        Float64Emulated m = (x < y) ? y : x;
        Float64Emulated n = (x < y) ? x : y;
        if (m == Float64Emulated(0)) return Float64Emulated(0);
        Float64Emulated scale = m > f64Const(0x46293E5939A08CEAull) ? f64Const(0x3DF0000000000000ull) : Float64Emulated(1);
        Float64Emulated r = (n * scale) / (m * scale);
        return m * ext_sqrt(Float64Emulated(1) + r * r);
    }

    public static Float64Emulated ext_xlogy(Float64Emulated a, Float64Emulated b) { if (a.isNaN()) return a; if (a == Float64Emulated(0)) return Float64Emulated(0); return a * ext_log(b); }

    public static Float64Emulated ext_logaddexp(Float64Emulated a, Float64Emulated b)
    {
        Float64Emulated m = ext_max(a, b);
        Float64Emulated n = ext_min(a, b);
        if (m.isInf()) return m;
        return m + ext_log1p(ext_exp(n - m));
    }

    public static Float64Emulated ext_logaddexp2(Float64Emulated a, Float64Emulated b)
    {
        Float64Emulated m = ext_max(a, b);
        Float64Emulated n = ext_min(a, b);
        if (m.isInf()) return m;
        return m + ext_log2(Float64Emulated(1) + ext_exp2(n - m));
    }

    public static Float64Emulated ext_gelu(Float64Emulated a)
    {
        return f64Const(0x3FE0000000000000ull) * a * (Float64Emulated(1) + ext_erf(a * f64Const(0x3FE6A09E667F3BCDull)));
    }

    public static Float64Emulated ext_gelu_tanh(Float64Emulated a)
    {
        Float64Emulated inner = f64Const(0x3FE9884533D43651ull) * (a + f64Const(0x3FA6E4E26D4801F7ull) * a * a * a);
        return f64Const(0x3FE0000000000000ull) * a * (Float64Emulated(1) + ext_tanh(inner));
    }

    public static Float64Emulated ext_gelu_backward(Float64Emulated grad, Float64Emulated a)
    {
        Float64Emulated half = f64Const(0x3FE0000000000000ull);
        Float64Emulated cdf = half * (Float64Emulated(1) + ext_erf(a * f64Const(0x3FE6A09E667F3BCDull)));
        Float64Emulated pdf = ext_exp(f64Const(0xBFE0000000000000ull) * a * a) * f64Const(0x3FD9884533D43651ull);
        return grad * (cdf + a * pdf);
    }

    public static Float64Emulated ext_gelu_tanh_backward(Float64Emulated grad, Float64Emulated a)
    {
        Float64Emulated half = f64Const(0x3FE0000000000000ull);
        Float64Emulated one = Float64Emulated(1);
        Float64Emulated x2 = a * a;
        Float64Emulated t = ext_tanh(f64Const(0x3FE9884533D43651ull) * (a + f64Const(0x3FA6E4E26D4801F7ull) * x2 * a));
        Float64Emulated dinner = f64Const(0x3FE9884533D43651ull) * (one + f64Const(0x3FC12BA9D1F60179ull) * x2);
        return grad * (half * (one + t) + half * a * (one - t * t) * dinner);
    }
}
