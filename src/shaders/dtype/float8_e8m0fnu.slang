implementing dtype;

// float8_e8m0fnu: an unsigned power of two, 2^(code - 127); no zero, NaN at 0xFF. Nothing runs it natively: it computes in float32
public struct Float8E8M0FNUCodec : IFloatCodec
{
    public static uint bytes() { return 1; }

    public static float decode(uint bits)
    {
        if (bits == 0xFF) return asfloat(0x7F800001u);  // NaN
        if (bits == 0) return asfloat(0x00400000u);    // 2^-127
        return asfloat(bits << 23);
    }

    // as c10: the exponent of |x|, rounded to nearest even on the mantissa; inf and NaN are NaN
    public static uint encode(float value)
    {
        uint x = asuint(value);
        uint e = (x >> 23) & 0xFFu;
        if (e == 0xFF) return 0xFF;
        uint g = (x >> 22) & 1;
        bool rest = (x & 0x3FFFFFu) != 0;
        if (g == 1 && (rest || e > 0)) e += 1;
        return e;
    }

    public static float sqrt(float value) { return decode(encode(float.ext_sqrt(value))); }
}

public typealias Float8E8M0FNU = EmulatedFloat<Float8E8M0FNUCodec>;
