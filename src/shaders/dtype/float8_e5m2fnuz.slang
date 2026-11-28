implementing dtype;

// float8_e5m2fnuz: 5 exponent bits (bias 16), 2 mantissa bits, no infinities, no negative zero, NaN only at 0x80. Nothing runs it natively: it computes in float32
public struct Float8E5M2FNUZCodec : IFloatCodec
{
    public static uint bytes() { return 1; }

    public static float decode(uint bits)
    {
        if (bits == 0x80) return asfloat(0x7F800001u);  // the only NaN
        return asfloat(((bits & 0x80u) << 24) | decodeMinifloat(bits & 0x7Fu, 2, 16));
    }

    // as c10: from 65536 up, and NaN, is 0x80; no negative zero
    public static uint encode(float value)
    {
        uint x = asuint(value);
        uint ax = x & 0x7FFFFFFFu;
        if (ax >= 0x47800000u) return 0x80;
        uint code = encodeMinifloat(ax, 2, 16);
        if (code == 0) return 0;
        return ((x >> 24) & 0x80u) | code;
    }

    public static float sqrt(float value) { return decode(encode(float.ext_sqrt(value))); }
}

public typealias Float8E5M2FNUZ = EmulatedFloat<Float8E5M2FNUZCodec>;
