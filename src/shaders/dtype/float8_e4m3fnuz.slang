implementing dtype;

// float8_e4m3fnuz: 4 exponent bits (bias 8), 3 mantissa bits, no infinities, no negative zero, NaN only at 0x80. Nothing runs it natively: it computes in float32
public struct Float8E4M3FNUZCodec : IFloatCodec
{
    public static uint bytes() { return 1; }

    public static float decode(uint bits)
    {
        if (bits == 0x80) return asfloat(0x7F800001u);  // the only NaN
        return asfloat(((bits & 0x80u) << 24) | decodeMinifloat(bits & 0x7Fu, 3, 8));
    }

    // as c10: from 256 up, and NaN, is 0x80; no negative zero
    public static uint encode(float value)
    {
        uint x = asuint(value);
        uint ax = x & 0x7FFFFFFFu;
        if (ax >= 0x43800000u) return 0x80;
        uint code = encodeMinifloat(ax, 3, 8);
        if (code == 0) return 0;
        return ((x >> 24) & 0x80u) | code;
    }

    public static float sqrt(float value) { return decode(encode(float.ext_sqrt(value))); }
}

public typealias Float8E4M3FNUZ = EmulatedFloat<Float8E4M3FNUZCodec>;
