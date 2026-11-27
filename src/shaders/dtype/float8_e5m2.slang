implementing dtype;

// float8_e5m2: float16 with its low byte dropped (5 exponent bits, 2 mantissa bits, IEEE infinities and NaNs). Nothing runs it natively: it computes in float32
public struct Float8E5M2Codec : IFloatCodec
{
    public static uint bytes() { return 1; }

    public static float decode(uint bits)
    {
        uint sign = (bits & 0x80u) << 24;
        if (((bits >> 2) & 0x1Fu) == 0x1F) return asfloat(sign | 0x7F800000u | ((bits & 0x3u) << 21));  // inf, NaN
        return asfloat(sign | decodeMinifloat(bits & 0x7Fu, 2, 15));
    }

    // as c10: from 65536 up is inf, NaN is 0x7F
    public static uint encode(float value)
    {
        uint x = asuint(value);
        uint ax = x & 0x7FFFFFFFu;
        uint sign = (x >> 24) & 0x80u;
        if (ax >= 0x47800000u) return sign | (ax > 0x7F800000u ? 0x7Fu : 0x7Cu);
        return sign | encodeMinifloat(ax, 2, 15);
    }

    public static float sqrt(float value) { return decode(encode(float.ext_sqrt(value))); }
}

public typealias Float8E5M2 = EmulatedFloat<Float8E5M2Codec>;
