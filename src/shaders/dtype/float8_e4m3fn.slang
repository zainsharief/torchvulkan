implementing dtype;

// float8_e4m3fn: 4 exponent bits, 3 mantissa bits, no infinities, NaN only at 0x7F/0xFF. Nothing runs it natively: it computes in float32
public struct Float8E4M3FNCodec : IFloatCodec
{
    public static uint bytes() { return 1; }

    public static float decode(uint bits)
    {
        uint sign = (bits & 0x80u) << 24;
        if ((bits & 0x7Fu) == 0x7F) return asfloat(sign | 0x7FF00000u);  // NaN, as c10 decodes it
        return asfloat(sign | decodeMinifloat(bits & 0x7Fu, 3, 7));
    }

    // as c10: NaN stays NaN, anything from 480 up (inf included) saturates to 448
    public static uint encode(float value)
    {
        uint x = asuint(value);
        uint ax = x & 0x7FFFFFFFu;
        uint sign = (x >> 24) & 0x80u;
        if (ax >= 0x43F00000u) return sign | (ax > 0x7F800000u ? 0x7Fu : 0x7Eu);
        uint code = encodeMinifloat(ax, 3, 7);
        if (code == 0x7F) code = 0x7E;
        return sign | code;
    }

    public static float sqrt(float value) { return decode(encode(float.ext_sqrt(value))); }
}

public typealias Float8E4M3FN = EmulatedFloat<Float8E4M3FNCodec>;
