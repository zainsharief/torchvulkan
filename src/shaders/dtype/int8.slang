implementing dtype;

// an out-of-range float wraps through int64, as the CPU converts it (torch defines that for uint8), rather than
// saturating as GPUs do
public extension int8_t : ITorchVulkan
{
    public static int8_t load(uint64_t address, uint64_t index) { return ((int8_t*)address)[index]; }
    public static void store(uint64_t address, uint64_t index, int8_t value) { ((int8_t*)address)[index] = value; }
    public static void storeVec<let N : int>(uint64_t address, uint64_t index, int8_t values[N], uint count)
    {
        [ForceUnroll] for (int k = 0; k < N; k++) if (uint(k) < count) ((int8_t*)address)[index + uint64_t(k)] = values[k];
    }

    public typealias Accum = int8_t;
    public int8_t toAccum() { return this; }
    public static int8_t fromAccum(int8_t v) { return v; }
    public static int8_t fromAccumExact(int8_t v) { return v; }
    public typealias PushScalar = int32_t;
    public static int8_t fromPushScalar(int32_t v) { return int8_t(v); }

    public typealias Wide = int64_t;
    public int64_t widen() { return int64_t(this); }
    public static int8_t castFromI64(int64_t v) { return int8_t(v); }
    public static int8_t castFromU64(uint64_t v) { return int8_t(v); }
    public static int8_t castFromF32(float v) { return int8_t(int64_t(v)); }
    public static int8_t castFromF64(double v) { return int8_t(int64_t(v)); }
}

// int8 without shaderInt8 and 8-bit storage: computes in int32, kept as 8-bit values in whole words
public struct Int8Codec : IIntCodec
{
    public static uint bytes() { return 1; }
    public static int decode(uint bits) { return int(bits << 24) >> 24; }
    public static uint encode(int value) { return uint(value) & 0xFFu; }
    public static int lowest() { return -128; }
    public static int highest() { return 127; }
}

public typealias Int8Emulated = EmulatedInt<Int8Codec>;
