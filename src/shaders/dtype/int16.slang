implementing dtype;

public extension int16_t : ITorchVulkan
{
    public static int16_t load(uint64_t address, uint64_t index) { return ((int16_t*)address)[index]; }
    public static void store(uint64_t address, uint64_t index, int16_t value) { ((int16_t*)address)[index] = value; }
    public static void storeVec<let N : int>(uint64_t address, uint64_t index, int16_t values[N], uint count)
    {
        [ForceUnroll] for (int k = 0; k < N; k++) if (uint(k) < count) ((int16_t*)address)[index + uint64_t(k)] = values[k];
    }

    public typealias Accum = int16_t;
    public int16_t toAccum() { return this; }
    public static int16_t fromAccum(int16_t v) { return v; }
    public static int16_t fromAccumExact(int16_t v) { return v; }
    public typealias PushScalar = int32_t;
    public static int16_t fromPushScalar(int32_t v) { return int16_t(v); }

    public typealias Wide = int64_t;
    public int64_t widen() { return int64_t(this); }
    public static int16_t castFromI64(int64_t v) { return int16_t(v); }
    public static int16_t castFromU64(uint64_t v) { return int16_t(v); }
    public static int16_t castFromF32(float v) { return int16_t(int64_t(v)); }
    public static int16_t castFromF64(double v) { return int16_t(int64_t(v)); }
    public static int16_t castFromEmulatedF64(Float64Emulated v) { return int16_t(f64ToI64(v.bits)); }
}

// int16 without shaderInt16 and 16-bit storage: computes in int32, kept as 16-bit values in whole words
public struct Int16Codec : IIntCodec
{
    public static uint bytes() { return 2; }
    public static int decode(uint bits) { return int(bits << 16) >> 16; }
    public static uint encode(int value) { return uint(value) & 0xFFFFu; }
    public static int lowest() { return -32768; }
    public static int highest() { return 32767; }
}

public typealias Int16Emulated = EmulatedInt<Int16Codec>;
