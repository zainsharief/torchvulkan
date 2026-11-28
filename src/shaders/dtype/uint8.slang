implementing dtype;

public extension uint8_t : ITorchVulkan
{
    public static uint8_t load(uint64_t address, uint64_t index) { return ((uint8_t*)address)[index]; }
    public static void store(uint64_t address, uint64_t index, uint8_t value) { ((uint8_t*)address)[index] = value; }
    public static void storeVec<let N : int>(uint64_t address, uint64_t index, uint8_t values[N], uint count)
    {
        [ForceUnroll] for (int k = 0; k < N; k++) if (uint(k) < count) ((uint8_t*)address)[index + uint64_t(k)] = values[k];
    }

    public typealias Accum = uint8_t;
    public uint8_t toAccum() { return this; }
    public static uint8_t fromAccum(uint8_t v) { return v; }
    public static uint8_t fromAccumExact(uint8_t v) { return v; }
    public typealias PushScalar = uint32_t;
    public static uint8_t fromPushScalar(uint32_t v) { return uint8_t(v); }

    public typealias Wide = uint64_t;
    public uint64_t widen() { return uint64_t(this); }
    public static uint8_t castFromI64(int64_t v) { return uint8_t(v); }
    public static uint8_t castFromU64(uint64_t v) { return uint8_t(v); }
    public static uint8_t castFromF32(float v) { return uint8_t(int64_t(v)); }
    public static uint8_t castFromF64(double v) { return uint8_t(int64_t(v)); }
}

// uint8 without shaderInt8 and 8-bit storage: computes in int32, kept as 8-bit values in whole words
public struct UInt8Codec : IIntCodec
{
    public static uint bytes() { return 1; }
    public static int decode(uint bits) { return int(bits); }
    public static uint encode(int value) { return uint(value) & 0xFFu; }
    public static int lowest() { return 0; }
    public static int highest() { return 255; }
}

public typealias UInt8Emulated = EmulatedInt<UInt8Codec>;
