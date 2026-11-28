implementing dtype;

public extension uint16_t : ITorchVulkan
{
    public static uint16_t load(uint64_t address, uint64_t index) { return ((uint16_t*)address)[index]; }
    public static void store(uint64_t address, uint64_t index, uint16_t value) { ((uint16_t*)address)[index] = value; }
    public static void storeVec<let N : int>(uint64_t address, uint64_t index, uint16_t values[N], uint count)
    {
        [ForceUnroll] for (int k = 0; k < N; k++) if (uint(k) < count) ((uint16_t*)address)[index + uint64_t(k)] = values[k];
    }

    public typealias Accum = uint16_t;
    public uint16_t toAccum() { return this; }
    public static uint16_t fromAccum(uint16_t v) { return v; }
    public static uint16_t fromAccumExact(uint16_t v) { return v; }
    public typealias PushScalar = uint32_t;
    public static uint16_t fromPushScalar(uint32_t v) { return uint16_t(v); }

    public typealias Wide = uint64_t;
    public uint64_t widen() { return uint64_t(this); }
    public static uint16_t castFromI64(int64_t v) { return uint16_t(v); }
    public static uint16_t castFromU64(uint64_t v) { return uint16_t(v); }
    public static uint16_t castFromF32(float v) { return uint16_t(int64_t(v)); }
    public static uint16_t castFromF64(double v) { return uint16_t(int64_t(v)); }
}

// uint16 without shaderInt16 and 16-bit storage: computes in int32, kept as 16-bit values in whole words
public struct UInt16Codec : IIntCodec
{
    public static uint bytes() { return 2; }
    public static int decode(uint bits) { return int(bits); }
    public static uint encode(int value) { return uint(value) & 0xFFFFu; }
    public static int lowest() { return 0; }
    public static int highest() { return 65535; }
}

public typealias UInt16Emulated = EmulatedInt<UInt16Codec>;
