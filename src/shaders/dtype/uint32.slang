implementing dtype;

public extension uint32_t : ITorchVulkan
{
    public static uint32_t load(uint64_t address, uint64_t index) { return ((uint32_t*)address)[index]; }
    public static void store(uint64_t address, uint64_t index, uint32_t value) { ((uint32_t*)address)[index] = value; }
    public static void storeVec<let N : int>(uint64_t address, uint64_t index, uint32_t values[N], uint count)
    {
        [ForceUnroll] for (int k = 0; k < N; k++) if (uint(k) < count) ((uint32_t*)address)[index + uint64_t(k)] = values[k];
    }

    public typealias Accum = uint32_t;
    public uint32_t toAccum() { return this; }
    public static uint32_t fromAccum(uint32_t v) { return v; }
    public static uint32_t fromAccumExact(uint32_t v) { return v; }
    public typealias PushScalar = uint32_t;
    public static uint32_t fromPushScalar(uint32_t v) { return v; }

    public typealias Wide = uint64_t;
    public uint64_t widen() { return uint64_t(this); }
    public static uint32_t castFromI64(int64_t v) { return uint32_t(v); }
    public static uint32_t castFromU64(uint64_t v) { return uint32_t(v); }
    public static uint32_t castFromF32(float v) { return uint32_t(v); }
    public static uint32_t castFromF64(double v) { return uint32_t(v); }
}
