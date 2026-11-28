implementing dtype;

public extension uint64_t : ITorchVulkan
{
    public static uint64_t load(uint64_t address, uint64_t index) { return ((uint64_t*)address)[index]; }
    public static void store(uint64_t address, uint64_t index, uint64_t value) { ((uint64_t*)address)[index] = value; }
    public static void storeVec<let N : int>(uint64_t address, uint64_t index, uint64_t values[N], uint count)
    {
        [ForceUnroll] for (int k = 0; k < N; k++) if (uint(k) < count) ((uint64_t*)address)[index + uint64_t(k)] = values[k];
    }

    public typealias Accum = uint64_t;
    public uint64_t toAccum() { return this; }
    public static uint64_t fromAccum(uint64_t v) { return v; }
    public static uint64_t fromAccumExact(uint64_t v) { return v; }
    public typealias PushScalar = uint64_t;
    public static uint64_t fromPushScalar(uint64_t v) { return v; }

    public typealias Wide = uint64_t;
    public uint64_t widen() { return this; }
    public static uint64_t castFromI64(int64_t v) { return uint64_t(v); }
    public static uint64_t castFromU64(uint64_t v) { return v; }
    public static uint64_t castFromF32(float v) { return uint64_t(v); }
    public static uint64_t castFromF64(double v) { return uint64_t(v); }
}
