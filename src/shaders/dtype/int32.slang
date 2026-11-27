implementing dtype;

public extension int32_t : ITorchVulkan
{
    public static int32_t load(uint64_t address, uint64_t index) { return ((int32_t*)address)[index]; }
    public static void store(uint64_t address, uint64_t index, int32_t value) { ((int32_t*)address)[index] = value; }
    public static void storeVec<let N : int>(uint64_t address, uint64_t index, int32_t values[N], uint count)
    {
        [ForceUnroll] for (int k = 0; k < N; k++) if (uint(k) < count) ((int32_t*)address)[index + uint64_t(k)] = values[k];
    }

    public typealias Accum = int32_t;
    public int32_t toAccum() { return this; }
    public static int32_t fromAccum(int32_t v) { return v; }
    public static int32_t fromAccumExact(int32_t v) { return v; }
    public typealias PushScalar = int32_t;
    public static int32_t fromPushScalar(int32_t v) { return v; }

    public typealias Wide = int64_t;
    public int64_t widen() { return int64_t(this); }
    public static int32_t castFromI64(int64_t v) { return int32_t(v); }
    public static int32_t castFromU64(uint64_t v) { return int32_t(v); }
    public static int32_t castFromF32(float v) { return int32_t(v); }
    public static int32_t castFromF64(double v) { return int32_t(v); }
    public static int32_t castFromEmulatedF64(Float64Emulated v) { return int32_t(f64ToI64(v.bits)); }
}
