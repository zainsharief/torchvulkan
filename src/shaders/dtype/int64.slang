implementing dtype;

public extension int64_t : ITorchVulkan
{
    public static int64_t load(uint64_t address, uint64_t index) { return ((int64_t*)address)[index]; }
    public static void store(uint64_t address, uint64_t index, int64_t value) { ((int64_t*)address)[index] = value; }
    public static void storeVec<let N : int>(uint64_t address, uint64_t index, int64_t values[N], uint count)
    {
        [ForceUnroll] for (int k = 0; k < N; k++) if (uint(k) < count) ((int64_t*)address)[index + uint64_t(k)] = values[k];
    }

    public typealias Accum = int64_t;
    public int64_t toAccum() { return this; }
    public static int64_t fromAccum(int64_t v) { return v; }
    public static int64_t fromAccumExact(int64_t v) { return v; }
    public typealias PushScalar = int64_t;
    public static int64_t fromPushScalar(int64_t v) { return v; }

    public typealias Wide = int64_t;
    public int64_t widen() { return this; }
    public static int64_t castFromI64(int64_t v) { return v; }
    public static int64_t castFromU64(uint64_t v) { return int64_t(v); }
    public static int64_t castFromF32(float v) { return int64_t(v); }
    public static int64_t castFromF64(double v) { return int64_t(v); }
}
