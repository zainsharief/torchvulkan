implementing dtype;

public extension double : ITorchVulkan
{
    public static double load(uint64_t address, uint64_t index) { return ((double*)address)[index]; }
    public static void store(uint64_t address, uint64_t index, double value) { ((double*)address)[index] = value; }
    public static void storeVec<let N : int>(uint64_t address, uint64_t index, double values[N], uint count)
    {
        [ForceUnroll] for (int k = 0; k < N; k++) if (uint(k) < count) ((double*)address)[index + uint64_t(k)] = values[k];
    }

    public typealias Accum = double;
    public double toAccum() { return this; }
    public static double fromAccum(double v) { return v; }
    public static double fromAccumExact(double v) { return v; }
    public typealias PushScalar = double;
    public static double fromPushScalar(double v) { return v; }

    public typealias Wide = double;
    public double widen() { return this; }
    public static double castFromI64(int64_t v) { return double(v); }
    public static double castFromU64(uint64_t v) { return double(v); }
    public static double castFromF32(float v) { return double(v); }
    public static double castFromF64(double v) { return v; }
}
