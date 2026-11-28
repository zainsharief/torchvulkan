implementing dtype;

public extension float : ITorchVulkan
{
    public static float load(uint64_t address, uint64_t index) { return ((float*)address)[index]; }
    public static void store(uint64_t address, uint64_t index, float value) { ((float*)address)[index] = value; }
    public static void storeVec<let N : int>(uint64_t address, uint64_t index, float values[N], uint count)
    {
        [ForceUnroll] for (int k = 0; k < N; k++) if (uint(k) < count) ((float*)address)[index + uint64_t(k)] = values[k];
    }

    public typealias Accum = float;
    public float toAccum() { return this; }
    public static float fromAccum(float v) { return v; }
    public static float fromAccumExact(float v) { return v; }
    public typealias PushScalar = float;
    public static float fromPushScalar(float v) { return v; }

    public typealias Wide = float;
    public float widen() { return this; }
    public static float castFromI64(int64_t v) { return float(v); }
    public static float castFromU64(uint64_t v) { return float(v); }
    public static float castFromF32(float v) { return v; }
    public static float castFromF64(double v) { return float(v); }
}
