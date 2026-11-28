implementing dtype;

/*
Conversion between any two dtypes:

    Dst d = castValue<Src, Dst>(s);

Each dtype widens to the widest type of its family, and every dtype can be
built from each of the four wide types:

    int8..int64                -> int64_t
    uint8..uint64              -> uint64_t
    float16, bfloat16, float32 -> float
    float64                    -> double

The path is picked by type, so a cast not involving float64 never emits Float64. 
Each dtype's conversions are in its ITorchVulkan conformance in dtype.slang.
*/

public interface ICastable
{
    associatedtype Wide : ICastWide;

    Wide widen();

    static This castFromI64(int64_t v);
    static This castFromU64(uint64_t v);
    static This castFromF32(float v);
    static This castFromF64(double v);
}

public interface ICastWide
{
    D narrow<D : ICastable>();
}

public extension int64_t : ICastWide
{
    public D narrow<D : ICastable>() { return D.castFromI64(this); }
}

public extension uint64_t : ICastWide
{
    public D narrow<D : ICastable>() { return D.castFromU64(this); }
}

public extension float : ICastWide
{
    public D narrow<D : ICastable>() { return D.castFromF32(this); }
}

public extension double : ICastWide
{
    public D narrow<D : ICastable>() { return D.castFromF64(this); }
}

public Dst castValue<Src : ICastable, Dst : ICastable>(Src v)
{
    return v.widen().narrow<Dst>();
}
