module philox;

__exported import core.dtype;

/*
Counter-based random numbers: Philox4x32-10 (Salmon et al., "Parallel Random Numbers:
As Easy as 1, 2, 3", 2011).

    var stream = RandomStream<T.Real>(pcs.seed, pcs.offset, thread);
    T.Real u = stream.uniform();   // [0, 1)
    T.Real n = stream.normal();    // N(0, 1)

The counter is (call, offset, thread), so every call in every thread gets its own
value. The host takes a new offset for each launch.

A kernel samples in its output dtype's T.Real: double for float64, float otherwise.
Transcendentals go through R.ext_log etc., which compute float64 in float32 as the
other kernels do.
*/

public struct Philox
{
    uint4 counter; // x: the call; y, z: the offset; w: the thread
    uint2 key;     // the seed

    public __init(uint64_t seed, uint64_t offset, uint thread)
    {
        key = uint2(uint(seed), uint(seed >> 32));
        counter = uint4(0, uint(offset), uint(offset >> 32), thread);
    }

    [mutating]
    public uint4 next()
    {
        uint4 c = counter;
        uint2 k = key;
        [ForceUnroll]
        for (int i = 0; i < 9; i++)
        {
            c = round(c, k);
            k += uint2(0x9E3779B9, 0xBB67AE85);
        }
        c = round(c, k);

        counter.x += 1;
        return c;
    }

    static uint4 round(uint4 c, uint2 k)
    {
        uint64_t product0 = uint64_t(0xD2511F53) * c.x;
        uint64_t product1 = uint64_t(0xCD9E8D57) * c.z;
        return uint4(
            uint(product1 >> 32) ^ c.y ^ k.x,
            uint(product1),
            uint(product0 >> 32) ^ c.w ^ k.y,
            uint(product0));
    }
}

public interface IRandomReal : ITorchVulkan
{
    static This fromBits(uint64_t bits);            // a value packed into 64 bits
    static This uniformFromBits(uint64_t bits);     // uniform in [0, 1) from 64 random bits
    static This openUniformFromBits(uint64_t bits); // uniform in (0, 1) from 64 random (log-safe) bits
}

public extension float : IRandomReal
{
    public static float fromBits(uint64_t bits) { return asfloat(uint(bits)); }
    public static float uniformFromBits(uint64_t bits) { return float(uint(bits >> 40)) * 5.9604644775390625e-8; }
    public static float openUniformFromBits(uint64_t bits) { return (float(uint(bits >> 40)) + 0.5) * 5.9604644775390625e-8; }
}

public extension double : IRandomReal
{
    public static double fromBits(uint64_t bits) { return asdouble(uint(bits), uint(bits >> 32)); }
    public static double uniformFromBits(uint64_t bits) { return double(bits >> 11) * 1.1102230246251565e-16; }
    public static double openUniformFromBits(uint64_t bits) { return (double(bits >> 11) + 0.5) * 1.1102230246251565e-16; }
}

public interface IRandomDtype : ITorchVulkan
{
    associatedtype Real : __BuiltinFloatingPointType & IRandomReal;
}

public extension int64_t : IRandomDtype { public typealias Real = float; }
public extension int32_t : IRandomDtype { public typealias Real = float; }
public extension int16_t : IRandomDtype { public typealias Real = float; }
public extension int8_t : IRandomDtype { public typealias Real = float; }
public extension uint64_t : IRandomDtype { public typealias Real = float; }
public extension uint32_t : IRandomDtype { public typealias Real = float; }
public extension uint16_t : IRandomDtype { public typealias Real = float; }
public extension uint8_t : IRandomDtype { public typealias Real = float; }
public extension double : IRandomDtype { public typealias Real = double; }
public extension float : IRandomDtype { public typealias Real = float; }
public extension half : IRandomDtype { public typealias Real = float; }
public extension BFloat16 : IRandomDtype { public typealias Real = float; }
// the emulated dtypes sample in float32
public extension Float64Emulated : IRandomDtype { public typealias Real = float; }
public extension<C : IFloatCodec> EmulatedFloat<C> : IRandomDtype { public typealias Real = float; }
public extension<C : IIntCodec> EmulatedInt<C> : IRandomDtype { public typealias Real = float; }

public struct RandomStream<R : __BuiltinFloatingPointType & IRandomReal>
{
    Philox philox;
    uint4 bits;
    uint used;
    bool hasSpare;
    R spare;

    public __init(uint64_t seed, uint64_t offset, uint thread)
    {
        philox = Philox(seed, offset, thread);
        bits = uint4(0, 0, 0, 0);
        used = 4;
        hasSpare = false;
        spare = R(0);
    }

    [mutating]
    public uint nextUint()
    {
        if (used == 4)
        {
            bits = philox.next();
            used = 0;
        }
        uint value = bits[used];
        used++;
        return value;
    }

    [mutating]
    public uint64_t nextUint64()
    {
        uint64_t high = nextUint();
        uint64_t low = nextUint();
        return (high << 32) | low;
    }

    [mutating]
    public R uniform() 
    {
        return R.uniformFromBits(nextUint64());
    }

    [mutating]
    public R openUniform() 
    {
        return R.openUniformFromBits(nextUint64());
    }

    [mutating]
    public R normal()
    {
        if (hasSpare)
        {
            hasSpare = false;
            return spare;
        }

        R radius = R.ext_sqrt(R(-2.0) * R.ext_log(openUniform()));
        R angle = R(6.283185307179586) * uniform();
        spare = radius * R.ext_sin(angle);
        hasSpare = true;
        return radius * R.ext_cos(angle);
    }
}
