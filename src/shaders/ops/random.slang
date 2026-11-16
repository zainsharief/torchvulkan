module random;

__exported import core.philox;
import core.indexing;

[[vk::constant_id(0)]] public const uint OP_TYPE = 0;
[[vk::constant_id(1)]] public const uint IS_CONTIG = 0;
[[vk::constant_id(2)]] public const uint NDIM = 0;
[[vk::constant_id(3)]] public const uint WG_x = 0;

public struct Metadata
{
    public uint64_t sizes;
    public uint64_t strides;
}

public struct RandomPushConstants
{
    public uint64_t tensor;
    public uint64_t metadata;
    public uint64_t numel;
    public uint64_t seed;
    public uint64_t offset;
    public uint64_t param0; // a T.Real's bits, or an integer for RANDOM
    public uint64_t param1; // a T.Real's bits, or an integer for RANDOM
    public uint64_t probs;  // contiguous T.Real values, for BERNOULLI_TENSOR
}

public interface IRandomOp<T : IRandomDtype>
{
    static T sample(inout RandomStream<T.Real> stream, RandomPushConstants pcs, uint element);
}

public struct UniformOp<T : IRandomDtype> : IRandomOp<T>
{
    public static T sample(inout RandomStream<T.Real> stream, RandomPushConstants pcs, uint element)
    {
        typealias R = T.Real;
        R from = R.fromBits(pcs.param0);
        R to = R.fromBits(pcs.param1);
        T value = castValue<R, T>(from + stream.uniform() * (to - from));
        // rounding to T can land on `to`, which [from, to) excludes
        return T.ext_eq(value, castValue<R, T>(to)) ? castValue<R, T>(from) : value;
    }
}

public struct NormalOp<T : IRandomDtype> : IRandomOp<T>
{
    public static T sample(inout RandomStream<T.Real> stream, RandomPushConstants pcs, uint element)
    {
        typealias R = T.Real;
        R mean = R.fromBits(pcs.param0);
        R std = R.fromBits(pcs.param1);
        return castValue<R, T>(mean + std * stream.normal());
    }
}

public struct ExponentialOp<T : IRandomDtype> : IRandomOp<T>
{
    public static T sample(inout RandomStream<T.Real> stream, RandomPushConstants pcs, uint element)
    {
        typealias R = T.Real;
        R lambda = R.fromBits(pcs.param0);
        R halfEpsilon = R.fromBits(pcs.param1);
        // log(u) <= -halfEpsilon keeps samples above zero once rounded to T
        R logU = R.ext_min(R.ext_log(stream.openUniform()), -halfEpsilon);
        return castValue<R, T>(-logU / lambda);
    }
}

public struct LogNormalOp<T : IRandomDtype> : IRandomOp<T>
{
    public static T sample(inout RandomStream<T.Real> stream, RandomPushConstants pcs, uint element)
    {
        typealias R = T.Real;
        R mean = R.fromBits(pcs.param0);
        R std = R.fromBits(pcs.param1);
        return castValue<R, T>(R.ext_exp(mean + std * stream.normal()));
    }
}

public struct CauchyOp<T : IRandomDtype> : IRandomOp<T>
{
    public static T sample(inout RandomStream<T.Real> stream, RandomPushConstants pcs, uint element)
    {
        typealias R = T.Real;
        R median = R.fromBits(pcs.param0);
        R sigma = R.fromBits(pcs.param1);
        return castValue<R, T>(median + sigma * R.ext_tan(R(3.141592653589793) * (stream.openUniform() - R(0.5))));
    }
}

public struct GeometricOp<T : IRandomDtype> : IRandomOp<T>
{
    public static T sample(inout RandomStream<T.Real> stream, RandomPushConstants pcs, uint element)
    {
        typealias R = T.Real;
        R p = R.fromBits(pcs.param0);
        return castValue<R, T>(R.ext_max(R.ext_ceil(R.ext_log(stream.openUniform()) / R.ext_log1p(-p)), R(1)));
    }
}

public struct BernoulliScalarOp<T : IRandomDtype> : IRandomOp<T>
{
    public static T sample(inout RandomStream<T.Real> stream, RandomPushConstants pcs, uint element)
    {
        typealias R = T.Real;
        R p = R.fromBits(pcs.param0);
        return castValue<R, T>(stream.uniform() < p ? R(1) : R(0));
    }
}

public struct BernoulliTensorOp<T : IRandomDtype> : IRandomOp<T>
{
    public static T sample(inout RandomStream<T.Real> stream, RandomPushConstants pcs, uint element)
    {
        typealias R = T.Real;
        R p = ((R*)pcs.probs)[element];
        return castValue<R, T>(stream.uniform() < p ? R(1) : R(0));
    }
}

public struct RandomIntOp<T : IRandomDtype> : IRandomOp<T>
{
    public static T sample(inout RandomStream<T.Real> stream, RandomPushConstants pcs, uint element)
    {
        uint64_t from = pcs.param0;
        uint64_t range = pcs.param1;
        if (range == 0) return castValue<int64_t, T>(int64_t(stream.nextUint64()));
        if (range >= (uint64_t(1) << 28)) return castValue<int64_t, T>(int64_t(from + stream.nextUint64() % range));
        return castValue<int64_t, T>(int64_t(from + uint64_t(stream.nextUint() % uint(range))));
    }
}

public void baseRandomKernel<T : IRandomDtype, Op : IRandomOp<T>, let VEC_SIZE : int>(uint3 gi, RandomPushConstants pcs)
{
    uint stream_id = flat_thread(gi, WG_x);
    uint first = stream_id * VEC_SIZE;
    if (first >= pcs.numel) return;

    var stream = RandomStream<T.Real>(pcs.seed, pcs.offset, stream_id);
    T* tensor = (T*)pcs.tensor;

    for (int k = 0; k < VEC_SIZE; k++)
    {
        uint element = first + k;
        if (element >= pcs.numel) return;

        T value = Op.sample(stream, pcs, element);

        if (IS_CONTIG != 0)
        {
            tensor[element] = value;
            continue;
        }

        Metadata md = *(Metadata*)pcs.metadata;
        uint* strides = (uint*)md.strides;
        uint index = 0;
        var walker = IndexWalker(md.sizes, NDIM, element);
        for (uint i = 0; i < NDIM; i++) index += walker.next(i) * strides[i];
        tensor[index] = value;
    }
}

public void dispatchRandom<T : IRandomDtype, let VEC_SIZE : int>(uint3 gi, RandomPushConstants pcs)
{
    switch (OP_TYPE)
    {
    case 0: baseRandomKernel<T, UniformOp<T>, VEC_SIZE>(gi, pcs); break;
    case 1: baseRandomKernel<T, NormalOp<T>, VEC_SIZE>(gi, pcs); break;
    case 2: baseRandomKernel<T, ExponentialOp<T>, VEC_SIZE>(gi, pcs); break;
    case 3: baseRandomKernel<T, LogNormalOp<T>, VEC_SIZE>(gi, pcs); break;
    case 4: baseRandomKernel<T, CauchyOp<T>, VEC_SIZE>(gi, pcs); break;
    case 5: baseRandomKernel<T, GeometricOp<T>, VEC_SIZE>(gi, pcs); break;
    case 6: baseRandomKernel<T, BernoulliScalarOp<T>, VEC_SIZE>(gi, pcs); break;
    case 7: baseRandomKernel<T, BernoulliTensorOp<T>, VEC_SIZE>(gi, pcs); break;
    case 8: baseRandomKernel<T, RandomIntOp<T>, VEC_SIZE>(gi, pcs); break;
    default: break;
    }
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void random_main<T : IRandomDtype, let VEC_SIZE : int>(
    uint3 gi : SV_DispatchThreadID,
    uniform RandomPushConstants pcs)
{
    dispatchRandom<T, VEC_SIZE>(gi, pcs);
}
