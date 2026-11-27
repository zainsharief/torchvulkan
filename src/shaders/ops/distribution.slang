module distribution;

__exported import core.philox;
import core.indexing;

[[vk::constant_id(0)]] public const uint OP_TYPE = 0;
[[vk::constant_id(1)]] public const uint WG_x = 0;

static const uint MAX_ITERATIONS = 256;

public struct DistributionPushConstants
{
    public uint64_t output;
    public uint64_t input0; // alpha, rate or count
    public uint64_t input1; // prob, for BINOMIAL
    public uint64_t numel;
    public uint64_t seed;
    public uint64_t offset;
    public uint64_t tiny;   // a T.Real's bits: GAMMA's smallest result
}

public interface IDistributionOp<R : __BuiltinFloatingPointType & IRandomReal>
{
    static R sample(inout RandomStream<R> stream, R a, R b, R tiny);
}

// Marsaglia and Tsang
public struct GammaOp<R : __BuiltinFloatingPointType & IRandomReal> : IDistributionOp<R>
{
    public static R sample(inout RandomStream<R> stream, R alpha, R b, R tiny)
    {
        // std::max(tiny, sample) order, so a nan alpha gives tiny
        R sample = draw(stream, alpha);
        return (tiny < sample) ? sample : tiny;
    }

    static R draw(inout RandomStream<R> stream, R alpha)
    {
        R scale = R(1);

        if (alpha < R(1))
        {
            if (alpha == R(0)) return R(0);
            scale = R.ext_pow(R(1) - stream.uniform(), R(1) / alpha);
            alpha = alpha + R(1);
        }

        R d = alpha - R(1) / R(3);
        R c = R(1) / R.ext_sqrt(R(9) * d);
        R candidate = d;

        for (uint i = 0; i < MAX_ITERATIONS; i++)
        {
            R x = stream.normal();
            R y = R(1) + c * x;
            if (y <= R(0)) continue;

            R v = y * y * y;
            R u = R(1) - stream.uniform();
            R xx = x * x;
            candidate = scale * d * v;

            if (u < R(1) - R(0.0331) * xx * xx) return candidate;
            if (R.ext_log(u) < R(0.5) * xx + d * (R(1) - v + R.ext_log(v))) return candidate;
        }
        return candidate;
    }
}

// PTRS (Hormann) for rates of 10 or more, multiplying uniforms below that
public struct PoissonOp<R : __BuiltinFloatingPointType & IRandomReal> : IDistributionOp<R>
{
    public static R sample(inout RandomStream<R> stream, R lambda, R unused, R tiny)
    {
        if (lambda >= R(10))
        {
            R slam = R.ext_sqrt(lambda);
            R loglam = R.ext_log(lambda);
            R b = R(0.931) + R(2.53) * slam;
            R a = R(-0.059) + R(0.02483) * b;
            R invalpha = R(1.1239) + R(1.1328) / (b - R(3.4));
            R vr = R(0.9277) - R(3.6224) / (b - R(2));
            R k = R.ext_floor(lambda);

            for (uint i = 0; i < MAX_ITERATIONS; i++)
            {
                R U = stream.uniform() - R(0.5);
                R V = stream.uniform();
                R us = R(0.5) - R.ext_abs(U);
                k = R.ext_floor((R(2) * a / us + b) * U + lambda + R(0.43));

                if (us >= R(0.07) && V <= vr) return k;
                if (k < R(0) || (us < R(0.013) && V > us)) continue;
                if (R.ext_log(V) + R.ext_log(invalpha) - R.ext_log(a / (us * us) + b) <= -lambda + k * loglam - R.ext_lgamma(k + R(1))) return k;
            }
            return R.ext_max(k, R(0));
        }

        if (lambda == R(0)) return R(0);

        R enlam = R.ext_exp(-lambda);
        R count = R(0);
        R product = R(1);
        for (uint i = 0; i < MAX_ITERATIONS; i++)
        {
            product = product * stream.uniform();
            if (product <= enlam) return count;
            count = count + R(1);
        }
        return count;
    }
}

// BTRS (Hormann) when count * prob is 10 or more, inversion below that
public struct BinomialOp<R : __BuiltinFloatingPointType & IRandomReal> : IDistributionOp<R>
{
    public static R sample(inout RandomStream<R> stream, R count, R prob, R tiny)
    {
        if (count <= R(0) || prob <= R(0)) return R(0);
        if (prob >= R(1)) return count;

        if (prob <= R(0.5))
        {
            if (count * prob >= R(10)) return btrs(stream, count, prob);
            return inversion(stream, count, prob);
        }

        if (prob > R(0.5))
        {
            R qprob = R(1) - prob;
            if (count * qprob >= R(10)) return count - btrs(stream, count, qprob);
            return count - inversion(stream, count, qprob);
        }

        // prob is nan
        return prob;
    }

    static R stirlingApproxTail(R k)
    {
        static const float TAIL_VALUES[10] = {
            0.0810614667953272,
            0.0413406959554092,
            0.0276779256849983,
            0.02079067210376509,
            0.0166446911898211,
            0.0138761288230707,
            0.0118967099458917,
            0.0104112652619720,
            0.00925546218271273,
            0.00833056343336287
        };
        if (k < R(10)) return R(TAIL_VALUES[castValue<R, int32_t>(k)]);

        R kp1sq = (k + R(1)) * (k + R(1));
        return (R(1.0 / 12) - (R(1.0 / 360) - R(1.0 / 1260) / kp1sq) / kp1sq) / (k + R(1));
    }

    static R inversion(inout RandomStream<R> stream, R count, R prob)
    {
        R geomSum = R(0);
        R numGeom = R(0);
        R logprob = R.ext_log1p(-prob);

        for (uint i = 0; i < MAX_ITERATIONS; i++)
        {
            geomSum = geomSum + R.ext_ceil(R.ext_log(stream.uniform()) / logprob);
            if (geomSum > count) return numGeom;
            numGeom = numGeom + R(1);
        }
        return numGeom;
    }

    static R btrs(inout RandomStream<R> stream, R count, R prob)
    {
        R stddev = R.ext_sqrt(count * prob * (R(1) - prob));

        R b = R(1.15) + R(2.53) * stddev;
        R a = R(-0.0873) + R(0.0248) * b + R(0.01) * prob;
        R c = count * prob + R(0.5);
        R v_r = R(0.92) - R(4.2) / b;
        R r = prob / (R(1) - prob);

        R alpha = (R(2.83) + R(5.1) / b) * stddev;
        R m = R.ext_floor((count + R(1)) * prob);
        R k = m;

        for (uint i = 0; i < MAX_ITERATIONS; i++)
        {
            R U = stream.uniform() - R(0.5);
            R V = stream.uniform();

            R us = R(0.5) - R.ext_abs(U);
            k = R.ext_floor((R(2) * a / us + b) * U + c);

            if (k < R(0) || k > count) continue;
            if (us >= R(0.07) && V <= v_r) return k;

            V = R.ext_log(V * alpha / (a / (us * us) + b));
            R upperbound =
                (m + R(0.5)) * R.ext_log((m + R(1)) / (r * (count - m + R(1)))) +
                (count + R(1)) * R.ext_log((count - m + R(1)) / (count - k + R(1))) +
                (k + R(0.5)) * R.ext_log(r * (count - k + R(1)) / (k + R(1))) +
                stirlingApproxTail(m) + stirlingApproxTail(count - m) -
                stirlingApproxTail(k) - stirlingApproxTail(count - k);

            if (V <= upperbound) return k;
        }
        return R.ext_min(R.ext_max(k, R(0)), count);
    }
}

public void baseDistributionKernel<T : IRandomDtype, Op : IDistributionOp<T.Real>>(uint3 gi, DistributionPushConstants pcs)
{
    typealias R = T.Real;

    uint index = flat_thread(gi, WG_x);
    if (index >= pcs.numel) return;

    var stream = RandomStream<R>(pcs.seed, pcs.offset, index);

    R a = castValue<T, R>(T.load(pcs.input0, index));
    R b = (pcs.input1 != 0) ? castValue<T, R>(T.load(pcs.input1, index)) : R(0);
    T.store(pcs.output, index, castValue<R, T>(Op.sample(stream, a, b, R.fromBits(pcs.tiny))));
}

public void dispatchDistribution<T : IRandomDtype>(uint3 gi, DistributionPushConstants pcs)
{
    switch (OP_TYPE)
    {
    case 0: baseDistributionKernel<T, GammaOp<T.Real>>(gi, pcs); break;
    case 1: baseDistributionKernel<T, PoissonOp<T.Real>>(gi, pcs); break;
    case 2: baseDistributionKernel<T, BinomialOp<T.Real>>(gi, pcs); break;
    default: break;
    }
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void distribution_main<T : IRandomDtype>(
    uint3 gi : SV_DispatchThreadID,
    uniform DistributionPushConstants pcs)
{
    dispatchDistribution<T>(gi, pcs);
}
