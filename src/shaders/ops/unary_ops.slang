module unary_ops;

__exported import core.dtype;

public interface IUnaryOp<T>
{
    static T eval(T a);
}

public struct ReluOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_max(a, T.ext_zero()); } }
public struct NegOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_neg(a); } }
public struct FloorOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_floor(a); } }
public struct CeilOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_ceil(a); } }
public struct TruncOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_trunc(a); } }
public struct RoundOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_round(a); } }
public struct FracOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_frac(a); } }
public struct AbsOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_abs(a); } }
public struct SignOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_sign(a); } }

public struct ExpOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_exp(a); } }
public struct LogOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_log(a); } }
public struct SqrtOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_sqrt(a); } }
public struct ReciprocalOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_reciprocal(a); } }
public struct SinOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_sin(a); } }
public struct CosOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_cos(a); } }
public struct TanOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_tan(a); } }
public struct AsinOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_asin(a); } }
public struct AcosOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_acos(a); } }
public struct AtanOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_atan(a); } }
public struct SinhOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_sinh(a); } }
public struct CoshOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_cosh(a); } }
public struct TanhOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_tanh(a); } }
public struct Exp2Op<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_exp2(a); } }
public struct Log2Op<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_log2(a); } }
public struct Log10Op<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_log10(a); } }
public struct Expm1Op<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_expm1(a); } }
public struct Log1pOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_log1p(a); } }
public struct RsqrtOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_rsqrt(a); } }
public struct SigmoidOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_sigmoid(a); } }
public struct AsinhOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_asinh(a); } }
public struct AcoshOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_acosh(a); } }
public struct AtanhOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_atanh(a); } }
public struct Deg2radOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_deg2rad(a); } }
public struct Rad2degOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_rad2deg(a); } }
public struct ErfOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_erf(a); } }
public struct ErfinvOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_erfinv(a); } }
public struct SincOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_sinc(a); } }
public struct EntrOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_entr(a); } }
public struct LgammaOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_lgamma(a); } }
public struct DigammaOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_digamma(a); } }
public struct I0Op<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_i0(a); } }
public struct I1Op<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_i1(a); } }

public struct GeluOp<T : ITorchVulkan> : IUnaryOp<T> { public static T eval(T a) { return T.ext_gelu(a); } }
