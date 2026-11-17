module complex_arc;

__exported import core.dtype;
import core.indexing;

[[vk::constant_id(0)]] public const uint OP = 0;
[[vk::constant_id(1)]] public const uint WG_x = 0;

public struct ComplexArcPushConstants
{
    public uint64_t input;
    public uint64_t output;
    public uint64_t numel;
}

static const uint ASINH = 0;
static const uint ASIN = 1;
static const uint ACOS = 2;
static const uint ACOSH = 3;
static const uint ATANH = 4;
static const uint ATAN = 5;

static const float FLT_EPSILON = 1.1920928955078125e-7f;
static const float A_CROSSOVER = 10.0f;
static const float B_CROSSOVER = 0.6417f;
static const float FOUR_SQRT_MIN = 4.336808689942018e-19f;    // 2^-61
static const float QUARTER_SQRT_MAX = 2.305843009213694e18f;  // 2^61
static const float SQRT_MIN = 1.0842021724855044e-19f;        // 2^-63
static const float M_E = 2.7182818285f;
static const float M_LN2 = 6.9314718056e-1f;
static const float PIO2_HI = 1.5707962513f;
static const float PIO2_LO = 7.5497899549e-8f;
static const float RECIP_EPSILON = 8388608.0f;
static const float SQRT_3_EPSILON = 5.9801995673e-4f;
static const float SQRT_6_EPSILON = 8.4572793338e-4f;
static const float FLT_MAX = 3.4028234663852886e38f;

// NaN and infinity tested on the bits, which fast math keeps
bool isNan(float x) { return (asuint(x) & 0x7fffffffu) > 0x7f800000u; }
bool isInf(float x) { return (asuint(x) & 0x7fffffffu) == 0x7f800000u; }
bool signBit(float x) { return (asuint(x) >> 31) != 0; }
float copySign(float a, float b) { return asfloat((asuint(a) & 0x7fffffffu) | (asuint(b) & 0x80000000u)); }
float quietNan() { return asfloat(0x7fc00000u); }
float negInf() { return asfloat(0xff800000u); }

float halfTerm(float a, float b, float hypot_a_b)
{
    if (b < 0.0f) return (hypot_a_b - b) / 2.0f;
    if (b == 0.0f) return a / 2.0f;
    return a * a / (hypot_a_b + b) / 2.0f;
}

struct HardWork
{
    float rx;
    bool B_is_usable;
    float B;
    float sqrt_A2my2;
    float new_y;
}

HardWork doHardWork(float x, float y)
{
    HardWork w;
    float R = float.ext_hypot(x, y + 1.0f);
    float S = float.ext_hypot(x, y - 1.0f);
    float A = max((R + S) / 2.0f, 1.0f);

    if (A < A_CROSSOVER) {
        if (y == 1.0f && x < FLT_EPSILON * FLT_EPSILON / 128.0f) {
            w.rx = sqrt(x);
        } else if (x >= FLT_EPSILON * abs(y - 1.0f)) {
            float Am1 = halfTerm(x, 1.0f + y, R) + halfTerm(x, 1.0f - y, S);
            w.rx = float.ext_log1p(Am1 + sqrt(Am1 * (A + 1.0f)));
        } else if (y < 1.0f) {
            w.rx = x / sqrt((1.0f - y) * (1.0f + y));
        } else {
            w.rx = float.ext_log1p((y - 1.0f) + sqrt((y - 1.0f) * (y + 1.0f)));
        }
    } else {
        w.rx = float.ext_log(A + sqrt(A * A - 1.0f));
    }

    w.new_y = y;
    w.B = 0.0f;
    if (y < FOUR_SQRT_MIN) {
        w.B_is_usable = false;
        w.sqrt_A2my2 = A * (2.0f / FLT_EPSILON);
        w.new_y = y * (2.0f / FLT_EPSILON);
        return w;
    }

    w.B = y / A;
    w.B_is_usable = true;
    w.sqrt_A2my2 = 0.0f;
    if (w.B > B_CROSSOVER) {
        w.B_is_usable = false;
        if (y == 1.0f && x < FLT_EPSILON / 128.0f) {
            w.sqrt_A2my2 = sqrt(x) * sqrt((A + y) / 2.0f);
        } else if (x >= FLT_EPSILON * abs(y - 1.0f)) {
            float Amy = halfTerm(x, y + 1.0f, R) + halfTerm(x, y - 1.0f, S);
            w.sqrt_A2my2 = sqrt(Amy * (A + y));
        } else if (y > 1.0f) {
            float scale = 4.0f / FLT_EPSILON / FLT_EPSILON;
            w.sqrt_A2my2 = x * scale * y / sqrt((y + 1.0f) * (y - 1.0f));
            w.new_y = y * scale;
        } else {
            w.sqrt_A2my2 = sqrt((1.0f - y) * (1.0f + y));
        }
    }
    return w;
}

float2 clogForLargeValues(float x, float y)
{
    float ax = max(abs(x), abs(y));
    float ay = min(abs(x), abs(y));
    float angle = float.ext_atan2(y, x);
    if (ax > FLT_MAX / 2.0f) return float2(float.ext_log(float.ext_hypot(x / M_E, y / M_E)) + 1.0f, angle);
    if (ax > QUARTER_SQRT_MAX || ay < SQRT_MIN) return float2(float.ext_log(float.ext_hypot(x, y)), angle);
    return float2(float.ext_log(ax * ax + ay * ay) / 2.0f, angle);
}

float2 casinh(float x, float y)
{
    float ax = abs(x);
    float ay = abs(y);
    if (isNan(x) || isNan(y)) {
        if (isInf(x)) return float2(x, quietNan());
        if (isInf(y)) return float2(y, quietNan());
        if (y == 0.0f) return float2(quietNan(), y);
        return float2(quietNan(), quietNan());
    }

    if (ax > RECIP_EPSILON || ay > RECIP_EPSILON) {
        float2 w = signBit(x) ? clogForLargeValues(-x, -y) : clogForLargeValues(x, y);
        return float2(copySign(w.x + M_LN2, x), copySign(w.y, y));
    }
    if (x == 0.0f && y == 0.0f) return float2(x, y);
    if (ax < SQRT_6_EPSILON / 4.0f && ay < SQRT_6_EPSILON / 4.0f) return float2(x, y);

    HardWork w = doHardWork(ax, ay);
    float ry = w.B_is_usable ? float.ext_asin(w.B) : float.ext_atan2(w.new_y, w.sqrt_A2my2);
    return float2(copySign(w.rx, x), copySign(ry, y));
}

float2 cacos(float x, float y)
{
    bool sx = signBit(x);
    bool sy = signBit(y);
    float ax = abs(x);
    float ay = abs(y);
    if (isNan(x) || isNan(y)) {
        if (isInf(x)) return float2(quietNan(), negInf());
        if (isInf(y)) return float2(quietNan(), -y);
        if (x == 0.0f) return float2(PIO2_HI + PIO2_LO, quietNan());
        return float2(quietNan(), quietNan());
    }

    if (ax > RECIP_EPSILON || ay > RECIP_EPSILON) {
        float2 w = clogForLargeValues(x, y);
        float rx = abs(w.y);
        float ry = w.x + M_LN2;
        return float2(rx, sy ? ry : -ry);
    }
    if (x == 1.0f && y == 0.0f) return float2(0.0f, -y);
    if (ax < SQRT_6_EPSILON / 4.0f && ay < SQRT_6_EPSILON / 4.0f) return float2(PIO2_HI - (x - PIO2_LO), -y);

    HardWork w = doHardWork(ay, ax);
    float rx;
    if (w.B_is_usable) {
        rx = float.ext_acos(sx ? -w.B : w.B);
    } else {
        rx = float.ext_atan2(w.sqrt_A2my2, sx ? -w.new_y : w.new_y);
    }
    return float2(rx, sy ? w.rx : -w.rx);
}

float2 cacosh(float x, float y)
{
    float2 w = cacos(x, y);
    if (isNan(w.x) && isNan(w.y)) return float2(w.y, w.x);
    if (isNan(w.x)) return float2(abs(w.y), w.x);
    if (isNan(w.y)) return float2(w.y, w.y);
    return float2(abs(w.y), copySign(w.x, y));
}

float sumSquares(float x, float y)
{
    if (y < SQRT_MIN) return x * x;
    return x * x + y * y;
}

float realPartReciprocal(float x, float y)
{
    int ix = int(asuint(x) & 0x7f800000u);
    int iy = int(asuint(y) & 0x7f800000u);
    const int CUTOFF = 13;  // FLT_MANT_DIG / 2 + 1
    if (ix - iy >= (CUTOFF << 23) || isInf(x)) return float.ext_div(1.0f, x);
    if (iy - ix >= (CUTOFF << 23)) return float.ext_div(float.ext_div(x, y), y);
    if (ix <= ((127 + 64 - CUTOFF) << 23)) return x / (x * x + y * y);
    float scale = asfloat(uint(0x7f800000 - ix));
    float sx = x * scale;
    float sy = y * scale;
    return sx / (sx * sx + sy * sy) * scale;
}

float2 catanh(float x, float y)
{
    float ax = abs(x);
    float ay = abs(y);
    if (isNan(x) || isNan(y)) {
        if (isInf(x)) return float2(copySign(0.0f, x), quietNan());
        if (isInf(y)) return float2(copySign(0.0f, x), copySign(PIO2_HI + PIO2_LO, y));
        if (x == 0.0f) return float2(x, quietNan());
        return float2(quietNan(), quietNan());
    }
    if (y == 0.0f && ax <= 1.0f) return float2(float.ext_atanh(x), y);
    if (x == 0.0f) return float2(x, float.ext_atan(y));

    if (ax > RECIP_EPSILON || ay > RECIP_EPSILON) return float2(realPartReciprocal(x, y), copySign(PIO2_HI + PIO2_LO, y));
    if (ax < SQRT_3_EPSILON / 2.0f && ay < SQRT_3_EPSILON / 2.0f) return float2(x, y);

    float rx;
    if (ax == 1.0f && ay < FLT_EPSILON) {
        rx = (M_LN2 - float.ext_log(ay)) / 2.0f;
    } else {
        rx = float.ext_log1p(4.0f * ax / sumSquares(ax - 1.0f, ay)) / 4.0f;
    }

    float ry;
    if (ax == 1.0f) {
        ry = float.ext_atan2(2.0f, -ay) / 2.0f;
    } else if (ay < FLT_EPSILON) {
        ry = float.ext_atan2(2.0f * ay, (1.0f - ax) * (1.0f + ax)) / 2.0f;
    } else {
        ry = float.ext_atan2(2.0f * ay, (1.0f - ax) * (1.0f + ax) - ay * ay) / 2.0f;
    }
    return float2(copySign(rx, x), copySign(ry, y));
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void complex_arc_main<T : __BuiltinFloatingPointType>(
    uint3 gi : SV_DispatchThreadID,
    uniform ComplexArcPushConstants pcs)
{
    uint idx = flat_thread(gi, WG_x);
    if (idx >= pcs.numel) return;

    float* input = (float*)pcs.input;
    float* output = (float*)pcs.output;
    float x = input[2 * idx];
    float y = input[2 * idx + 1];

    float2 r;
    if (OP == ASINH) {
        r = casinh(x, y);
    } else if (OP == ASIN) {
        r = casinh(y, x).yx;
    } else if (OP == ACOS) {
        r = cacos(x, y);
    } else if (OP == ACOSH) {
        r = cacosh(x, y);
    } else if (OP == ATANH) {
        r = catanh(x, y);
    } else {
        r = catanh(y, x).yx;
    }
    output[2 * idx] = r.x;
    output[2 * idx + 1] = r.y;
}
