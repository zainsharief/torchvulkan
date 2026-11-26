module bf16_convert;

import core.indexing;

// bfloat16 <-> float32 as bits, for devices without shader bfloat16 (can_emulate_bfloat16): only 16- and 32-bit
// integer storage is used. Widening shifts each code up 16 bits; narrowing rounds to nearest even and turns every NaN
// into 0xFFFF, as the CPU's float32 -> bfloat16 copy does. Each thread converts 8 elements; with 16-byte aligned
// buffers they move as uint4 (8 bfloat16 codes, 2 x 4 float32 words), otherwise one element at a time. T only picks
// the specialization (u32 for widen, u16 for narrow).

[[vk::constant_id(0)]] const uint WG_x = 256;
[[vk::constant_id(1)]] const uint IS_ALIGNED = 0;

struct Bf16ConvertPushConstants
{
    uint64_t src;
    uint64_t dst;
    uint64_t numel;
}

uint narrowBits(uint x)
{
    if ((x & 0x7fffffffu) > 0x7f800000u) return 0xffffu;
    return (x + 0x7fffu + ((x >> 16) & 1u)) >> 16;
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void bf16_widen_main<T>(uint3 gi : SV_DispatchThreadID, uniform Bf16ConvertPushConstants pcs)
{
    uint64_t first = uint64_t(flat_thread(gi, WG_x)) * 8;
    if (first >= pcs.numel) return;
    if (IS_ALIGNED == 1 && first + 8 <= pcs.numel)
    {
        uint4 codes = ((uint4*)pcs.src)[first / 8];
        uint4 lo, hi;
        lo.x = codes.x << 16; lo.y = codes.x & 0xffff0000u; lo.z = codes.y << 16; lo.w = codes.y & 0xffff0000u;
        hi.x = codes.z << 16; hi.y = codes.z & 0xffff0000u; hi.z = codes.w << 16; hi.w = codes.w & 0xffff0000u;
        ((uint4*)pcs.dst)[first / 4] = lo;
        ((uint4*)pcs.dst)[first / 4 + 1] = hi;
        return;
    }
    for (uint64_t i = first; i < min(first + 8, pcs.numel); i++)
        ((uint*)pcs.dst)[i] = uint(((uint16_t*)pcs.src)[i]) << 16;
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void bf16_narrow_main<T>(uint3 gi : SV_DispatchThreadID, uniform Bf16ConvertPushConstants pcs)
{
    uint64_t first = uint64_t(flat_thread(gi, WG_x)) * 8;
    if (first >= pcs.numel) return;
    if (IS_ALIGNED == 1 && first + 8 <= pcs.numel)
    {
        uint4 a = ((uint4*)pcs.src)[first / 4];
        uint4 b = ((uint4*)pcs.src)[first / 4 + 1];
        uint4 out;
        out.x = narrowBits(a.x) | (narrowBits(a.y) << 16);
        out.y = narrowBits(a.z) | (narrowBits(a.w) << 16);
        out.z = narrowBits(b.x) | (narrowBits(b.y) << 16);
        out.w = narrowBits(b.z) | (narrowBits(b.w) << 16);
        ((uint4*)pcs.dst)[first / 8] = out;
        return;
    }
    for (uint64_t i = first; i < min(first + 8, pcs.numel); i++)
        ((uint16_t*)pcs.dst)[i] = uint16_t(narrowBits(((uint*)pcs.src)[i]));
}
