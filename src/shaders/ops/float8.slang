module float8;

import core.indexing;

// float8 codes -> a 2- or 4-byte dtype in one pass: each code indexes a 256-entry table of the destination's bit patterns,
// which the host builds with c10's own conversions, so the result matches the CPU bit for bit. T only sets the output width.

[[vk::constant_id(0)]] const uint WG_x = 256;
// source and destination are 16-byte aligned, so whole runs of 16 codes load and store as uint4
[[vk::constant_id(1)]] const uint IS_ALIGNED = 0;

struct Float8DecodePushConstants
{
    uint64_t src;
    uint64_t dst;
    uint64_t table;
    uint64_t numel;
}

static groupshared uint table_bits[256];

uint codeAt(uint4 codes, uint i)
{
    return (codes[i / 4] >> (8 * (i % 4))) & 0xffu;
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void float8_decode_main<T>(
    uint3 gi : SV_DispatchThreadID,
    uint3 li : SV_GroupThreadID,
    uniform Float8DecodePushConstants pcs)
{
    for (uint i = li.x; i < 256; i += WG_x) table_bits[i] = ((uint*)pcs.table)[i];
    GroupMemoryBarrierWithGroupSync();

    uint64_t first = uint64_t(flat_thread(gi, WG_x)) * 16;
    if (first >= pcs.numel) return;

    if (IS_ALIGNED == 1 && first + 16 <= pcs.numel)
    {
        uint4 codes = ((uint4*)pcs.src)[first / 16];
        if (sizeof(T) == 2)
        {
            [unroll] for (uint q = 0; q < 2; q++)
            {
                uint4 out;
                [unroll] for (uint w = 0; w < 4; w++)
                {
                    uint i = q * 8 + w * 2;
                    out[w] = table_bits[codeAt(codes, i)] | (table_bits[codeAt(codes, i + 1)] << 16);
                }
                ((uint4*)pcs.dst)[first / 8 + q] = out;
            }
        }
        else
        {
            [unroll] for (uint q = 0; q < 4; q++)
            {
                uint4 out;
                [unroll] for (uint w = 0; w < 4; w++) out[w] = table_bits[codeAt(codes, q * 4 + w)];
                ((uint4*)pcs.dst)[first / 4 + q] = out;
            }
        }
        return;
    }

    for (uint64_t i = first; i < min(first + 16, pcs.numel); i++)
    {
        uint bits = table_bits[uint(((uint8_t*)pcs.src)[i])];
        if (sizeof(T) == 2) ((uint16_t*)pcs.dst)[i] = uint16_t(bits);
        else ((uint*)pcs.dst)[i] = bits;
    }
}
