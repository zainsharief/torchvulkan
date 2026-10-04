module scan;

__exported import core.dtype;
import core.indexing;

[[vk::constant_id(0)]] public const uint OP_TYPE = 0;
[[vk::constant_id(1)]] public const uint NDIM = 0;
[[vk::constant_id(2)]] public const uint WG_x = 0;

public struct Metadata
{
    public uint64_t sizes;
    public uint64_t strides_in;
    public uint64_t strides_out;
}

public struct ScanPushConstants
{
    public uint64_t tensor_in;
    public uint64_t tensor_out;
    public uint64_t metadata;
    public uint64_t num_lines;
    public uint scan_size;
    public uint scan_stride_in;
    public uint scan_stride_out;
}

public interface ICumScanOp<T>
{
    static T identity();
    static T combine(T a, T b);
}

public struct SumScanOp<T : ITorchVulkan> : ICumScanOp<T>
{
    public static T identity() { return T.ext_zero(); }
    public static T combine(T a, T b) { return T.ext_add(a, b); }
}

public struct ProdScanOp<T : ITorchVulkan> : ICumScanOp<T>
{
    public static T identity() { return T.ext_one(); }
    public static T combine(T a, T b) { return T.ext_mul(a, b); }
}

public void scanKernel<T : ITorchVulkan, Op : ICumScanOp<T>, let SMEM_SIZE : int>(
    uint line,
    uint lane,
    __ref T smem[SMEM_SIZE],
    uint64_t address_in,
    uint64_t address_out,
    uint64_t address_metadata,
    uint64_t num_lines,
    uint scan_size,
    uint scan_stride_in,
    uint scan_stride_out)
{
    if (line >= num_lines || scan_size == 0) return;

    T* tensor_in = (T*)address_in;
    T* tensor_out = (T*)address_out;

    Metadata md = *(Metadata*)address_metadata;
    uint* strides_in = (uint*)md.strides_in;
    uint* strides_out = (uint*)md.strides_out;

    uint base_in = 0;
    uint base_out = 0;

    var walker = IndexWalker(md.sizes, NDIM, line);
    for (uint i = 0; i < NDIM; i++)
    {
        uint coord = walker.next(i);
        base_in += coord * strides_in[i];
        base_out += coord * strides_out[i];
    }

    T carry = Op.identity();

    for (uint chunk_start = 0; chunk_start < scan_size; chunk_start += WG_x)
    {
        // make sure every thread is done reading last chunk's smem before we overwrite it
        GroupMemoryBarrierWithGroupSync();

        uint j = chunk_start + lane;
        T value = (j < scan_size) ? tensor_in[base_in + j * scan_stride_in] : Op.identity();
        smem[lane] = value;
        GroupMemoryBarrierWithGroupSync();

        [loop] for (uint step = 1; step < WG_x; step *= 2)
        {
            T v = smem[lane];
            if (lane >= step) v = Op.combine(smem[lane - step], smem[lane]);
            GroupMemoryBarrierWithGroupSync(); // every lane has read before any lane writes
            smem[lane] = v;
            GroupMemoryBarrierWithGroupSync(); // every lane sees this step's result before the next
        }

        if (j < scan_size) tensor_out[base_out + j * scan_stride_out] = Op.combine(carry, smem[lane]);

        uint last_lane = min(WG_x, scan_size - chunk_start) - 1;
        carry = Op.combine(carry, smem[last_lane]);
    }
}

public void dispatchScan<T : ITorchVulkan, let SMEM_SIZE : int>(
    uint line, uint lane, __ref T smem[SMEM_SIZE], ScanPushConstants pcs)
{
    switch (OP_TYPE)
    {
    case 0: scanKernel<T, SumScanOp<T>, SMEM_SIZE>(line, lane, smem, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.num_lines, pcs.scan_size, pcs.scan_stride_in, pcs.scan_stride_out); break;
    case 1: scanKernel<T, ProdScanOp<T>, SMEM_SIZE>(line, lane, smem, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.num_lines, pcs.scan_size, pcs.scan_stride_in, pcs.scan_stride_out); break;
    default: break;
    }
}

public static const int SCAN_SMEM_SIZE = 256;

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void scan_main<T : ITorchVulkan>(
    uint3 group_id : SV_GroupID,
    uint3 group_thread_id : SV_GroupThreadID,
    uniform ScanPushConstants pcs)
{
    static groupshared T smem[SCAN_SMEM_SIZE];
    dispatchScan<T, SCAN_SMEM_SIZE>(group_id.x, group_thread_id.x, smem, pcs);
}
