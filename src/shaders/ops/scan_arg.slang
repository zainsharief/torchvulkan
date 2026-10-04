module scan_arg;

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

public struct ScanArgPushConstants
{
    public uint64_t tensor_in;
    public uint64_t tensor_val;
    public uint64_t tensor_idx;
    public uint64_t metadata;
    public uint64_t num_lines;
    public uint scan_size;
    public uint scan_stride_in;
    public uint scan_stride_out;
}

public struct ScanArgPair<T>
{
    public T val;
    public uint idx;
}

public ScanArgPair<T> combinePair<T : ITorchVulkan>(ScanArgPair<T> a, ScanArgPair<T> b)
{
    if (T.ext_isnan(a.val)) return a;
    if (T.ext_isnan(b.val)) return b;
    bool take_b = (OP_TYPE == 0) ? T.ext_ge(b.val, a.val) : T.ext_le(b.val, a.val);
    return take_b ? b : a;
}

public void scanArgKernel<T : ITorchVulkan, let SMEM_SIZE : int>(
    uint line,
    uint lane,
    __ref T smem_val[SMEM_SIZE],
    __ref uint smem_idx[SMEM_SIZE],
    uint64_t address_in,
    uint64_t address_val,
    uint64_t address_idx,
    uint64_t address_metadata,
    uint64_t num_lines,
    uint scan_size,
    uint scan_stride_in,
    uint scan_stride_out)
{
    if (line >= num_lines || scan_size == 0) return;

    T* tensor_in = (T*)address_in;
    T* tensor_val = (T*)address_val;
    int64_t* tensor_idx = (int64_t*)address_idx;

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

    ScanArgPair<T> carry;
    carry.val = (OP_TYPE == 0) ? T.ext_lowest() : T.ext_highest();
    carry.idx = 0;

    for (uint chunk_start = 0; chunk_start < scan_size; chunk_start += WG_x)
    {
        // make sure every thread is done reading last chunk's smem before we overwrite it
        GroupMemoryBarrierWithGroupSync();

        uint j = chunk_start + lane;
        ScanArgPair<T> value;
        if (j < scan_size) { value.val = tensor_in[base_in + j * scan_stride_in]; value.idx = j; }
        else { value.val = carry.val; value.idx = 0; }
        smem_val[lane] = value.val;
        smem_idx[lane] = value.idx;
        GroupMemoryBarrierWithGroupSync();

        [loop] for (uint step = 1; step < WG_x; step *= 2)
        {
            ScanArgPair<T> v;
            v.val = smem_val[lane]; v.idx = smem_idx[lane];
            if (lane >= step)
            {
                ScanArgPair<T> prev;
                prev.val = smem_val[lane - step]; prev.idx = smem_idx[lane - step];
                v = combinePair<T>(prev, v);
            }
            GroupMemoryBarrierWithGroupSync(); // every lane has read before any lane writes
            smem_val[lane] = v.val; smem_idx[lane] = v.idx;
            GroupMemoryBarrierWithGroupSync(); // every lane sees this step's result before the next
        }

        if (j < scan_size)
        {
            ScanArgPair<T> local;
            local.val = smem_val[lane]; local.idx = smem_idx[lane];
            ScanArgPair<T> result = combinePair<T>(carry, local);
            tensor_val[base_out + j * scan_stride_out] = result.val;
            tensor_idx[base_out + j * scan_stride_out] = int64_t(result.idx);
        }

        uint last_lane = min(WG_x, scan_size - chunk_start) - 1;
        ScanArgPair<T> last;
        last.val = smem_val[last_lane]; last.idx = smem_idx[last_lane];
        carry = combinePair<T>(carry, last);
    }
}

public static const int SCAN_ARG_SMEM_SIZE = 256;

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void scan_arg_main<T : ITorchVulkan>(
    uint3 group_id : SV_GroupID,
    uint3 group_thread_id : SV_GroupThreadID,
    uniform ScanArgPushConstants pcs)
{
    static groupshared T smem_val[SCAN_ARG_SMEM_SIZE];
    static groupshared uint smem_idx[SCAN_ARG_SMEM_SIZE];

    scanArgKernel<T, SCAN_ARG_SMEM_SIZE>(
        group_id.x, group_thread_id.x, smem_val, smem_idx,
        pcs.tensor_in, pcs.tensor_val, pcs.tensor_idx, pcs.metadata,
        pcs.num_lines, pcs.scan_size, pcs.scan_stride_in, pcs.scan_stride_out
    );
}
