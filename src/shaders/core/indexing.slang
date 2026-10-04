module indexing;

/*
Turns a linear element index into coordinates, one dimension at a time:

    var walker = IndexWalker(md.sizes, NDIM, idx);
    for (uint i = 0; i < NDIM; i++)
        offset += walker.next(i) * strides[i];

`sizes` is the address of the host's table of sizes and fast-division constants.
*/
public struct IndexWalker
{
    uint64_t sizes;
    uint ndim;
    uint linear_idx;

    public __init(uint64_t sizes, uint ndim, uint linear_idx)
    {
        this.sizes = sizes;
        this.ndim = ndim;
        this.linear_idx = linear_idx;
    }

    private uint fast_div(uint64_t d, uint n, uint dim, uint ndim)
    {
        uint *base = (uint *)d;
        uint multiplier = base[ndim + dim];
        uint shift_val = base[2 * ndim + dim];
        uint t = (uint)(((uint64_t)n * multiplier) >> 32);
        return (uint)(((uint64_t)t + n) >> shift_val);
    }

    private uint get_divisor(uint64_t d, int dim)
    {
        uint *base = (uint *)d;
        return base[dim];
    }

    // returns the coordinate along `dim`, consuming it from the linear index
    [mutating]
    public uint next(uint dim)
    {
        uint next_linear_idx = fast_div(sizes, linear_idx, dim, ndim);
        uint divisor = get_divisor(sizes, dim);
        uint coord = linear_idx - (next_linear_idx * divisor);
        linear_idx = next_linear_idx;
        return coord;
    }
}
