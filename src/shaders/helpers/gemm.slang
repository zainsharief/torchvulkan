module gemm;

public struct TensorStrides
{
    public uint batch;
    public uint row;
    public uint col;
    public uint padding;
}

public interface IMatmulLoader<T>
{
    T load(uint batch, uint row, uint col);
}

public interface IMatmulStore<T>
{
    void store(uint batch, uint row, uint col, T value);
}

public interface IMatmulEpilogue<T>
{
    T apply(uint batch, uint row, uint col, T acc);
}
