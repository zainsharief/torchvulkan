module matmul_simd;

__exported import core.dtype;
__exported import ops.unary_ops;

[[vk::constant_id(0)]] public const uint WG_x = 0;
[[vk::constant_id(1)]] public const uint WG_y = 0;
[[vk::constant_id(2)]] public const uint IS_ALIGNED = 1;
[[vk::constant_id(3)]] public const uint HAS_BIAS = 0;
[[vk::constant_id(4)]] public const uint ACTIVATION = 0;

public struct TensorStrides
{
    public uint batch;
    public uint row;
    public uint col;
    public uint padding;
}

public struct MatmulPushConstants<T>
{
    public uint64_t tensor_a;
    public uint64_t tensor_b;
    public uint64_t tensor_out;
    public uint64_t tensor_bias;
    public TensorStrides strides_A;
    public TensorStrides strides_B;
    public TensorStrides strides_C;
    public TensorStrides strides_bias;
    public uint M;
    public uint N;
    public uint K;
    public uint padding;
    public T alpha;
    public T beta;
}

public static const int TILE_N = 128;
public static const int TILE_K = 16;

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

public struct StridedMatrix<T> : IMatmulLoader<T>, IMatmulStore<T>
{
    public uint64_t address;
    public TensorStrides strides;

    public __init(uint64_t address, TensorStrides strides)
    {
        this.address = address;
        this.strides = strides;
    }

    uint offset(uint batch, uint row, uint col)
    {
        uint col_offset = (IS_ALIGNED == 1) ? col : col * strides.col;
        return batch * strides.batch + row * strides.row + col_offset;
    }

    public T load(uint batch, uint row, uint col)
    {
        T* tensor = (T*)address;
        return tensor[offset(batch, row, col)];
    }

    public void store(uint batch, uint row, uint col, T value)
    {
        T* tensor = (T*)address;
        tensor[offset(batch, row, col)] = value;
    }
}

public struct AccumMatrix<T : ITorchVulkan> : IMatmulLoader<T.Accum>, IMatmulStore<T.Accum>
{
    public StridedMatrix<T> matrix;

    public __init(uint64_t address, TensorStrides strides)
    {
        this.matrix = StridedMatrix<T>(address, strides);
    }

    public T.Accum load(uint batch, uint row, uint col)
    {
        return matrix.load(batch, row, col).toAccum();
    }

    public void store(uint batch, uint row, uint col, T.Accum value)
    {
        matrix.store(batch, row, col, T.fromAccum(value));
    }
}

public struct LinearEpilogue<T : ITorchVulkan, Act : IUnaryOp<T>, Bias : IMatmulLoader<T>> : IMatmulEpilogue<T>
{
    public T alpha;
    public T beta;
    public Bias bias;

    public __init(T alpha, T beta, Bias bias)
    {
        this.alpha = alpha;
        this.beta = beta;
        this.bias = bias;
    }

    public T apply(uint batch, uint row, uint col, T acc)
    {
        T value = T.ext_mul(acc, alpha);
        if (HAS_BIAS == 1) value = T.ext_add(value, T.ext_mul(bias.load(batch, row, col), beta));
        return Act.eval(value);
    }
}

public struct Activation<T : ITorchVulkan> : IUnaryOp<T>
{
    public static T eval(T a)
    {
        switch (ACTIVATION)
        {
        case 1: return ReluOp<T>.eval(a);
        case 2: return GeluOp<T>.eval(a);
        default: return a;
        }
    }
}

public void matmulTile<
    T : __BuiltinArithmeticType,
    let VEC : int,
    let TILE_M : int,
    A : IMatmulLoader<T>,
    B : IMatmulLoader<T>,
    E : IMatmulEpilogue<T>,
    S : IMatmulStore<T>>(
    uint3 li,
    uint3 wg,
    uint batch,
    __ref T smemA[TILE_M * (TILE_K + 1)],
    __ref vector<T, VEC> smemB[TILE_K * (TILE_N / VEC)],
    A a,
    B b,
    E epilogue,
    S out,
    uint M,
    uint N,
    uint K)
{
    const int THREAD_SIZE_M = TILE_M / TILE_K;
    const int THREAD_SIZE_N = TILE_N / TILE_K;
    const int VECTORS_PER_TILE_ROW_N = TILE_N / VEC;
    const int VECTORS_PER_TILE_ROW_K = TILE_K / VEC;
    const int VECTORS_PER_THREAD_ROW_M = THREAD_SIZE_N / VEC;

    vector<T, VEC> accum[THREAD_SIZE_M * (THREAD_SIZE_N / VEC)];
    [ForceUnroll] for (int acc = 0; acc < THREAD_SIZE_M * (THREAD_SIZE_N / VEC); ++acc) accum[acc] = vector<T, VEC>(T(0));

    uint thread_idx = li.y * WG_x + li.x;
    uint threads = WG_x * WG_y;
    uint thread_row_start = li.y * THREAD_SIZE_M;
    uint thread_col_start = li.x * THREAD_SIZE_N;

    uint vectors_A = (TILE_M * TILE_K) / VEC;
    uint vectors_B = (TILE_K * TILE_N) / VEC;
    uint loads_per_thread_A = vectors_A / threads;
    uint loads_per_thread_B = vectors_B / threads;

    for (uint k_block = 0; k_block < K; k_block += TILE_K)
    {
        // loads a TILE_M x TILE_K block of A into shared memory
        [unroll] for (int i = 0; i < loads_per_thread_A; ++i)
        {
            uint load_idx = thread_idx + (i * threads);
            uint row_A = load_idx / VECTORS_PER_TILE_ROW_K;
            uint col_A = (load_idx % VECTORS_PER_TILE_ROW_K) * VEC;

            uint global_row = wg.y * TILE_M + row_A;
            uint global_col = k_block + col_A;

            if (IS_ALIGNED == 1)
            {
                vector<T, VEC> loaded_vec = vector<T, VEC>(T(0));
                if (global_row < M && global_col < K)
                {
                    [unroll] for (int v = 0; v < VEC; ++v) loaded_vec[v] = a.load(batch, global_row, global_col + v);
                }

                [unroll] for (int v = 0; v < VEC; ++v) smemA[row_A * (TILE_K + 1) + col_A + v] = loaded_vec[v];
            }
            else
            {
                // bounds check each element, so a non-multiple tail is padded with zero
                [unroll] for (int v = 0; v < VEC; ++v)
                {
                    if (global_row < M && (global_col + v) < K)
                    {
                        smemA[row_A * (TILE_K + 1) + col_A + v] = a.load(batch, global_row, global_col + v);
                    }
                    else
                    {
                        smemA[row_A * (TILE_K + 1) + col_A + v] = T(0);
                    }
                }
            }
        }

        // loads a TILE_K x TILE_N block of B into shared memory
        [unroll] for (int i = 0; i < loads_per_thread_B; ++i)
        {
            uint load_idx = thread_idx + (i * threads);
            uint row_B = load_idx / VECTORS_PER_TILE_ROW_N;
            uint col_B = (load_idx % VECTORS_PER_TILE_ROW_N) * VEC;

            uint global_row = k_block + row_B;
            uint global_col = wg.x * TILE_N + col_B;

            vector<T, VEC> loaded_vec = vector<T, VEC>(T(0));

            if (IS_ALIGNED == 1)
            {
                if (global_row < K && global_col < N)
                {
                    [unroll] for (int v = 0; v < VEC; ++v) loaded_vec[v] = b.load(batch, global_row, global_col + v);
                }
            }
            else
            {
                [unroll] for (int v = 0; v < VEC; ++v)
                {
                    if (global_row < K && (global_col + v) < N)
                    {
                        loaded_vec[v] = b.load(batch, global_row, global_col + v);
                    }
                }
            }

            uint vec_col = col_B / VEC;
            smemB[row_B * VECTORS_PER_TILE_ROW_N + vec_col] = loaded_vec;
        }

        GroupMemoryBarrierWithGroupSync();

        // compute the matmul for this tile
        for (int k = 0; k < TILE_K; k++)
        {
            vector<T, VEC> b_frag[THREAD_SIZE_N / VEC];

            [unroll] for (int j = 0; j < THREAD_SIZE_N / VEC; ++j)
            {
                uint base_vec_col = (thread_col_start / VEC) + j;
                b_frag[j] = smemB[k * VECTORS_PER_TILE_ROW_N + base_vec_col];
            }

            [unroll] for (int i = 0; i < THREAD_SIZE_M; ++i)
            {
                T a_val = smemA[(thread_row_start + i) * (TILE_K + 1) + k];

                [unroll] for (int j = 0; j < THREAD_SIZE_N / VEC; ++j)
                {
                    accum[i * VECTORS_PER_THREAD_ROW_M + j] += b_frag[j] * a_val;
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    uint global_row_start = wg.y * TILE_M + thread_row_start;
    uint global_col_start = wg.x * TILE_N + thread_col_start;

    // write the results back to global memory
    [unroll] for (int i = 0; i < THREAD_SIZE_M; ++i)
    {
        uint g_row = global_row_start + i;

        [unroll] for (int j_vec = 0; j_vec < VECTORS_PER_THREAD_ROW_M; ++j_vec)
        {
            uint g_col_base = global_col_start + (j_vec * VEC);
            vector<T, VEC> result_vec = accum[i * VECTORS_PER_THREAD_ROW_M + j_vec];

            if (IS_ALIGNED == 1)
            {
                if (g_row < M && g_col_base < N)
                {
                    [unroll] for (int v = 0; v < VEC; ++v)
                    {
                        uint g_col = g_col_base + v;
                        out.store(batch, g_row, g_col, epilogue.apply(batch, g_row, g_col, result_vec[v]));
                    }
                }
            }
            else
            {
                [unroll] for (int v = 0; v < VEC; ++v)
                {
                    uint g_col = g_col_base + v;
                    if (g_row < M && g_col < N)
                    {
                        out.store(batch, g_row, g_col, epilogue.apply(batch, g_row, g_col, result_vec[v]));
                    }
                }
            }
        }
    }
}

[shader("compute")]
[numthreads(WG_x, WG_y, 1)]
void matmul_simd_main<T : ITorchVulkan, let VEC : int, let TILE_M : int>(
    uint3 li : SV_GroupThreadID,
    uint3 wg : SV_GroupID,
    uniform MatmulPushConstants<T.Accum> pcs)
{
    typealias Acc = T.Accum;
    typealias Epilogue = LinearEpilogue<Acc, Activation<Acc>, AccumMatrix<T>>;

    static groupshared Acc smemA[TILE_M * (TILE_K + 1)];
    static groupshared vector<Acc, VEC> smemB[TILE_K * (TILE_N / VEC)];

    AccumMatrix<T> a = AccumMatrix<T>(pcs.tensor_a, pcs.strides_A);
    AccumMatrix<T> b = AccumMatrix<T>(pcs.tensor_b, pcs.strides_B);
    AccumMatrix<T> bias = AccumMatrix<T>(pcs.tensor_bias, pcs.strides_bias);
    AccumMatrix<T> out = AccumMatrix<T>(pcs.tensor_out, pcs.strides_C);
    Epilogue epilogue = Epilogue(pcs.alpha, pcs.beta, bias);

    matmulTile<Acc, VEC, TILE_M, AccumMatrix<T>, AccumMatrix<T>, Epilogue, AccumMatrix<T>>(
        li, wg, wg.z, smemA, smemB, a, b, epilogue, out, pcs.M, pcs.N, pcs.K);
}
