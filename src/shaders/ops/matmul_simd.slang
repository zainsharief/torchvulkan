module matmul_simd;

import helpers.gemm_simd;

struct MatmulPushConstants<T : ITorchVulkan>
{
    uint64_t tensor_a;
    uint64_t tensor_b;
    uint64_t tensor_out;
    uint64_t tensor_bias;
    TensorStrides strides_A;
    TensorStrides strides_B;
    TensorStrides strides_C;
    TensorStrides strides_bias;
    uint M;
    uint N;
    uint K;
    uint padding;
    T.PushScalar alpha;
    T.PushScalar beta;
}

[shader("compute")]
[numthreads(WG_x, WG_y, 1)]
void matmul_simd_main<T : ITorchVulkan, let VEC : int>(
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
    Epilogue epilogue = Epilogue(Acc.fromPushScalar(pcs.alpha), Acc.fromPushScalar(pcs.beta), bias);

    matmulTile<Acc, VEC, AccumMatrix<T>, AccumMatrix<T>, Epilogue, AccumMatrix<T>>(
        li, wg, wg.z, smemA, smemB, a, b, epilogue, out, pcs.M, pcs.N, pcs.K);
}
