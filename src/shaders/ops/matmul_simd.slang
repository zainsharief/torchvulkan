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
    uniform MatmulPushConstants<T.Accum> pcs) where T.Accum : __BuiltinArithmeticType
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

// one output element per thread, for an accumulate type the tiles' vector registers can't hold (Float64Emulated)
[shader("compute")]
[numthreads(WG_x, WG_y, 1)]
void matmul_soft_main<T : ITorchVulkan>(
    uint3 li : SV_GroupThreadID,
    uint3 wg : SV_GroupID,
    uniform MatmulPushConstants<T.Accum> pcs)
{
    typealias Acc = T.Accum;
    uint col = wg.x * WG_x + li.x;
    uint row = wg.y * WG_y + li.y;
    if (row >= pcs.M || col >= pcs.N) return;

    AccumMatrix<T> a = AccumMatrix<T>(pcs.tensor_a, pcs.strides_A);
    AccumMatrix<T> b = AccumMatrix<T>(pcs.tensor_b, pcs.strides_B);
    AccumMatrix<T> bias = AccumMatrix<T>(pcs.tensor_bias, pcs.strides_bias);
    AccumMatrix<T> out = AccumMatrix<T>(pcs.tensor_out, pcs.strides_C);
    LinearEpilogue<Acc, Activation<Acc>, AccumMatrix<T>> epilogue = LinearEpilogue<Acc, Activation<Acc>, AccumMatrix<T>>(
        Acc.fromPushScalar(pcs.alpha), Acc.fromPushScalar(pcs.beta), bias);

    Acc acc = Acc.ext_zero();
    for (uint k = 0; k < pcs.K; k++) acc = Acc.ext_add(acc, Acc.ext_mul(a.load(wg.z, row, k), b.load(wg.z, k, col)));
    out.store(wg.z, row, col, epilogue.apply(wg.z, row, col, acc));
}
