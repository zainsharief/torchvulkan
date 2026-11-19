module gemm_coop;

__exported import core.dtype;
__exported import helpers.gemm;

[[vk::constant_id(0)]] public const uint WORKGROUP_SIZE = 128;
[[vk::constant_id(1)]] public const uint SUBGROUP_SIZE = 32;
[[vk::constant_id(2)]] public const uint WARPS_M = 2;
[[vk::constant_id(3)]] public const uint WARPS_N = 2;
[[vk::constant_id(4)]] public const uint WARP_FRAGS_M = 2;
[[vk::constant_id(5)]] public const uint WARP_FRAGS_N = 2;
[[vk::constant_id(6)]] public const uint BK = 16;

[[vk::constant_id(7)]] public const uint HAS_ALPHA = 0;
[[vk::constant_id(8)]] public const uint ACTIVATION = 0;
[[vk::constant_id(9)]] public const uint HAS_BIAS = 0;

[[vk::constant_id(10)]] public const uint A_TRANSPOSED = 0;
[[vk::constant_id(11)]] public const uint B_TRANSPOSED = 0;
[[vk::constant_id(12)]] public const uint OUT_ALIGNED = 0;
[[vk::constant_id(13)]] public const uint LOADS_A = 1;

[[vk::constant_id(14)]] public const uint M_ALIGNED = 0;
[[vk::constant_id(15)]] public const uint N_ALIGNED = 0;

[[vk::constant_id(16)]] public const uint A_VEC_ALIGNED = 0;
[[vk::constant_id(17)]] public const uint B_VEC_ALIGNED = 0;

[[vk::constant_id(18)]] public const uint BLOCK_SIZE = 16;

// Slang can't size a CoopMat by a spec constant: scripts/splice_coop_dims.py swaps this placeholder for BLOCK_SIZE,
// so it must only appear in the typealiases below
static const uint COOP_DIM = 1009;

[[vk::constant_id(19)]] public const uint USE_SHARED_MEMORY = 1;
[[vk::constant_id(20)]] public const uint LOADS_B = 1;

// padding per shared-memory row, so consecutive rows fall in different banks; 8 keeps rows 16-byte aligned
public static const uint PAD = 8;

public static const uint SMEM_ROWS_A = WARPS_M * WARP_FRAGS_M * BLOCK_SIZE;
public static const uint SMEM_COLS_B = WARPS_N * WARP_FRAGS_N * BLOCK_SIZE;

// Slang can only emit a select of spec constants inside an array size, so code uses these
public uint stageA() { return A_TRANSPOSED == 1 ? BK * (SMEM_ROWS_A + PAD) : SMEM_ROWS_A * (BK + PAD); }
public uint stageB() { return B_TRANSPOSED == 1 ? SMEM_COLS_B * (BK + PAD) : BK * (SMEM_COLS_B + PAD); }

public struct Shared<T>
{
    public static groupshared T smemA[USE_SHARED_MEMORY == 1 ? 2 * (A_TRANSPOSED == 1 ? BK * (SMEM_ROWS_A + PAD) : SMEM_ROWS_A * (BK + PAD)) : SMEM_ROWS_A * BK];
    public static groupshared T smemB[USE_SHARED_MEMORY == 1 ? 2 * (B_TRANSPOSED == 1 ? SMEM_COLS_B * (BK + PAD) : BK * (SMEM_COLS_B + PAD)) : BK * SMEM_COLS_B];
}

public struct EpilogueShared<U>
{
    public static groupshared U smem[WARPS_M * WARPS_N * BLOCK_SIZE * BLOCK_SIZE];
}

public typealias Accumulator<U : __BuiltinArithmeticType> =
    linalg::CoopMat<U, MemoryScope.Subgroup, COOP_DIM, COOP_DIM, linalg::CoopMatMatrixUse.MatrixAccumulator>;
public typealias Output<T : ICoopElement> =
    linalg::CoopMat<T, MemoryScope.Subgroup, COOP_DIM, COOP_DIM, linalg::CoopMatMatrixUse.MatrixAccumulator>;
public typealias FragmentA<T : ICoopElement> =
    linalg::CoopMat<T, MemoryScope.Subgroup, COOP_DIM, COOP_DIM, linalg::CoopMatMatrixUse.MatrixA>;
public typealias FragmentB<T : ICoopElement> =
    linalg::CoopMat<T, MemoryScope.Subgroup, COOP_DIM, COOP_DIM, linalg::CoopMatMatrixUse.MatrixB>;
