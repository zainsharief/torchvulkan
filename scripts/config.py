"""
Which shader variants are compiled ahead of time.

Each Kernel names a Slang module and its generic entry point; each Entry lists
the dtypes to specialise it for. One variant is built as:

    slangc ops/binary.slang -entry binary_main -specialize float32_t -specialize 4

The host looks a variant up by ShaderKey{Kernel::BINARY, dtype}, in the mode the
device runs that dtype in (DeviceContext::dtype_mode): natively, or as its
emulated type (src/shaders/dtype/). To add a kernel, write its module in src/shaders/ops/ and add a
Kernel here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DType:
    name:  str   # short name; part of the generated .spv filename
    slang: str   # the Slang type the entry point is specialised with
    bytes: int   # size in bytes
    aten:  tuple # the c10::ScalarType values this variant serves
    mode:  str = 'native' # how it runs them: DtypeMode::<MODE> on the host


U64 = DType('u64', 'uint64_t', 8, ('UInt64',))
U32 = DType('u32', 'uint32_t', 4, ('UInt32',))
U16 = DType('u16', 'uint16_t', 2, ('UInt16',))
U8  = DType('u8',  'uint8_t',  1, ('Byte', 'Bool'))

I64 = DType('i64', 'int64_t', 8, ('Long',))
I32 = DType('i32', 'int32_t', 4, ('Int',))
I16 = DType('i16', 'int16_t', 2, ('Short',))
I8  = DType('i8',  'int8_t',  1, ('Char',))

F64 = DType('f64', 'float64_t', 8, ('Double',))
F32 = DType('f32', 'float32_t', 4, ('Float',))
F16 = DType('f16', 'float16_t', 2, ('Half',))

BF16 = DType('bf16', 'BFloat16', 2, ('BFloat16',))

UNSIGNED_INTEGERS = [U64, U32, U16, U8]
SIGNED_INTEGERS = [I64, I32, I16, I8]
INTEGERS = UNSIGNED_INTEGERS + SIGNED_INTEGERS
BUILTIN_FLOATS = [F64, F32, F16]
FLOATS = BUILTIN_FLOATS + [BF16]
DTYPES = INTEGERS + FLOATS

BYTE_WIDTHS = [
    DType('16', 'uint4',    16, ('ComplexDouble',)),
    DType('8',  'uint64_t', 8,  ('UInt64',)),
    DType('4',  'uint32_t', 4,  ('UInt32',)),
    DType('2',  'uint16_t', 2,  ('UInt16',)),
    DType('1',  'uint8_t',  1,  ('Byte',)),
]

# the dtypes a device can't run natively, each an emulated type of its own (src/shaders/dtype/)
MODES = ['native', 'emulated']

F16_EMULATED = DType('f16_emulated', 'Float16Emulated', 2, ('Half',), 'emulated')
BF16_EMULATED = DType('bf16_emulated', 'BFloat16Emulated', 2, ('BFloat16',), 'emulated')
F64_EMULATED = DType('f64_emulated', 'Float64Emulated', 8, ('Double',), 'emulated')
EMULATED_FLOATS = [F16_EMULATED, BF16_EMULATED]

FLOAT8 = [DType(name.lower(), name, 1, (aten,), 'emulated')
          for name, aten in [('Float8E4M3FN', 'Float8_e4m3fn'), ('Float8E5M2', 'Float8_e5m2'), ('Float8E4M3FNUZ', 'Float8_e4m3fnuz'),
                             ('Float8E5M2FNUZ', 'Float8_e5m2fnuz'), ('Float8E8M0FNU', 'Float8_e8m0fnu')]]

I8_EMULATED = DType('i8_emulated', 'Int8Emulated', 1, ('Char',), 'emulated')
U8_EMULATED = DType('u8_emulated', 'UInt8Emulated', 1, ('Byte',), 'emulated')
BOOL_EMULATED = DType('bool_emulated', 'BoolEmulated', 1, ('Bool',), 'emulated')
I16_EMULATED = DType('i16_emulated', 'Int16Emulated', 2, ('Short',), 'emulated')
U16_EMULATED = DType('u16_emulated', 'UInt16Emulated', 2, ('UInt16',), 'emulated')
EMULATED_INTEGERS = [I8_EMULATED, U8_EMULATED, BOOL_EMULATED, I16_EMULATED, U16_EMULATED]

ALL_DTYPES = DTYPES + EMULATED_FLOATS + EMULATED_INTEGERS + [F64_EMULATED] + FLOAT8
ALL_FLOATS = FLOATS + EMULATED_FLOATS + [F64_EMULATED] + FLOAT8
ALL_INTEGERS = INTEGERS + EMULATED_INTEGERS
BOOLS = [U8, BOOL_EMULATED] # a bool operand of a kernel computing in another dtype

# casts: every native pair; an emulated dtype to and from float32 and int64 (which hold its values exactly) and
# itself; float8 to and from the 16-bit floats (fp8 weights are decoded into the compute dtype every step); float64
# emulated to and from every native dtype. The host routes any other pair through float32 or int64
EMULATED = EMULATED_FLOATS + FLOAT8 + EMULATED_INTEGERS
CAST_PAIRS = ([(s, d) for s in DTYPES for d in DTYPES] +
              [p for e in EMULATED for b in (F32, I64) for p in ((e, b), (b, e))] +
              [p for f in FLOAT8 for h in (F16, BF16, F16_EMULATED, BF16_EMULATED) for p in ((f, h), (h, f))] +
              [(e, e) for e in EMULATED] +
              [p for n in DTYPES for p in ((F64_EMULATED, n), (n, F64_EMULATED))] + [(F64_EMULATED, F64_EMULATED)])


def vec_size(nbytes):
    """
    How many elements a thread handles per step: as many as fit in 128 bits,
    capped at 4. The host reads it from VEC_SIZE_FOR_BYTES in shader_kernels.h.
    """
    return int(min(16 / nbytes, 4))


# The integer arguments an entry point takes after its type arguments, derived
# from the dtype's size. For a pairwise entry point they come from the wider of
# the two dtypes, since both tensors are stepped by the same amount.
VALUE_ARGS = {
    'vec': vec_size,
}


@dataclass(frozen=True)
class Entry:
    function: str          # the entrypoint in the kernel's module
    dtypes: list           # the dtypes to specialise it for
    value_args: tuple = () # names from VALUE_ARGS, in the order the entry point declares them
    constants: tuple = ()  # literal integer arguments that follow the value args
    pairwise: bool = False # specialise over every (src, dst) pair rather than one dtype
    dtypes2: list = None   # a second type argument, specialised over every (dtype, dtype2) pair
    pairs: list = None     # explicit (dtype, dtype2) pairs


@dataclass(frozen=True)
class Kernel:
    name: str
    module: str # path relative to src/shaders
    entries: list
    coop_dims: tuple = None # (spec constant id, placeholder) that scripts/splice_coop_dims.py swaps into CoopMat dimensions


KERNELS = [
    Kernel('binary', 'ops/binary.slang', [
        Entry('binary_main', ALL_DTYPES, ('vec',)),
    ]),
    Kernel('unary', 'ops/unary.slang', [
        Entry('unary_main', ALL_DTYPES, ('vec',)),
    ]),
    Kernel('compare', 'ops/compare.slang', [
        Entry('compare_main', ALL_DTYPES, ('vec',), dtypes2=BOOLS),
    ]),
    Kernel('integer_binary', 'ops/integer_binary.slang', [
        Entry('integer_binary_main', ALL_INTEGERS),
    ]),
    Kernel('special', 'ops/special.slang', [
        Entry('special_main', ALL_FLOATS),
    ]),
    Kernel('fill', 'ops/fill.slang', [
        Entry('fill_main', ALL_DTYPES, ('vec',)),
    ]),
    Kernel('where', 'ops/where.slang', [
        Entry('where_main', ALL_DTYPES, dtypes2=BOOLS),
    ]),
    Kernel('addc', 'ops/addc.slang', [
        Entry('addc_main', ALL_DTYPES),
    ]),
    Kernel('reduce', 'ops/reduce.slang', [
        Entry('reduce_main', ALL_DTYPES),
    ]),
    Kernel('reduce_subgroup', 'ops/reduce_subgroup.slang', [
        Entry('reduce_subgroup_main', [F32, F16]),
    ]),
    Kernel('arg_reduce', 'ops/arg_reduce.slang', [
        Entry('arg_reduce_main', ALL_DTYPES),
    ]),
    Kernel('scan', 'ops/scan.slang', [
        Entry('scan_main', ALL_DTYPES),
    ]),
    Kernel('scan_arg', 'ops/scan_arg.slang', [
        Entry('scan_arg_main', ALL_DTYPES),
    ]),
    Kernel('norm', 'ops/norm.slang', [
        Entry('norm_main', ALL_FLOATS),
    ]),
    Kernel('attention_simd', 'ops/attention_simd.slang', [
        Entry('attention_simd_main', FLOATS),
    ]),
    Kernel('attention_simd_backward', 'ops/attention_simd.slang', [
        Entry('attention_simd_backward_main', FLOATS),
    ]),
    Kernel('nllloss', 'ops/nllloss.slang', [
        Entry('nllloss_main', ALL_FLOATS),
    ]),
    Kernel('copy', 'ops/copy.slang', [
        Entry('copy_main', BYTE_WIDTHS),
        Entry('copy_emulated_main', [U8_EMULATED, U16_EMULATED]),
    ]),
    Kernel('index_gather', 'ops/index_gather.slang', [
        Entry('index_gather_main', BYTE_WIDTHS),
        Entry('index_gather_emulated_main', [U8_EMULATED, U16_EMULATED]),
    ]),
    Kernel('embedding_backward', 'ops/embedding_backward.slang', [
        Entry('embedding_backward_main', ALL_FLOATS),
    ]),
    Kernel('sort', 'ops/sort.slang', [
        Entry('sort_main', ALL_DTYPES),
    ]),
    Kernel('scatter', 'ops/scatter.slang', [
        Entry('scatter_main', ALL_DTYPES),
    ]),
    Kernel('searchsorted', 'ops/searchsorted.slang', [
        Entry('searchsorted_main', ALL_DTYPES),
    ]),
    Kernel('pool', 'ops/pool.slang', [
        Entry('pool_main', ALL_DTYPES),
    ]),
    Kernel('resample', 'ops/resample.slang', [
        Entry('resample_main', [F64, F32]),
    ]),
    Kernel('resample_u8', 'ops/resample_u8.slang', [
        Entry('resample_u8_main', [U8, U8_EMULATED], dtypes2=[I16, I16_EMULATED]),
    ]),
    Kernel('grid_sample', 'ops/grid_sample.slang', [
        Entry('grid_sample_main', [F64, F32]),
    ]),
    Kernel('ctc', 'ops/ctc.slang', [
        Entry('ctc_main', [F64, F32]),
    ]),
    Kernel('cholesky', 'ops/cholesky.slang', [
        Entry('cholesky_main', [F64, F32]),
    ]),
    Kernel('lu', 'ops/lu.slang', [
        Entry('lu_main', [F64, F32]),
    ]),
    Kernel('triangular_solve', 'ops/triangular_solve.slang', [
        Entry('triangular_solve_main', [F64, F32]),
    ]),
    Kernel('householder', 'ops/householder.slang', [
        Entry('householder_main', [F64, F32]),
    ]),
    Kernel('jacobi', 'ops/jacobi.slang', [
        Entry('jacobi_main', [F64, F32]),
    ]),
    Kernel('ldl', 'ops/ldl.slang', [
        Entry('ldl_main', [F64, F32]),
    ]),
    Kernel('eig', 'ops/eig.slang', [
        Entry('eig_main', [F64, F32]),
    ]),
    Kernel('complex_arc', 'ops/complex_arc.slang', [
        Entry('complex_arc_main', [F32]),
    ]),
    Kernel('fft', 'ops/fft.slang', [
        Entry('fft_main', [F64, F32]),
    ]),
    Kernel('cast', 'ops/cast.slang', [
        Entry('cast_main', [], ('vec',), pairs=CAST_PAIRS),
    ]),
    Kernel('matmul_simd', 'ops/matmul_simd.slang', [
        Entry('matmul_simd_main', DTYPES + EMULATED_FLOATS + EMULATED_INTEGERS + FLOAT8, ('vec',)),
        Entry('matmul_soft_main', [F64_EMULATED]),
    ]),
    Kernel('conv_simd', 'ops/conv_simd.slang', [
        Entry('conv_simd_main', FLOATS, ('vec',)),
    ]),
    Kernel('winograd_transform', 'ops/winograd_transform.slang', [
        Entry('winograd_transform_main', FLOATS),
    ]),
    Kernel('winograd_simd', 'ops/winograd_simd.slang', [
        Entry('winograd_simd_main', FLOATS),
    ]),
    Kernel('matmul_coop', 'ops/matmul_coop.slang', [
        Entry('matmul_coop_main', DTYPES, ('vec',)),
    ], coop_dims=(18, 1009)), # BLOCK_SIZE, COOP_DIM in helpers/gemm_coop.slang
    Kernel('conv_coop', 'ops/conv_coop.slang', [
        Entry('conv_coop_main', FLOATS, ('vec',)),
    ], coop_dims=(18, 1009)),
    Kernel('attention_coop', 'ops/attention_coop.slang', [
        Entry('attention_coop_main', FLOATS),
    ], coop_dims=(11, 1009)), # BLOCK_SIZE, COOP_DIM in attention_coop.slang
    Kernel('attention_coop_backward', 'ops/attention_coop.slang', [
        Entry('attention_coop_backward_main', FLOATS),
    ], coop_dims=(11, 1009)),
    Kernel('winograd_coop', 'ops/winograd_coop.slang', [
        Entry('winograd_coop_main', FLOATS),
    ], coop_dims=(18, 1009)),
    Kernel('random', 'ops/random.slang', [
        Entry('random_main', ALL_DTYPES, ('vec',)),
    ]),
    Kernel('dropout', 'ops/dropout.slang', [
        Entry('dropout_main', ALL_FLOATS, ('vec',), dtypes2=BOOLS),
    ]),
    Kernel('multinomial', 'ops/multinomial.slang', [
        Entry('multinomial_main', [F64, F32]),
    ]),
    Kernel('distribution', 'ops/distribution.slang', [
        Entry('distribution_main', ALL_FLOATS),
    ]),
]