"""
Which shader variants are compiled ahead of time.

Each Kernel names a Slang module and its generic entry point; each Entry lists
the dtypes to specialise it for. One variant is built as:

    slangc ops/binary.slang -entry binary_main -specialize float32_t -specialize 4

The host looks a variant up by ShaderKey{Kernel::BINARY, dtype}. To add a
kernel, write its module in src/shaders/ops/ and add a Kernel here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DType:
    name:  str   # short name; part of the generated .spv filename
    slang: str   # the Slang type the entry point is specialised with
    bytes: int   # size in bytes
    aten:  tuple # the c10::ScalarType values this variant serves


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


@dataclass(frozen=True)
class Kernel:
    name: str
    module: str # path relative to src/shaders
    entries: list 


KERNELS = [
    Kernel('binary', 'ops/binary.slang', [
        Entry('binary_main', DTYPES, ('vec',)),
    ]),
    Kernel('unary', 'ops/unary.slang', [
        Entry('unary_main', DTYPES, ('vec',)),
    ]),
    Kernel('compare', 'ops/compare.slang', [
        Entry('compare_main', DTYPES, ('vec',)),
    ]),
    Kernel('integer_binary', 'ops/integer_binary.slang', [
        Entry('integer_binary_main', INTEGERS),
    ]),
    Kernel('special', 'ops/special.slang', [
        Entry('special_main', FLOATS),
    ]),
    Kernel('fill', 'ops/fill.slang', [
        Entry('fill_main', DTYPES, ('vec',)),
    ]),
    Kernel('where', 'ops/where.slang', [
        Entry('where_main', DTYPES),
    ]),
    Kernel('reduce', 'ops/reduce.slang', [
        Entry('reduce_main', DTYPES),
    ]),
    Kernel('reduce_subgroup', 'ops/reduce_subgroup.slang', [
        Entry('reduce_subgroup_main', [F32, F16]),
    ]),
    Kernel('arg_reduce', 'ops/arg_reduce.slang', [
        Entry('arg_reduce_main', DTYPES),
    ]),
    Kernel('scan', 'ops/scan.slang', [
        Entry('scan_main', DTYPES),
    ]),
    Kernel('scan_arg', 'ops/scan_arg.slang', [
        Entry('scan_arg_main', DTYPES),
    ]),
    Kernel('nllloss', 'ops/nllloss.slang', [
        Entry('nllloss_main', FLOATS),
    ]),
    Kernel('copy', 'ops/copy.slang', [
        Entry('copy_main', BYTE_WIDTHS),
    ]),
    Kernel('index_gather', 'ops/index_gather.slang', [
        Entry('index_gather_main', BYTE_WIDTHS),
    ]),
    Kernel('embedding_backward', 'ops/embedding_backward.slang', [
        Entry('embedding_backward_main', FLOATS),
    ]),
    Kernel('sort', 'ops/sort.slang', [
        Entry('sort_main', DTYPES),
    ]),
    Kernel('scatter', 'ops/scatter.slang', [
        Entry('scatter_main', DTYPES),
    ]),
    Kernel('searchsorted', 'ops/searchsorted.slang', [
        Entry('searchsorted_main', DTYPES),
    ]),
    Kernel('pool', 'ops/pool.slang', [
        Entry('pool_main', DTYPES),
    ]),
    Kernel('resample', 'ops/resample.slang', [
        Entry('resample_main', [F64, F32]),
    ]),
    Kernel('int_mm', 'ops/int_mm.slang', [
        Entry('int_mm_main', [I8]),
    ]),
    Kernel('resample_u8', 'ops/resample_u8.slang', [
        Entry('resample_u8_main', [U8]),
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
    Kernel('fft', 'ops/fft.slang', [
        Entry('fft_main', [F64, F32]),
    ]),
    Kernel('cast', 'ops/cast.slang', [
        Entry('cast_main', DTYPES, ('vec',), pairwise=True),
    ]),
    Kernel('matmul_simd_128', 'ops/matmul_simd.slang', [
        Entry('matmul_simd_main', [d for d in DTYPES if d.bytes < 8], ('vec',), constants=(128,)),
    ]),
    Kernel('matmul_simd_64', 'ops/matmul_simd.slang', [
        Entry('matmul_simd_main', DTYPES, ('vec',), constants=(64,)),
    ]),
    Kernel('conv_128', 'ops/conv.slang', [
        Entry('conv_main', [d for d in FLOATS if d.bytes < 8], ('vec',), constants=(128,)),
    ]),
    Kernel('conv_64', 'ops/conv.slang', [
        Entry('conv_main', FLOATS, ('vec',), constants=(64,)),
    ]),
    Kernel('matmul_coop_8', 'ops/matmul_coop.slang', [
        Entry('matmul_coop_main', DTYPES, ('vec',), constants=(8,)),
    ]),
    Kernel('matmul_coop_16', 'ops/matmul_coop.slang', [
        Entry('matmul_coop_main', DTYPES, ('vec',), constants=(16,)),
    ]),
    Kernel('matmul_coop_32', 'ops/matmul_coop.slang', [
        Entry('matmul_coop_main', DTYPES, ('vec',), constants=(32,)),
    ]),
    Kernel('matmul_coop_64', 'ops/matmul_coop.slang', [
        Entry('matmul_coop_main', DTYPES, ('vec',), constants=(64,)),
    ]),
    Kernel('random', 'ops/random.slang', [
        Entry('random_main', DTYPES, ('vec',)),
    ]),
    Kernel('dropout', 'ops/dropout.slang', [
        Entry('dropout_main', FLOATS, ('vec',)),
    ]),
    Kernel('multinomial', 'ops/multinomial.slang', [
        Entry('multinomial_main', [F64, F32]),
    ]),
    Kernel('distribution', 'ops/distribution.slang', [
        Entry('distribution_main', FLOATS),
    ]),
]