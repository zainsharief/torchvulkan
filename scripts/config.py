import sys

SHADER_DIR = 'src/shaders'

UNSIGNED_INTEGERS = [
    [{'name' : 'u64', 'dtype' : 'uint64_t', 'bytes' : 8, 'kwargs' : {}}],
    [{'name' : 'u32', 'dtype' : 'uint32_t', 'bytes' : 4, 'kwargs' : {}}],
    [{'name' : 'u16', 'dtype' : 'uint16_t', 'bytes' : 2, 'kwargs' : {}}],
    [{'name' : 'u8',  'dtype' : 'uint8_t',  'bytes' : 1, 'kwargs' : {}}]
]

SIGNED_INTEGERS = [
    [{'name' : 'i64', 'dtype' : 'int64_t', 'bytes' : 8, 'kwargs' : {}}],
    [{'name' : 'i32', 'dtype' : 'int32_t', 'bytes' : 4, 'kwargs' : {}}],
    [{'name' : 'i16', 'dtype' : 'int16_t', 'bytes' : 2, 'kwargs' : {}}],
    [{'name' : 'i8',  'dtype' : 'int8_t',  'bytes' : 1, 'kwargs' : {}}]
]

FLOATS = [
    [{'name' : 'f64', 'dtype' : 'float64_t', 'bytes' : 8, 'kwargs' : {}}],
    [{'name' : 'f32', 'dtype' : 'float32_t', 'bytes' : 4, 'kwargs' : {}}],
    [{'name' : 'f16', 'dtype' : 'float16_t', 'bytes' : 2, 'kwargs' : {}}]
]

BYTES = [
    [{'name' : '16', 'dtype' : 'uint4',    'bytes' : 16, 'kwargs' : {}}],
    [{'name' : '8',  'dtype' : 'uint64_t', 'bytes' : 8,  'kwargs' : {}}],
    [{'name' : '4',  'dtype' : 'uint32_t', 'bytes' : 4,  'kwargs' : {}}],
    [{'name' : '2',  'dtype' : 'uint16_t', 'bytes' : 2,  'kwargs' : {}}],
    [{'name' : '1',  'dtype' : 'uint8_t',  'bytes' : 1,  'kwargs' : {}}]
]

INTEGERS = UNSIGNED_INTEGERS + SIGNED_INTEGERS
DTYPES = INTEGERS + FLOATS
DTYPES_SUPERSET = [[i[0], j[0]] for i in DTYPES for j in DTYPES] 

SHADERS = [
    {
        'name' : 'binaryop.slang.j2',
        'dtypes' : DTYPES,
        'kwargs' : {'OPERATIONS' : [
                    {"name" : '0', 'struct' : 'AddOp'},
                    {"name" : '1', 'struct' : 'SubOp'},
                    {"name" : '2', 'struct' : 'RSubOp'},
                    {"name" : '3', 'struct' : 'MulOp'},
                    {"name" : '4', 'struct' : 'DivOp'},
                    {"name" : '5', 'struct' : 'MaxOp'},
                    {"name" : '6', 'struct' : 'MinOp'},
                    {"name" : '7', 'struct' : 'PowOp'},
                    {"name" : '8', 'struct' : 'RPowOp'},
                    {"name" : '9', 'struct' : 'Atan2Op'}
                ]}
    },
    {
        'name' : 'copy.slang.j2',
        'dtypes' : BYTES,
        'kwargs' : {}
    },
    {
        'name' : 'cast.slang.j2',
        'dtypes' : DTYPES_SUPERSET,
        'kwargs' : {}
    },
    {
        'name' : 'fill.slang.j2',
        'dtypes' : DTYPES,
        'kwargs' : {}
    },
    {
        'name' : 'matmul_simd.slang.j2',
        'dtypes' : DTYPES,
        'kwargs' : {}
    },
    {
        'name' : 'matmul_coop.slang.j2',
        'dtypes' : [[{**t[0], 'kwargs': {**t[0]['kwargs'], 'BLOCK_SIZE': s}}] for s in [8, 16, 32, 64] for t in DTYPES],
        'kwargs' : {'os': sys.platform}
    }
]

