SHADER_DIR = 'src/shaders'

UNSIGNED_INTEGERS = [
    [{'name' : 'u64', 'dtype' : 'uint64_t', 'bytes' : 8}],
    [{'name' : 'u32', 'dtype' : 'uint32_t', 'bytes' : 4}],
    [{'name' : 'u16', 'dtype' : 'uint16_t', 'bytes' : 2}],
    [{'name' : 'u8',  'dtype' : 'uint8_t',  'bytes' : 1}]
]

SIGNED_INTEGERS = [
    [{'name' : 'i64', 'dtype' : 'int64_t', 'bytes' : 8}],
    [{'name' : 'i32', 'dtype' : 'int32_t', 'bytes' : 4}],
    [{'name' : 'i16', 'dtype' : 'int16_t', 'bytes' : 2}],
    [{'name' : 'i8',  'dtype' : 'int8_t',  'bytes' : 1}]
]

FLOATS = [
    [{'name' : 'f64', 'dtype' : 'float64_t', 'bytes' : 8}],
    [{'name' : 'f32', 'dtype' : 'float32_t', 'bytes' : 4}],
    [{'name' : 'f16', 'dtype' : 'float16_t', 'bytes' : 2}]
]

BYTES = [
    [{'name' : '16', 'dtype' : 'uint4',    'bytes' : 16}],
    [{'name' : '8',  'dtype' : 'uint64_t', 'bytes' : 8}],
    [{'name' : '4',  'dtype' : 'uint32_t', 'bytes' : 4}],
    [{'name' : '2',  'dtype' : 'uint16_t', 'bytes' : 2}],
    [{'name' : '1',  'dtype' : 'uint8_t',  'bytes' : 1}]
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
        'name' : 'matmul.slang.j2',
        'dtypes' : DTYPES,
        'kwargs' : {}
    },
]

