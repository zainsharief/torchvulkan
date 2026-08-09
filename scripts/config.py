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
                    {"name" : '9', 'struct' : 'Atan2Op', 'float_only' : True},
                    {"name" : '10', 'struct' : 'ThresholdBackwardOp'},
                    {"name" : '11', 'struct' : 'FmaxOp'},
                    {"name" : '12', 'struct' : 'FminOp'},
                    {"name" : '13', 'struct' : 'FmodOp'},
                    {"name" : '14', 'struct' : 'RemainderOp'},
                    {"name" : '15', 'struct' : 'HypotOp', 'float_only' : True},
                    {"name" : '16', 'struct' : 'XlogyOp', 'float_only' : True},
                    {"name" : '17', 'struct' : 'LogaddexpOp', 'float_only' : True},
                    {"name" : '18', 'struct' : 'Logaddexp2Op', 'float_only' : True}
                ]}
    },
    {
        'name' : 'copy.slang.j2',
        'dtypes' : BYTES,
        'kwargs' : {}
    },
    {
        'name' : 'unaryop.slang.j2',
        'dtypes' : DTYPES,
        'kwargs' : {'OPERATIONS' : [
                    {"name" : '0', 'struct' : 'ReluOp'},
                    {"name" : '1', 'struct' : 'ExpOp', 'float_only' : True},
                    {"name" : '2', 'struct' : 'LogOp', 'float_only' : True},
                    {"name" : '3', 'struct' : 'SqrtOp', 'float_only' : True},
                    {"name" : '4', 'struct' : 'NegOp'},
                    {"name" : '5', 'struct' : 'ReciprocalOp', 'float_only' : True},
                    {"name" : '6', 'struct' : 'SinOp', 'float_only' : True},
                    {"name" : '7', 'struct' : 'CosOp', 'float_only' : True},
                    {"name" : '8', 'struct' : 'TanOp', 'float_only' : True},
                    {"name" : '9', 'struct' : 'AsinOp', 'float_only' : True},
                    {"name" : '10', 'struct' : 'AcosOp', 'float_only' : True},
                    {"name" : '11', 'struct' : 'AtanOp', 'float_only' : True},
                    {"name" : '12', 'struct' : 'SinhOp', 'float_only' : True},
                    {"name" : '13', 'struct' : 'CoshOp', 'float_only' : True},
                    {"name" : '14', 'struct' : 'TanhOp', 'float_only' : True},
                    {"name" : '15', 'struct' : 'Exp2Op', 'float_only' : True},
                    {"name" : '16', 'struct' : 'Log2Op', 'float_only' : True},
                    {"name" : '17', 'struct' : 'Log10Op', 'float_only' : True},
                    {"name" : '18', 'struct' : 'Expm1Op', 'float_only' : True},
                    {"name" : '19', 'struct' : 'Log1pOp', 'float_only' : True},
                    {"name" : '20', 'struct' : 'RsqrtOp', 'float_only' : True},
                    {"name" : '21', 'struct' : 'SigmoidOp', 'float_only' : True},
                    {"name" : '22', 'struct' : 'AsinhOp', 'float_only' : True},
                    {"name" : '23', 'struct' : 'AcoshOp', 'float_only' : True},
                    {"name" : '24', 'struct' : 'AtanhOp', 'float_only' : True},
                    {"name" : '25', 'struct' : 'Deg2radOp', 'float_only' : True},
                    {"name" : '26', 'struct' : 'Rad2degOp', 'float_only' : True},
                    {"name" : '27', 'struct' : 'FloorOp'},
                    {"name" : '28', 'struct' : 'CeilOp'},
                    {"name" : '29', 'struct' : 'TruncOp'},
                    {"name" : '30', 'struct' : 'RoundOp'},
                    {"name" : '31', 'struct' : 'FracOp'},
                    {"name" : '32', 'struct' : 'AbsOp'},
                    {"name" : '33', 'struct' : 'SignOp'},
                    {"name" : '34', 'struct' : 'ErfOp', 'float_only' : True},
                    {"name" : '35', 'struct' : 'ErfinvOp', 'float_only' : True},
                    {"name" : '36', 'struct' : 'SincOp', 'float_only' : True},
                    {"name" : '37', 'struct' : 'EntrOp', 'float_only' : True},
                    {"name" : '38', 'struct' : 'LgammaOp', 'float_only' : True},
                    {"name" : '39', 'struct' : 'DigammaOp', 'float_only' : True},
                    {"name" : '40', 'struct' : 'I0Op', 'float_only' : True},
                    {"name" : '41', 'struct' : 'I1Op', 'float_only' : True}
                ]}
    },
    {
        'name' : 'cast.slang.j2',
        'dtypes' : DTYPES_SUPERSET,
        'kwargs' : {}
    },
    {
        'name' : 'compareop.slang.j2',
        'dtypes' : DTYPES,
        'kwargs' : {'OPERATIONS' : [
                    {"name" : '0', 'struct' : 'EqOp'},
                    {"name" : '1', 'struct' : 'NeOp'},
                    {"name" : '2', 'struct' : 'LtOp'},
                    {"name" : '3', 'struct' : 'LeOp'},
                    {"name" : '4', 'struct' : 'GtOp'},
                    {"name" : '5', 'struct' : 'GeOp'},
                    {"name" : '6', 'struct' : 'LogicalAndOp'},
                    {"name" : '7', 'struct' : 'LogicalOrOp'},
                    {"name" : '8', 'struct' : 'LogicalXorOp'}
                ]}
    },
    {
        'name' : 'fill.slang.j2',
        'dtypes' : DTYPES,
        'kwargs' : {}
    },
    {
        'name' : 'where.slang.j2',
        'dtypes' : DTYPES,
        'kwargs' : {}
    },
    {
        'name' : 'reduce.slang.j2',
        'dtypes' : DTYPES,
        'kwargs' : {'OPERATIONS' : [
                    {"name" : '0', 'struct' : 'SumReduceOp'},
                    {"name" : '1', 'struct' : 'AmaxReduceOp'},
                    {"name" : '2', 'struct' : 'AminReduceOp'},
                    {"name" : '3', 'struct' : 'ProdReduceOp'}
                ]}
    },
    {
        'name' : 'reduce_subgroup.slang.j2',
        'dtypes' : [f for f in FLOATS if f[0]['bytes'] <= 4],
        'kwargs' : {'OPERATIONS' : [
                    {"name" : '0', 'struct' : 'SumReduceOp'},
                    {"name" : '1', 'struct' : 'AmaxReduceOp'},
                    {"name" : '2', 'struct' : 'AminReduceOp'}
                ]}
    },
    {
        'name' : 'nllloss.slang.j2',
        'dtypes' : FLOATS,
        'kwargs' : {}
    },
    {
        'name' : 'scan.slang.j2',
        'dtypes' : DTYPES,
        'kwargs' : {'OPERATIONS' : [
                    {"name" : '0', 'struct' : 'SumScanOp'},
                    {"name" : '1', 'struct' : 'ProdScanOp'}
                ]}
    },
    {
        'name' : 'arg_reduce.slang.j2',
        'dtypes' : DTYPES,
        'kwargs' : {}
    },
    {
        'name' : 'scan_arg.slang.j2',
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

