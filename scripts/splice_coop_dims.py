"""
Points the rows and columns of a kernel's cooperative matrices at a
specialization constant, which Slang can't emit itself: slangc requires CoopMat
dimensions to be compile-time constants, though SPIR-V allows a spec constant
there. The shader declares every CoopMat with a placeholder dimension, and this
rewrites the compiled SPIR-V, run by src/shaders/CMakeLists.txt after slangc for
the kernels scripts/config.py lists with coop_dims:

    splice_coop_dims.py in.spv out.spv --spec-id 18 --placeholder 1009 [--spec-id 22 --placeholder 1013 ...]

Fails when the placeholder is used anywhere but a cooperative matrix type.
"""

import argparse
import struct
from pathlib import Path


OP_SOURCE = 3
OP_SOURCE_EXTENSION = 4
OP_NAME = 5
OP_MEMBER_NAME = 6
OP_STRING = 7
OP_LINE = 8
OP_EXTENSION = 10
OP_EXT_INST_IMPORT = 11
OP_EXT_INST = 12
OP_MEMORY_MODEL = 14
OP_ENTRY_POINT = 15
OP_EXECUTION_MODE = 16
OP_CAPABILITY = 17
OP_TYPE_INT = 21
OP_TYPE_FLOAT = 22
OP_TYPE_VECTOR = 23
OP_TYPE_MATRIX = 24
OP_TYPE_POINTER = 32
OP_TYPE_FORWARD_POINTER = 39
OP_CONSTANT_TRUE = 41
OP_CONSTANT_FALSE = 42
OP_CONSTANT = 43
OP_SPEC_CONSTANT_TRUE = 48
OP_SPEC_CONSTANT_FALSE = 49
OP_SPEC_CONSTANT = 50
OP_SPEC_CONSTANT_OP = 52
OP_VARIABLE = 59
OP_LOAD = 61
OP_STORE = 62
OP_COPY_MEMORY = 63
OP_DECORATE = 71
OP_MEMBER_DECORATE = 72
OP_VECTOR_SHUFFLE = 79
OP_COMPOSITE_EXTRACT = 81
OP_COMPOSITE_INSERT = 82
OP_LOOP_MERGE = 246
OP_SELECTION_MERGE = 247
OP_BRANCH_CONDITIONAL = 250
OP_SWITCH = 251
OP_NO_LINE = 317
OP_TYPE_COOPERATIVE_MATRIX = 4456
OP_COOPERATIVE_MATRIX_LOAD = 4457
OP_COOPERATIVE_MATRIX_STORE = 4458
DECORATION_SPEC_ID = 1

# declarations, annotations and debug info: no operand is a use of an id
NO_ID_OPERANDS = {
    OP_SOURCE, OP_SOURCE_EXTENSION, OP_NAME, OP_MEMBER_NAME, OP_STRING, OP_LINE, OP_EXTENSION, OP_EXT_INST_IMPORT,
    OP_MEMORY_MODEL, OP_ENTRY_POINT, OP_EXECUTION_MODE, OP_CAPABILITY, OP_TYPE_INT, OP_TYPE_FLOAT, OP_TYPE_VECTOR,
    OP_TYPE_MATRIX, OP_TYPE_POINTER, OP_TYPE_FORWARD_POINTER, OP_CONSTANT_TRUE, OP_CONSTANT_FALSE, OP_CONSTANT,
    OP_SPEC_CONSTANT_TRUE, OP_SPEC_CONSTANT_FALSE, OP_SPEC_CONSTANT, OP_DECORATE, OP_MEMBER_DECORATE, OP_NO_LINE,
    OP_TYPE_COOPERATIVE_MATRIX,
}

# the word offsets that are ids, for instructions that also hold literals
ID_OFFSETS = {
    OP_EXT_INST: lambda count: [1, 2, 3] + list(range(5, count)),
    OP_SPEC_CONSTANT_OP: lambda count: [1, 2] + list(range(4, count)),
    OP_VARIABLE: lambda count: [1, 2] + list(range(4, count)),
    OP_LOAD: lambda count: [1, 2, 3],
    OP_STORE: lambda count: [1, 2],
    OP_COPY_MEMORY: lambda count: [1, 2],
    OP_VECTOR_SHUFFLE: lambda count: [1, 2, 3, 4],
    OP_COMPOSITE_EXTRACT: lambda count: [1, 2, 3],
    OP_COMPOSITE_INSERT: lambda count: [1, 2, 3, 4],
    OP_LOOP_MERGE: lambda count: [1, 2],
    OP_SELECTION_MERGE: lambda count: [1],
    OP_BRANCH_CONDITIONAL: lambda count: [1, 2, 3],
    OP_SWITCH: lambda count: [1, 2] + list(range(4, count, 2)),
    OP_COOPERATIVE_MATRIX_LOAD: lambda count: [1, 2, 3, 4, 5],
    OP_COOPERATIVE_MATRIX_STORE: lambda count: [1, 2, 3, 4],
}


def id_operands(words, start, op, count):
    """The words of an instruction that use an id: every word after the opcode, unless ID_OFFSETS says otherwise."""
    if op in NO_ID_OPERANDS:
        return []
    offsets = ID_OFFSETS.get(op, lambda count: range(1, count))(count)
    return [words[start + i] for i in offsets if i < count]


def instructions(words):
    """Yields (start, opcode, word count) for each instruction after the 5-word header."""
    i = 5
    while i < len(words):
        count = words[i] >> 16
        if count == 0:
            raise ValueError(f'malformed SPIR-V instruction at word {i}')
        yield i, words[i] & 0xFFFF, count
        i += count


def splice_coop_dims(words, spec_id, placeholder, name):
    """
    Points the rows and columns of every CoopMat declared with `placeholder` at
    the specialization constant `spec_id`, which Slang can't do itself. Fails
    when the placeholder is used anywhere else.
    """
    words = list(words)
    spec = None
    placeholders = set()
    for start, op, _ in instructions(words):
        if op == OP_DECORATE and words[start + 2] == DECORATION_SPEC_ID and words[start + 3] == spec_id:
            spec = words[start + 1]
        if op == OP_CONSTANT and words[start + 3] == placeholder:
            placeholders.add(words[start + 2])
        if op == OP_CONSTANT and words[start + 3] not in (0, placeholder) and words[start + 3] % placeholder == 0:
            raise ValueError(f'{name}: a constant of {words[start + 3]} suggests the placeholder {placeholder} was folded into arithmetic')
    if spec is None:
        raise ValueError(f'{name}: no specialization constant with SpecId {spec_id}')

    spliced = 0
    first_matrix = None
    for start, op, _ in instructions(words):
        if op != OP_TYPE_COOPERATIVE_MATRIX:
            continue
        first_matrix = start if first_matrix is None else first_matrix
        for operand in (start + 4, start + 5): # rows, columns
            if words[operand] in placeholders:
                words[operand] = spec
                spliced += 1
    if spliced == 0:
        raise ValueError(f'{name}: no cooperative matrix uses the placeholder dimension {placeholder}')

    for start, op, count in instructions(words):
        if any(word in placeholders for word in id_operands(words, start, op, count)):
            raise ValueError(f'{name}: the placeholder dimension {placeholder} is used outside a cooperative matrix type (opcode {op})')

    # an id must be defined before its first use
    for start, op, count in instructions(words):
        if op == OP_SPEC_CONSTANT and words[start + 2] == spec:
            if start > first_matrix:
                definition = words[start:start + count]
                del words[start:start + count]
                words[first_matrix:first_matrix] = definition
            break
    return words


def main(args):
    data = args.input.read_bytes()
    words = struct.unpack(f'<{len(data) // 4}I', data)
    if len(args.spec_id) != len(args.placeholder):
        raise ValueError(f'{args.input.name}: every --spec-id needs a --placeholder')
    for spec_id, placeholder in zip(args.spec_id, args.placeholder):
        words = splice_coop_dims(words, spec_id, placeholder, args.input.name)
    args.output.write_bytes(struct.pack(f'<{len(words)}I', *words))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Splice a specialization constant into cooperative matrix dimensions.')
    parser.add_argument('input', type=Path, help='SPIR-V from slangc')
    parser.add_argument('output', type=Path, help='spliced SPIR-V')
    parser.add_argument('--spec-id', type=int, action='append', required=True, help='SpecId of the dimension constant (repeatable)')
    parser.add_argument('--placeholder', type=int, action='append', required=True, help='the literal dimension the shader declares, one per --spec-id')
    main(parser.parse_args())
