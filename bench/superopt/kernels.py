"""z3 models of the engine's line detection, current and nibble-domain

Each function mirrors a C++ routine in corintho_ai/cpp/src/game.cpp line
for line, so that prove.py can check the rewrite against the original over
all 2^64 board words. The board word holds four bits per space s at bits
4s..4s+3: base, column, capital, frozen.
"""

from __future__ import annotations

import z3

W = 64
M1 = 0x1111111111111111

# util.h
LINE_STRIDE = [1, 4, 5, 3]
RUN_START = [
    (0x3333, 0x1111),
    (0x00FF, 0x000F),
    (0x0033, 0x0001),
    (0x00CC, 0x0008),
]


def shr(x, k):
    return z3.LShR(x, k) if isinstance(x, z3.ExprRef) else x >> k


def pext_nibbles(b, bit):
    """_pext_u64(b, 0x1111... << bit): bit `bit` of every nibble, gathered
    into a 16-bit value (zero-extended to 64)"""
    bits = [z3.Extract(4 * s + bit, 4 * s + bit, b) for s in range(16)]
    return z3.ZeroExt(48, z3.Concat(*reversed(bits)))


def nibble_spread(m16: int, low_bits: int = 0x7) -> int:
    """A 16-bit space mask as a board-word mask: `low_bits` in each chosen
    space's nibble"""
    return sum(low_bits << (4 * s) for s in range(16) if (m16 >> s) & 1)


# --- Current code ----------------------------------------------------------


def top_planes(b):
    """computeSpaceInfo's top_plane[base, column, capital], 16-bit each"""
    base, col, cap = (pext_nibbles(b, k) for k in range(3))
    return [base & ~col & ~cap, col & ~cap, cap]


def has_line_current(b):
    """Game::boardHasLine: three planes in 16-bit lanes of one word"""
    p = top_planes(b)
    planes = p[0] | (p[1] << 16) | (p[2] << 32)
    lanes = lambda m: m * 0x0000000100010001
    any_ = z3.BitVecVal(0, W)
    for d in range(4):
        s = LINE_STRIDE[d]
        any_ = any_ | (
            planes
            & shr(planes, s)
            & shr(planes, 2 * s)
            & lanes(RUN_START[d][0])
        )
    return any_ != 0


def find_lines_current(b):
    """Game::findLines' run words: run3[type][d], run4[type][d] (16-bit)"""
    p = top_planes(b)
    run3 = [[None] * 4 for _ in range(3)]
    run4 = [[None] * 4 for _ in range(3)]
    for t in range(3):
        for d in range(4):
            s = LINE_STRIDE[d]
            three = p[t] & shr(p[t], s) & shr(p[t], 2 * s)
            run3[t][d] = three & RUN_START[d][0]
            run4[t][d] = three & shr(p[t], 3 * s) & RUN_START[d][1]
    return run3, run4


# --- Nibble domain ---------------------------------------------------------


def tops_nibbles(b):
    """Bit k of space s's nibble set when its top piece is type k. The
    frozen bit (3) is don't-care: every consumer masks it off."""
    # Found by synth.py, minimal at five instructions
    c = shr(b & 0x6666666666666666, 1)
    return b & ~c & ~shr(c, 1)


def runs_nibbles(t, d):
    """Runs of 3 and 4 in direction d for all three types at once: bit
    4c + k set when a run of type k starts at space c"""
    s = 4 * LINE_STRIDE[d]
    three = t & shr(t, s) & shr(t, 2 * s)
    r3 = three & nibble_spread(RUN_START[d][0])
    r4 = three & shr(t, 3 * s) & nibble_spread(RUN_START[d][1])
    return r3, r4


def has_line_nibbles(b):
    t = tops_nibbles(b)
    any_ = z3.BitVecVal(0, W)
    for d in range(4):
        s = 4 * LINE_STRIDE[d]
        any_ = any_ | (
            t & shr(t, s) & shr(t, 2 * s) & nibble_spread(RUN_START[d][0])
        )
    return any_ != 0


# --- Position key (Game::key, solver.cpp packRest) ------------------------


def pext(x, mask: int):
    """_pext_u64 with a constant mask: the masked bits of x, packed low"""
    bits = [z3.Extract(i, i, x) for i in range(64) if (mask >> i) & 1]
    return z3.ZeroExt(64 - len(bits), z3.Concat(*reversed(bits)))


def key_rest_current(pieces, to_play):
    """Game::key's rest: side to move in bit 0, reserve i in bits 4 + 4i"""
    rest = z3.ZeroExt(63, to_play)
    for i in range(6):
        rest = rest | (z3.ZeroExt(56, pieces[i]) << (4 + 4 * i))
    return rest


def reserve_bytes(pieces):
    """The six reserves as one memcpy'd little-endian word"""
    return z3.ZeroExt(16, z3.Concat(*reversed(pieces)))


def key_rest_pext(pieces, to_play):
    return (pext(reserve_bytes(pieces), 0x0F0F0F0F0F0F) << 4) | z3.ZeroExt(
        63, to_play
    )


def key_rest_portable(pieces, to_play):
    x = reserve_bytes(pieces)
    x = (x | shr(x, 4)) & 0x00FF00FF00FF
    x = (x | shr(x, 8)) & 0x0000FFFF0000FFFF
    x = (x | shr(x, 16)) & 0xFFFFFF
    return (x << 4) | z3.ZeroExt(63, to_play)


def pack_rest_current(rest):
    """solver.cpp packRest: side to move, then 3 bits per reserve"""
    out = rest & 1
    for i in range(6):
        out = out | ((shr(rest, 4 + 4 * i) & 7) << (1 + 3 * i))
    return out


def pack_rest_pext(rest):
    return pext(rest, 0x7777771)


def rest_reserves_current(rest):
    """The reserve-sum loop in Solver::store and recordCutoff"""
    total = z3.BitVecVal(0, 64)
    for i in range(6):
        total = total + (shr(rest, 4 + 4 * i) & 0xF)
    return total


def rest_reserves_mul(rest):
    pairs = (rest & 0x0F0F0F0) + (shr(rest, 4) & 0x0F0F0F0)
    return shr(pairs * 0x10101, 20) & 0xFF


def horizon_reserves_current(pieces):
    """Game::horizon's loop over the six reserve bytes"""
    total = z3.BitVecVal(0, 64)
    for p in pieces:
        total = total + z3.ZeroExt(56, p)
    return total


def horizon_reserves_mul(pieces):
    return shr(reserve_bytes(pieces) * 0x010101010101, 40) & 0xFF


# --- Line breaking by extension (Game::lineBreakers) -----------------------


def move_from_squares():
    """The source square of each stack-move ID 0-47, from the ID layout
    basicLegalMoves writes: right (row * 3 + col), down (12 + from), left
    (24 + row * 3 + col - 1), up (36 + from - 4)"""
    src = {}
    for row in range(4):
        for col in range(4):
            sq = 4 * row + col
            if col <= 2:
                src[row * 3 + col] = sq
            if sq <= 11:
                src[12 + sq] = sq
            if col >= 1:
                src[24 + row * 3 + col - 1] = sq
            if sq >= 4:
                src[36 + sq - 4] = sq
    return [src[i] for i in range(48)]


def moves_from(tops):
    """Every stack move whose source is set in the 16-bit `tops`, as move
    ID bits (zero-extended 64-bit input)"""
    return (
        pext(tops, 0x7777)
        | ((tops & 0x0FFF) << 12)
        | (pext(tops, 0xEEEE) << 24)
        | (shr(tops, 4) << 36)
    )


def moves_from_portable(tops):
    """movesFrom's non-BMI2 path"""
    right = (
        (tops & 0x7)
        | (shr(tops, 1) & 0x38)
        | (shr(tops, 2) & 0x1C0)
        | (shr(tops, 3) & 0xE00)
    )
    left = (
        (shr(tops, 1) & 0x7)
        | (shr(tops, 2) & 0x38)
        | (shr(tops, 3) & 0x1C0)
        | (shr(tops, 4) & 0xE00)
    )
    return (
        right | ((tops & 0x0FFF) << 12) | (left << 24) | (shr(tops, 4) << 36)
    )
