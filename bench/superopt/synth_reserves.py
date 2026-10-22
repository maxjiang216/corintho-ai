"""Synthesize the reserve sum of a key's `rest` word (solver.cpp)

Game::key packs side to move in bit 0 and reserve i (0-4) in bits
4 + 4i, i = 0..5. The solver adds the six reserves at every store and
cutoff, with a loop. Spec: that sum, exactly, for every valid `rest`.

    python3 synth_reserves.py [max_n]
"""

from __future__ import annotations

import sys

import synth
import z3


def nibble(x, i):
    return z3.Extract(4 * i + 3, 4 * i, x)


def valid_rest(x):
    """Bit 0 any; bits 1-3 clear; reserves 1-6 (nibbles) at most 4; the rest
    clear"""
    return z3.And(
        z3.Extract(3, 1, x) == 0,
        z3.Extract(63, 28, x) == 0,
        *[z3.ULE(nibble(x, i), 4) for i in range(1, 7)],
    )


def reserves_spec(x):
    if not isinstance(x, z3.ExprRef):
        return sum((x >> (4 + 4 * i)) & 0xF for i in range(6))
    total = z3.BitVecVal(0, 64)
    for i in range(1, 7):
        total = total + z3.ZeroExt(60, nibble(x, i))
    return total


def main():
    max_n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    seeds = (0, 1, 0x4444440, 0x4444441, 0x1234300, 0x0432101)
    prog, _ = synth.synthesize(
        reserves_spec,
        max_n=max_n,
        ops=synth.OPS,
        seeds=seeds,
        pre=valid_rest,
    )
    print(synth.render(prog, "rest") if prog else f"none within {max_n}")


if __name__ == "__main__":
    main()
