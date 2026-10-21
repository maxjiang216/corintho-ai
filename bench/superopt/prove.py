"""Prove the nibble-domain line detectors equal the current ones

Each check asks z3 for a board word on which the two disagree; UNSAT is a
proof over all 2^64 words (including ones no game reaches, so the proofs
assume nothing about legality).

    python3 prove.py
"""

from __future__ import annotations

import sys
import time

import kernels as k
import z3


def prove(name, claim):
    s = z3.SolverFor("QF_BV")
    s.add(z3.Not(claim))
    t0 = time.time()
    r = s.check()
    ok = r == z3.unsat
    print(f"{'PROVED' if ok else 'FAILED'}  {name}  ({time.time() - t0:.2f}s)")
    if not ok and r == z3.sat:
        print(f"  counterexample: {s.model()}")
    return ok


def main():
    b = z3.BitVec("b", 64)
    ok = True

    # Nibble tops match computeSpaceInfo's top planes, bit for bit
    t = k.tops_nibbles(b)
    p = k.top_planes(b)
    ok &= prove(
        "tops_nibbles == top_plane[0..2]",
        z3.And(
            [
                z3.Extract(4 * s + ty, 4 * s + ty, t)
                == z3.Extract(s, s, p[ty])
                for s in range(16)
                for ty in range(3)
            ]
        ),
    )

    ok &= prove(
        "has_line_nibbles == boardHasLine",
        k.has_line_nibbles(b) == k.has_line_current(b),
    )

    # findLines: the same runs, type by type and direction by direction
    run3, run4 = k.find_lines_current(b)
    claims = []
    for d in range(4):
        r3, r4 = k.runs_nibbles(t, d)
        # nothing outside the bits the decoder reads
        claims.append(r3 & ~k.nibble_spread(0xFFFF) == 0)
        claims.append(r4 & ~k.nibble_spread(0xFFFF) == 0)
        for s in range(16):
            for ty in range(3):
                claims.append(
                    z3.Extract(4 * s + ty, 4 * s + ty, r3)
                    == z3.Extract(s, s, run3[ty][d])
                )
                claims.append(
                    z3.Extract(4 * s + ty, 4 * s + ty, r4)
                    == z3.Extract(s, s, run4[ty][d])
                )
    ok &= prove("runs_nibbles == findLines run3/run4", z3.And(claims))

    # Game::key and packRest, for every reserve count 0-4
    pieces = [z3.BitVec(f"p{i}", 8) for i in range(6)]
    to_play = z3.BitVec("to_play", 1)
    in_range = z3.And([z3.ULE(p, 4) for p in pieces])
    current = k.key_rest_current(pieces, to_play)
    ok &= prove(
        "key rest: pext == loop",
        z3.Implies(in_range, k.key_rest_pext(pieces, to_play) == current),
    )
    ok &= prove(
        "key rest: portable == loop",
        z3.Implies(in_range, k.key_rest_portable(pieces, to_play) == current),
    )
    ok &= prove(
        "packRest: pext == loop",
        z3.Implies(
            in_range,
            k.pack_rest_pext(current) == k.pack_rest_current(current),
        ),
    )
    ok &= prove(
        "reserve sum: multiply == loop",
        z3.Implies(
            in_range,
            k.rest_reserves_mul(current) == k.rest_reserves_current(current),
        ),
    )
    ok &= prove(
        "horizon reserves: multiply == loop",
        z3.Implies(
            in_range,
            k.horizon_reserves_mul(pieces)
            == k.horizon_reserves_current(pieces),
        ),
    )
    # lineBreakers: bit `id` of moves_from(tops) is tops' bit at the move's
    # source, for every stack move, so ANDing with a shape's extend moves
    # gives what the per-move loop built
    tops16 = z3.BitVec("tops", 16)
    tops = z3.ZeroExt(48, tops16)
    m = k.moves_from(tops)
    src = k.move_from_squares()
    ok &= prove(
        "moves_from == per-move source test",
        z3.And(
            [
                z3.Extract(i, i, m) == z3.Extract(src[i], src[i], tops16)
                for i in range(48)
            ]
            + [z3.Extract(63, 48, m) == 0]
        ),
    )
    ok &= prove(
        "moves_from: portable == pext",
        k.moves_from_portable(tops) == m,
    )

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
