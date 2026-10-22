"""Can the line test use fewer shifted ANDs? A sketch query

boardHasLine is OR over the four directions of (t & t >> a & t >> b & C),
with t the nibble tops word. Holding that shape fixed and leaving every
shift amount and mask symbolic, ask whether K terms suffice for K < 4. An
UNSAT answer proves no choice of shifts and masks works, so four is minimal
for this shape. CEGIS as in synth.py: solve on examples, verify on all
2^64 boards, add the counterexample.

    python3 sketch.py [K ...]
"""

from __future__ import annotations

import sys
import time

import kernels as k
import z3


def term_sum(t, params):
    any_ = z3.BitVecVal(0, 64)
    for a, b, c in params:
        any_ = any_ | (t & z3.LShR(t, a) & z3.LShR(t, b) & c)
    return any_


def solve(n_terms: int, timeout_ms: int = 1_800_000):
    params = [
        (
            z3.BitVec(f"a{i}", 64),
            z3.BitVec(f"b{i}", 64),
            z3.BitVec(f"c{i}", 64),
        )
        for i in range(n_terms)
    ]
    s = z3.SolverFor("QF_BV")
    s.set("timeout", timeout_ms)
    for i, (a, b, _) in enumerate(params):
        s.add(z3.UGE(a, 1), z3.ULT(a, b), z3.ULE(b, 63))
        if i > 0:  # terms in increasing order of a: symmetry breaking
            s.add(z3.ULE(params[i - 1][0], a))
    examples = [0, (1 << 64) - 1, 0x0000000000000111, 0x1111000000000000]
    t0 = time.time()
    while True:
        s.push()
        for ex in examples:
            bv = z3.BitVecVal(ex, 64)
            s.add(
                (term_sum(k.tops_nibbles(bv), params) != 0)
                == z3.simplify(k.has_line_current(bv))
            )
        r = s.check()
        if r != z3.sat:
            s.pop()
            return r, None, len(examples), time.time() - t0
        m = s.model()
        concrete = [
            tuple(m.eval(v, model_completion=True).as_long() for v in p)
            for p in params
        ]
        s.pop()
        x = z3.BitVec("x", 64)
        v = z3.SolverFor("QF_BV")
        v.add(
            (term_sum(k.tops_nibbles(x), concrete) != 0)
            != k.has_line_current(x)
        )
        if v.check() == z3.unsat:
            return z3.sat, concrete, len(examples), time.time() - t0
        examples.append(v.model().eval(x, model_completion=True).as_long())


def main():
    for n in [int(a) for a in sys.argv[1:]] or [3, 4]:
        r, params, n_ex, secs = solve(n)
        print(f"K={n}: {r} after {n_ex} examples, {secs:.1f}s")
        for a, b, c in params or []:
            print(f"  t & t >> {a} & t >> {b} & {c:#018x}")


if __name__ == "__main__":
    main()
