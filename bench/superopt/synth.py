"""A small SAT-based superoptimizer for 64-bit bitboard kernels.

Counterexample-guided inductive synthesis (CEGIS) over straight-line
programs, with z3's bit-vector theory (bit-blasted to SAT) doing the search:

  1. Synthesis: find a program of N instructions -- opcodes, operands, shift
     amounts and 64-bit constants all symbolic -- that matches the spec on a
     small set of example inputs.
  2. Verification: ask the solver for any input on which that concrete
     program differs from the spec. UNSAT means it is correct on all 2^64
     inputs; otherwise the counterexample joins the examples and we go back
     to 1.

Trying N = 1, 2, ... in turn, the first N that succeeds is a proven minimum
for this instruction set: an UNSAT synthesis query at N - 1 says no program
that short exists, however clever. That is what makes it "brute force by
SAT" rather than a heuristic search.

The instruction set is what x86-64 with BMI1 executes in one instruction:
and, or, xor, andn (~a & b), not, shifts by a constant, and and/or/andn with
a 64-bit constant (a mov-imm64 plus the op; constants are hoisted out of
the hot loops, so they are counted as free). Worklog
2026-09-25-nn-architectures, entry 24.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import z3

W = 64
MASK = (1 << W) - 1

# Opcodes. Binary ops take two earlier values; *_c ops take one value and a
# symbolic constant; shifts take one value and a symbolic amount.
OPS = ["and", "or", "xor", "andn", "not", "shr", "shl", "and_c", "or_c",
       "andn_c"]


@dataclass
class Insn:
    op: str
    a: int
    b: int = 0
    k: int = 0  # shift amount or constant

    def render(self, names: list[str]) -> str:
        a, b = names[self.a], names[self.b] if self.b < len(names) else "?"
        return {
            "and": f"{a} & {b}",
            "or": f"{a} | {b}",
            "xor": f"{a} ^ {b}",
            "andn": f"~{a} & {b}",
            "not": f"~{a}",
            "shr": f"{a} >> {self.k}",
            "shl": f"{a} << {self.k}",
            "and_c": f"{a} & {self.k:#018x}",
            "or_c": f"{a} | {self.k:#018x}",
            "andn_c": f"~{a} & {self.k:#018x}",
        }[self.op]


def run(prog: list[Insn], x):
    """Evaluate a concrete program on an int or a z3 bit-vector"""
    vals = [x]
    is_z3 = isinstance(x, z3.ExprRef)
    for i in prog:
        a, b = vals[i.a], vals[i.b]
        if i.op == "and":
            v = a & b
        elif i.op == "or":
            v = a | b
        elif i.op == "xor":
            v = a ^ b
        elif i.op == "andn":
            v = ~a & b
        elif i.op == "not":
            v = ~a
        elif i.op == "shr":
            v = z3.LShR(a, i.k) if is_z3 else a >> i.k
        elif i.op == "shl":
            v = a << i.k
        elif i.op == "and_c":
            v = a & i.k
        elif i.op == "or_c":
            v = a | i.k
        else:
            v = ~a & i.k
        if not is_z3:
            v &= MASK
        vals.append(v)
    return vals[-1]


def _symbolic(n: int, examples: list[int], spec, care: int, ops: list[str],
              timeout_ms: int):
    """One synthesis query: some n-instruction program fits every example"""
    s = z3.SolverFor("QF_BV")
    s.set("timeout", timeout_ms)
    opc = [z3.BitVec(f"op{i}", 4) for i in range(n)]
    sa = [z3.BitVec(f"a{i}", 4) for i in range(n)]
    sb = [z3.BitVec(f"b{i}", 4) for i in range(n)]
    sk = [z3.BitVec(f"k{i}", W) for i in range(n)]
    allowed = [OPS.index(o) for o in ops]
    for i in range(n):
        s.add(z3.Or([opc[i] == o for o in allowed]))
        s.add(z3.ULE(sa[i], i), z3.ULE(sb[i], i))  # operands: earlier values
        # Shifts by 1..63 only
        is_shift = z3.Or(opc[i] == OPS.index("shr"), opc[i] == OPS.index("shl"))
        s.add(z3.Implies(is_shift, z3.And(z3.UGE(sk[i], 1), z3.ULE(sk[i], 63))))
        # Symmetry breaking: commutative ops take a <= b, and the last value
        # must be used (no dead tail); unused operands pinned to 0
        comm = z3.Or([opc[i] == OPS.index(o) for o in ("and", "or", "xor")])
        s.add(z3.Implies(comm, z3.ULT(sa[i], sb[i])))
        unary = z3.Not(z3.Or([opc[i] == OPS.index(o)
                              for o in ("and", "or", "xor", "andn")]))
        s.add(z3.Implies(unary, sb[i] == 0))
    # Every intermediate is used by a later instruction (otherwise a shorter
    # program exists)
    for j in range(1, n):
        s.add(z3.Or([z3.Or(sa[i] == j, sb[i] == j) for i in range(j, n)]))

    def sel(vals, idx):
        e = vals[0]
        for j in range(1, len(vals)):
            e = z3.If(idx == j, vals[j], e)
        return e

    for ex in examples:
        vals = [z3.BitVecVal(ex, W)]
        for i in range(n):
            a, b = sel(vals, sa[i]), sel(vals, sb[i])
            k = sk[i]
            v = z3.If(opc[i] == OPS.index("and"), a & b,
                z3.If(opc[i] == OPS.index("or"), a | b,
                z3.If(opc[i] == OPS.index("xor"), a ^ b,
                z3.If(opc[i] == OPS.index("andn"), ~a & b,
                z3.If(opc[i] == OPS.index("not"), ~a,
                z3.If(opc[i] == OPS.index("shr"), z3.LShR(a, k),
                z3.If(opc[i] == OPS.index("shl"), a << k,
                z3.If(opc[i] == OPS.index("and_c"), a & k,
                z3.If(opc[i] == OPS.index("or_c"), a | k,
                      ~a & k)))))))))
            vals.append(v)
        s.add((vals[-1] ^ spec(ex)) & care == 0)
    r = s.check()
    if r != z3.sat:
        return r, None
    m = s.model()
    prog = []
    for i in range(n):
        ev = lambda v: m.eval(v, model_completion=True).as_long()
        prog.append(Insn(OPS[ev(opc[i])], ev(sa[i]), ev(sb[i]), ev(sk[i])))
    return r, prog


def verify(prog: list[Insn], spec, care: int):
    """None if prog equals spec on every input (under care), else a
    counterexample"""
    x = z3.BitVec("x", W)
    s = z3.SolverFor("QF_BV")
    s.add((run(prog, x) ^ spec(x)) & care != 0)
    if s.check() == z3.unsat:
        return None
    return s.model().eval(x, model_completion=True).as_long()


def synthesize(spec, care: int = MASK, max_n: int = 8, ops=None,
               seeds=(0, MASK, 0x0123456789ABCDEF, 0xFEDCBA9876543210),
               timeout_ms: int = 600_000, log=print):
    """The shortest program for spec, with a proof of minimality

    @param spec Python function of an int or a z3 bit-vector
    @param care Output bits that matter; others may be anything
    @return (program, list of (n, verdict, seconds)) where verdict is 'unsat'
        for each length proven impossible
    """
    ops = ops or OPS
    examples = list(seeds)
    trail = []
    for n in range(1, max_n + 1):
        t0 = time.time()
        while True:
            r, prog = _symbolic(n, examples, spec, care, ops, timeout_ms)
            if prog is None:
                trail.append((n, str(r), time.time() - t0))
                log(f"  n={n}: {r} ({time.time() - t0:.1f}s, "
                    f"{len(examples)} examples)")
                break
            cex = verify(prog, spec, care)
            if cex is None:
                trail.append((n, "found", time.time() - t0))
                log(f"  n={n}: found ({time.time() - t0:.1f}s, "
                    f"{len(examples)} examples)")
                return prog, trail
            examples.append(cex)
        if r != z3.unsat:
            return None, trail  # timed out: minimality unknown beyond here
    return None, trail


def render(prog: list[Insn], input_name: str = "b") -> str:
    names = [input_name] + [f"v{i + 1}" for i in range(len(prog))]
    return "\n".join(f"  v{i + 1} = {insn.render(names)}"
                     for i, insn in enumerate(prog))
