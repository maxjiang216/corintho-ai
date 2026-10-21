"""Synthesize topNibbles (game.cpp) and prove it minimal

Spec: for each space's nibble (base, column, capital, frozen bits), bit k
of the result is set when the top piece is type k: base & ~column &
~capital, column & ~capital, capital. Bit 3 is don't-care by default
(every caller masks it off); --clean-bit3 asks for it to be zero too.

    python3 synth_tops.py [--clean-bit3]
"""

from __future__ import annotations

import sys

import kernels as k
import synth

M1 = k.M1


def tops_spec(b):
    b0 = b & M1
    b1 = k.shr(b, 1) & M1
    b2 = k.shr(b, 2) & M1
    return (b0 & ~b1 & ~b2) | ((b1 & ~b2) << 1) | (b2 << 2)


def main():
    care = synth.MASK if "--clean-bit3" in sys.argv else 0x7777777777777777
    prog, _ = synth.synthesize(tops_spec, care=care, max_n=7)
    print(synth.render(prog) if prog else "none within 7 instructions")


if __name__ == "__main__":
    main()
