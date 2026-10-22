"""Write a random small network in small_net.h's format, for benchmarks

Engine speed and fixed-depth node counts (cpp/ab_bench.cpp) do not need a
trained network. Strength does.

    python arch/random_small_net.py OUT.net [hidden=256] [bias=0.2] [seed=1]

A negative bias gives sparser activations.
"""

import struct
import sys

import numpy as np


def main():
    out = sys.argv[1]
    h = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    bias = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    rng = np.random.default_rng(int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    w1 = rng.normal(0, 0.08, (70, h)).astype(np.float32)
    b1 = rng.normal(bias, 0.05, h).astype(np.float32)
    w2 = rng.normal(0, 0.08, (32, h)).astype(np.float32)
    b2 = rng.normal(0, 0.05, 32).astype(np.float32)
    w3 = rng.normal(0, 0.3, 32).astype(np.float32)
    with open(out, "wb") as f:
        f.write(struct.pack("<i", h))
        for a in (w1, b1, w2, b2, w3):
            f.write(a.tobytes())
        f.write(struct.pack("<f", 0.0))


if __name__ == "__main__":
    main()
