"""One compact supervised dataset from self-play sample directories.

    python arch/dataset.py OUT.npz runs/full-2/gen_{1,2,3,4}/samples

Each samples directory stores every position 8 times, as its 8 board
symmetries (selfplayer.cpp writeSamples). This keeps the first copy (the
identity) and checks that the other 7 are exactly its symmetric images, so
training can apply symmetries on the fly instead. It adds the legal-move mask
and line spaces, and checks that every move the search visited is legal.

Arrays: states uint8 [n, 70] (4x the network input), values float32 [n],
policies float16 [n, 96], legal packed uint8 [n, 12], lines uint16 [n, 3],
source int8 [n] (index of the samples directory).
"""
import sys

import numpy as np
from features import features, symmetries

CHUNK = 100_000  # positions per check


def load(d, sym_state, sym_move):
    states = np.load(f"{d}/states.npy", mmap_mode="r")
    values = np.load(f"{d}/values.npy", mmap_mode="r").reshape(-1)
    policies = np.load(f"{d}/policies.npy", mmap_mode="r")
    rows = states.shape[0]
    assert rows % 8 == 0
    n = rows // 8
    out_s = np.empty((n, 70), np.uint8)
    out_v = np.empty(n, np.float32)
    out_p = np.empty((n, 96), np.float16)
    bad_sym = 0
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        s = np.asarray(states[8 * a : 8 * b]).reshape(b - a, 8, 70)
        v = np.asarray(values[8 * a : 8 * b]).reshape(b - a, 8)
        p = np.asarray(policies[8 * a : 8 * b]).reshape(b - a, 8, 96)
        base_s, base_p = s[:, 0], p[:, 0]
        for k in range(8):
            bad_sym += int(
                (s[:, k] != base_s[:, sym_state[k]]).any(1).sum()
                + (p[:, k] != base_p[:, sym_move[k]]).any(1).sum()
                + (v[:, k] != v[:, 0]).sum()
            )
        q = np.rint(base_s * 4)
        assert np.array_equal(q / 4, base_s), "state not a multiple of 0.25"
        out_s[a:b] = q.astype(np.uint8)
        out_v[a:b] = v[:, 0]
        out_p[a:b] = base_p.astype(np.float16)
    first = np.ascontiguousarray(states[::8], dtype=np.float32)
    legal, lines = features(first)
    visited_illegal = int(((out_p > 0) & ~legal).any(1).sum())
    no_legal = int((~legal.any(1)).sum())
    print(
        f"{d}: {n} positions; symmetry mismatches {bad_sym}; positions with "
        f"visits on an illegal move {visited_illegal}; with no legal move "
        f"{no_legal}; legal moves mean {legal.sum(1).mean():.1f}; "
        f"positions with a line {(lines.any(1)).mean():.3f}",
        flush=True,
    )
    assert bad_sym == 0 and visited_illegal == 0 and no_legal == 0
    return out_s, out_v, out_p, np.packbits(legal, 1, "little"), lines


def main():
    out, dirs = sys.argv[1], sys.argv[2:]
    sym_state, sym_move, _ = symmetries()
    parts = [load(d, sym_state, sym_move) for d in dirs]
    source = np.concatenate(
        [np.full(len(p[0]), i, np.int8) for i, p in enumerate(parts)]
    )
    names = ("states", "values", "policies", "legal", "lines")
    arrays = {
        k: np.concatenate([p[i] for p in parts]) for i, k in enumerate(names)
    }
    np.savez(out, source=source, dirs=np.array(dirs), **arrays)
    print(f"{out}: {len(source)} positions")


if __name__ == "__main__":
    main()
