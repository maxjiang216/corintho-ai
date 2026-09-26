# 13 — The solver's transposition table

2026-09-26. Commits `5323b2d` (16-byte entries), `070804a` (4-way buckets),
`01c4b6c` (prefetch). Data: `data/solver/bench-13-buckets-p30.tsv`.

The table probes were ~10% of instructions but ~98% of the L1 misses.
Measured on the deeper sets (6 threads, summed solve time), where the table
matters most; gate: identical results.

| change | P <= 27 | P <= 30 | nodes |
|---|---|---|---|
| before (32-byte entries, direct-mapped) | 5.47 s | 74.7 s | |
| 16-byte entries (board + one packed word: side, reserves, score, bound, move, P, 28-bit epoch) | 5.18 s | 66.8 s | identical |
| 4-way buckets, entries interleaved (no SIMD probe) | 5.69 s | 67.5 s | 0.99x / 0.94x, 7 more solved |
| 4-way buckets, 4 boards then 4 metas, one AVX2 compare | 5.31 s | 65.8 s | same |
| prefetch the first 0 / 2 / 4 / 8 / all children's lines | | 66.1 / 66.0 / 61.8 / 58.7 / **56.5 s** | same |
| final | **4.69 s** | **~56.3 s** | |

Net: P <= 27 -14%, **P <= 30 -25%** and 7 more positions solved within
the 5M cap; P <= 24 one thread pinned 0.24 s.

Replacement: same position, else a stale entry, else the entry with the
smallest P (nearest the end of the game, cheapest to recompute).

Note: with several threads, node counts now vary by a few positions between
runs of the same binary (stale entries from a thread's earlier solves take
part in replacement, and positions go to threads in varying order).
Results are exact regardless; the node-count gate compares totals only
approximately.
