# 07 — Repeated evaluations, and an evaluation cache (parked)

2026-09-26. Commits `ba8cd5a`, `4d91ba7`, `a9512ad`, `90d84af`.

## How often positions repeat

`corintho_play` with `CORINTHO_DUP_DUMP=FILE` records every row sent to the
network (game, call, a hash, the hash least over the 8 symmetries, the
row's horizon P and its search root's). 1000 self-play games, 36.4M rows:

| | unique | repeats, all games | repeats, same game |
|---|---|---|---|
| exact | 18.2M | 50.0% | 32.6% |
| up to symmetry | 15.9M | 56.5% | 37.7% |

Repeats are spread out: the top 10k positions are only 6.8% of rows.
Within-game repeats are transpositions plus positions evaluated in parts of
the tree discarded when a move is played. The AZ search is a plain tree:
no transposition handling.

## The cache

`--cache L` (eval_cache.h): exact keys (board bytes shuffled into the least
of 8 frames with pshufb, packed; plus reserves), 4-way sets, policy stored
in the canonical frame and rotated back on hits. Verified: hits stored
from the same frame are bit-exact. Cross-frame hits return the network's
evaluation of a symmetric copy; the network is only approximately
invariant (legal-masked: value correlation 0.97-0.99, same top move 74-82%,
policy TV 0.11-0.16; without masking the illegal outputs, which masked
training leaves untrained, the comparison is meaningless).

Timing (4000 games, 14 threads): none of the designs paid. Parallel
lookups on the main thread: GPU wait 17.5 -> 3.4 s but ~9 s of cache work
on the critical path (wall 57 -> 58-62 s). One worker thread for all
network calls: 470 ns per row (~1 KB of memory traffic per row), 67 s,
wall 49 -> 90 s. With the GPU wait only ~20% of self-play (512x4, 14
threads), the cache cannot pay. Parked; `--cache` off by default. The
worker-thread driver is in the source; the loop runs on a build from
before it.

Unverified observation, check pending: with the cache off, the
worker-thread binary ran 49.1 / 49.2 s (GPU wait 9.7 s) where the older
binary (a std::async thread per network call, so the two groups' calls
could overlap on the GPU) had run 57-60 s (wait ~17.5 s), but on a
different network (gen 5 vs gen 4). If it holds with the same network,
serializing the groups' GPU calls is itself a ~15% self-play speed-up. The
A/B (same network, both binaries, cache off) was interrupted.

Better route: merging transpositions inside the search (a graph instead of
a tree; KataGo's graph search) avoids the repeats at the source and shares
statistics; Corintho has no cycles (P falls every move). Parked by the
developer as a large change.

Also parked (the developer's idea): the spread of the network's
evaluations over symmetric copies of one position measures part of the
model's own noise on that position (test-time-augmentation variance); the
data to study it (networks, compact datasets, solver-proven positions) is
all kept.
