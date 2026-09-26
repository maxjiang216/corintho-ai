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

Driver check (was an unverified observation; `data/driver-groups-ab.txt`):
same network, 8000 games, 4000 in flight. The old driver (a thread per
network call, so the two groups' calls overlap on the GPU) took 112.8 /
122.2 s; the worker-thread driver (calls one at a time, cache off) 105.2 /
108.8 s: -7% / -11%. Summed call time fell 140-151 s -> 102-104 s: the
concurrent calls slowed each other. More groups (3, 4; same total in
flight) do not help: the worker is busy ~97% of the wall time (GPU side
saturated) and the engine gets slower with smaller groups (87-93 -> 96-100
s; per-group synchronization). The developer's framework (engine threads
feed per-thread queues, one GPU thread batches) would remove engine-side
waits but cannot beat the GPU side while it is saturated; the lever there
is overlapping each call's copies and conversions with compute (CUDA
streams, double buffering): calls are ~1.9 ms of which compute ~0.8.

Better route: merging transpositions inside the search (a graph instead of
a tree; KataGo's graph search) avoids the repeats at the source and shares
statistics; Corintho has no cycles (P falls every move). Parked by the
developer as a large change.

Also parked (the developer's idea): the spread of the network's
evaluations over symmetric copies of one position measures part of the
model's own noise on that position (test-time-augmentation variance); the
data to study it (networks, compact datasets, solver-proven positions) is
all kept.
