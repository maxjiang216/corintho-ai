# 09 — Network calls: queued helps, more groups and pipelining do not

2026-09-26. Commit `a9512ad` (worker thread); pipelining attempt reverted,
kept as `data/pipeline-attempt.patch`. Data: `data/driver-groups-ab.txt`.

## Queued network calls (adopted in the source)

Self-play splits games into groups; the engine searches one group while
the GPU evaluates another's batch. The old driver ran each network call on
its own std::async thread, so the two groups' calls could run on the GPU
at the same time. The worker-thread driver (built for the cache, entry 07)
runs all calls one at a time. Same network (loop-1 gen 15, fp16 TensorRT),
8000 games, 4000 in flight, 14 threads:

- old 112.8 / 122.2 s; queued 105.2 / 108.8 s (with a stray process of
  mine using a core throughout). Summed call time 140-151 s -> 102-104 s:
  concurrent calls slowed each other.
- Clean rerun (quiet machine), queued: 102.0 / 104.2 / 105.1 s.
- 3 or 4 groups (same total in flight): no better (engine slower with
  smaller groups, 87-93 -> 96-100 s; the GPU side is saturated: the worker
  is busy ~97% of wall).

The main binary (`pipeline/build`) still has the old driver; rebuilding it
adopts the gain.

## The developer's framework

Engine threads feeding per-thread lock-free queues (single producer,
single consumer), one GPU thread batching whatever is queued, static game
ownership with per-game ready flags: removes engine-side waits entirely.
Not built: with the GPU side saturated, it cannot beat the GPU time.

## Pipelining copies with compute (tried, dropped)

Each backend on its own CUDA stream (ORT user_compute_stream), inputs and
outputs bound to our device buffers, the worker enqueuing input copy ->
wait for the other batch's compute (event) -> run (no sync) -> output copy,
up to two batches in flight. Correct (identical sample digests). Run
returns in 0.37 ms (truly asynchronous), finish waits ~1.9 ms.

Clean A/B (3 rounds, interleaved): queued 102.0 / 104.2 / 105.1 s,
pipelined 114.0 / 111.3 / 113.8 s (engine waits 16-17 -> 24-25 s): 9%
slower. With one group (no overlap possible) the two paths cost the same
per call (4.44 vs 4.43 ms at ~15k rows), so the async path is not the
cause; the premise was: my estimate (call 1.9 ms, compute 0.8) came from
the 12x100 network. With the 512x4 a call is ~4.4 ms of which compute is
~3.7 ms (entry 04): copies are ~15%, too little to repay the coordination.

## Where the GPU side can still shrink

Compute is ~250 ns per row for the 512x4 (fp16). Fewer rows: the game
solver (entry 08; skips ~35-40% of positions' searches) and graph search
(33% within-game repeats). Cheaper rows: TensorRT tuning (profile shapes at
the real batch size, CUDA graphs), INT8.

Also found: a stray test process of mine (a first solver-test attempt,
plain minimax) used one core from ~08:40 to ~10:55; timings in that window
(solver pinned times, the driver groups A/B) are noisier; node counts,
instruction counts and results are unaffected.
