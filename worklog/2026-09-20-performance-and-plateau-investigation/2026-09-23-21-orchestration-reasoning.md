# 21 — Orchestrating self-play on one laptop: the design so far, and the reasoning behind it

A synthesis entry, with no new code. Entries 18–20 hold the measurements. This
entry records how they fit together, what the developer decided and why, and
why the next step is the engine rather than the GPU backend. Read it before
building the new self-play driver.

## Where the session started

The developer asked for a high-level time breakdown of a training generation,
intending to optimize inference. The only full generation on record is gen_93:
a 2023 cloud run, 32 vCPUs, CPU only, Keras `.predict`.

| Phase | Time | Share |
|---|---|---|
| Self-play inference | 24m58s | 44% |
| Self-play engine | 14m47s | 26% |
| Fitting | 9m23s | 17% |
| Testing inference | 6m20s | 11% |
| Testing engine | 31s | 1% |

Inference was ~55% of the generation, and ran at 10–15% of the CPU's peak.
Keras executes op by op over a 250k-row batch, streaming activations through
main memory. So inference looked like the target.

## Decisions (the developer's), 2026-09-23

1. **Train on this laptop, overnight.** It has an RTX 3060 Laptop GPU (6 GB),
   an i7-12700H (6 P-cores + 8 E-cores, 20 threads) and 15 GB RAM. The cloud's
   CPU-only numbers stop being the reference.
2. **The C++ side owns inference; Cython goes.** Once the model is a runtime
   artifact that a C++ library loads, the pipeline can take any structure:
   rolling games, overlapping groups, pinned buffers. Python only fits.
3. **PyTorch for fitting.** Chosen for room to experiment with the model later.
   The planned route is supervised training on the current best model to
   compare architectures, then continuing self-play with the winner. The
   architecture stays as it is for now.
4. **Python packages through uv.**
5. **Efficiency is judged where the GPU saturates**, not at 25k games per
   batch.

## What the measurements changed

### The GPU is not the bottleneck; the engine is (entries 18–20)

Per 25k-game generation, 1600 searches, ~888M network rows:

| | CPU network (`Mlp`, 20 threads) | GPU network (fp32) |
|---|---|---|
| network | ~7.9 min, on the engine's cores | ~2.6 min (~175 ns/row end to end) |
| engine | ~3.4 min (after the dynamic schedule) | ~3.4 min |

On the GPU, the network alone takes less time than the engine. So:

- A faster GPU kernel (fused MLP, fp16) buys nothing yet. That is fortunate,
  because fp16 and TF32 put probabilities off by up to 0.18 with this network
  (entry 20).
- The GPU stays at fp32, and the engine sets the pace.
- Overlapping the engine and the GPU can at best hide the GPU's ~2.6 minutes
  behind the engine's ~3.4. It cannot make the engine faster.

### Memory limits games in flight, and the stagger is real (entry 19)

- The real network's trees cost ~0.7 GB per 1k games (after the schedule
  change; 0.9 before). So 25k games at once, ~18–23 GB, will not fit in 15 GB.
- **Games in flight and games per generation are separate numbers.** The
  developer's framing: keep only as many games in flight as the GPU needs to
  run efficiently, and start a new game whenever one finishes.
- The developer asked whether the existing staggered start was needed. It had
  been added to cut peak memory. Measured: it cuts peak memory by ~30% at
  unchanged engine time. The mechanism is that each tree grows over a turn and
  shrinks at the move, so games in phase peak together.
- Rolling starts keep the benefit: stagger the first launch, and replacement
  games land at scattered points in the turn anyway.

### How many games in flight

Each game adds ~12.5 rows per batch, not 16. `searches_per_eval = 16`
descents, but ~22% end on terminal positions and need no evaluation (1,255
requests per 1,600 searches). The GPU is within 5–12% of its plateau at
16k–32k rows:

- one group: ~1,300–2,600 games, ~0.9–1.8 GB;
- two alternating groups: double that.

### `searches_per_eval` stays at 16

- The developer noted that batching 16 descents is slightly inexact. The
  descents do not see each other's results, only a provisional +1 evaluation
  that acts as a virtual loss.
- They asked whether a smaller value is now affordable. It costs engine time:
  +13% at 8, +27% at 4, +60% at 1, all per-iteration overhead. The engine is
  the bottleneck, so **16 stays**, until a strength test says otherwise.

### Use all cores

- The developer asked what P-cores and E-cores are, and whether P-cores can be
  guaranteed. Pinning is harmless: `taskset` or `sched_setaffinity` limits one
  process only.
- But with the dynamic schedule, the E-cores add ~22% engine throughput (all
  20 threads 8.0 s, P-cores only 10.2 s). So use every core.
- Pin only for measurement, and possibly for the thread that feeds the GPU,
  which is untested.

## The orchestration design as it stands (not built)

- **A C++ self-play driver process**, replacing `main.pyx`. It loads the
  network through a backend interface, runs the games, and writes samples.
  - Backends: `Mlp` on CPU (exists; the reference and fallback) and a GPU
    backend. ONNX Runtime from its release tarball is preferred over the
    libtorch in the torch wheel: it takes any ONNX-exported architecture, has
    the lowest overhead per call, and has a TensorRT path later.
- **A Python/PyTorch fitter**, which exports the network each generation.
- **A driver loop** around both, including the promotion gate. The gate
  changes from entry 12 would move in here.
- **Inside the driver:**
  - rolling game starts with an initial stagger;
  - games in flight sized to the GPU;
  - possibly two alternating groups, so one group searches while the GPU
    evaluates the other. The alternative is one group with the GPU idle
    during the search.

## Why the engine comes next

With the GPU at ~2.6 minutes and the engine at ~3.4, every orchestration
tradeoff is set by engine speed:

- how much alternating groups can gain: up to hiding the GPU time entirely,
  and nothing if the engine keeps it busy anyway;
- whether a smaller `searches_per_eval` is affordable;
- how many games in flight the GPU needs;
- how much memory the trees take.

A faster engine moves all of these, so the developer chose to return to it
before building the driver. Tree memory is the other engine lever: smaller
nodes allow more games in flight and less cache pressure.

## Open threads

- The strength questions remain the real goal, and none of this speed work
  has been shown to move strength yet:
  - colour imbalance (entries 10–11);
  - the rules fix (entry 04);
  - the draw-visit and solver-backup questions (entry 17);
  - the cost of `searches_per_eval`.

  A **match harness** (same network, different settings, alternating colours)
  would answer them. It should come soon after the driver.
- Why peak memory per game fell from ~0.9 to ~0.7 GB after the schedule
  change. The guess is glibc's per-thread arenas; unverified.
- The E-core effect on the testing loop's schedule is by analogy, not
  measured.
