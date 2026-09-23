# 19 — The engine at generation scale: memory rules out 25k games at once; a dynamic schedule saves 20–32%

This is step 2 of the inference plan in entry 18. It measures the engine's real
per-generation cost on this laptop, with the real network in-process. Raw
outputs are in `data/engine-at-scale/runs.txt`. All runs use `model_93.mlp`,
1600 searches, 16 per evaluation, and 20 threads.

## At scale (before the schedule change)

| games at once | engine | network (CPU `Mlp`) | requests | peak RSS |
|---|---|---|---|---|
| 2,000 | 23.2 s | 37.7 s | 71.0M | 1.81 GB |
| 8,000 | 94.9 s | 166.0 s | 285.3M | **7.35 GB** |

Time per request is flat with game count: engine ~0.33 µs and network ~0.55 µs
of wall time at 20 threads.

### Memory: about 0.9 GB per 1,000 games in flight

A 25k-game generation run all at once would need **~23 GB**, and this laptop has
15 GB. The earlier estimate of ~276 KB per game (entry 16) came from the stub,
whose trees are smaller. The real network's are ~3× larger.

This is not the network:

- `Mlp` weights are 0.5 MB, plus ~13 KB of per-thread scratch;
- `selfplay_nn`'s batch buffers are 8k × 16 × 167 floats ≈ 85 MB.

The remainder is the engine: two search trees per game (one per player) plus
game state. In 2023 the other big memory users were the `to_eval_` buffer, 100×
oversized (fixed in entry 09), and Keras `.predict` on ~400k rows.

**Consequence, agreed with Max:** games in flight and games per generation are
separate numbers. Keep only enough games in flight to saturate the GPU (step 3
measures where that is), and start a new game whenever one finishes, until the
generation has its games. That uses less memory and less cache pressure, and
the rolling starts replace the staggered-start trick in `doIteration`.
Shrinking the trees themselves is a separate, later question.

## The OpenMP schedule

`Trainer::doIteration` parallelizes over games with `#pragma omp parallel for`
and no schedule clause, which libgomp treats as static: each thread gets one
contiguous range of game indices. That leaves threads idle in two ways:

1. **Staggered starts.** Game `i` does not start until iteration
   `i / (games / max_searches)`, to spread memory use. Early on, only the
   low-index games are running, so most threads' ranges have nothing to do.
   Late on, games finish at different times.
2. **Hybrid CPU.** An E-core is ~3.4× slower than a P-core (entry 18), and a
   static split waits on the slowest thread at every iteration's barrier.

Every game has its own `mt19937`, and splitmix64 noise is seeded from it, so
the order threads take games in cannot change any game. The runs below confirm
it: identical turn and request counts under every schedule.

Sweep, using a temporary `schedule(runtime)` build, 2,000 games (engine
seconds, two repetitions):

| `OMP_SCHEDULE` | engine s |
|---|---|
| static (what the code did) | 23.98 / 24.26 |
| **dynamic,1** | **16.34 / 16.35** |
| dynamic,8 | 16.92 / 17.35 |
| dynamic,32 | 19.16 / 19.24 |
| guided | 19.88 / 20.13 |

Committed: `schedule(dynamic, 1)` on both game loops, training and testing.
Per-iteration scheduling overhead is negligible next to a game's work between
barriers: 16 searches.

Paired A/B, interleaved, 5 seeds × 1,000 games:

- **engine time −20.3%** (95% CI −21.4 to −19.1);
- turns and requests identical in all 5 pairs.

The gain depends on the game count (−20% at 1k, −32% at 2k), because the
stagger spreads over more iterations when there are more games. The testing
loop has the same structure but cannot be measured with `selfplay_nn`
(`to_play = -1` only). Its change is by analogy.

The golden digest is unchanged (`d7581196f9d30591`), but it is single-threaded,
so it is not evidence here. The identical request counts are.

## Projected generation cost on this laptop

With the dynamic schedule and 20 threads, the engine is ~0.23 µs of wall time
per request (16.3 s / 71.0M). A 25k-game generation at 1600 searches is about
888M requests (25k × 28.3 turns × ~1255 requests per turn):

| | CPU network (now) | GPU network (step 3) |
|---|---|---|
| engine | ~3.4 min | ~3.4 min, all 20 threads on search |
| network | ~7.9 min, sharing the same cores | off the CPU; overlap with alternating groups |
| self-play total | ~11 min | **~3.5–4 min if overlap works** |

These are extrapolations from 1k–8k-game runs, not a measured generation.
Fitting and testing still come on top.
