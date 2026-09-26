# Network architectures

Started 2026-09-25 on branch `exp/nn-architectures`, cut from `main` after
PR #168 (the engine work and the GPU pipeline, worklog
`2026-09-20-performance-and-plateau-investigation/`, entries 01–29).

**Goal: a stronger network.** Try architectures, inputs and training changes
on fixed data first (supervised), then confirm the best by matches.

## Entries

| # | entry | what |
|---|---|---|
| 01 | [self-play is GPU-bound; TensorRT and compact I/O](2026-09-25-01-gpu-bound-tensorrt-compact-io.md) | The engine waited ~38% of self-play on the GPU. TensorRT + uint8/fp16 I/O: call 3.97 → 1.94 ms, self-play wall −22%. Same-network match even. Now `run.py`'s default. |
| 02 | [float in selection and backup](2026-09-25-02-float-engine.md) | Double literals removed. Engine −1.5%, not bit-identical, games alike. |
| 03 | [supervised bench, masking, the symmetry bug](2026-09-25-03-supervised-bench-masking-symmetry-bug.md) | **`move_symmetries` rows 2 and 6 were swapped since 2023: 25% of every generation's policy targets were wrong.** Fixed. Masking helps; residual 256x4 is best so far. |
| 04 | [recipe, size, first matches, overnight loop](2026-09-25-04-recipe-size-first-matches-overnight.md) | lr 3.2e-2 (16x) is the big lever; width beats depth; residual 512x4 (fp16) beats gen 1 65% after 20 epochs. Overnight loop `runs/night-1`. |
| 05 | [night-1: plateau and cycling](2026-09-26-05-night-1-plateau-and-cycling.md) | One real step (it1), then flat: each iteration beats the last 55-57% but not fixed anchors. Narrow window suspected; night-2 tests a 4-iteration window. |
| 06 | [wide window, small generations](2026-09-26-06-wide-window-and-small-generations.md) | A 4-iteration window broke night-1's plateau (70% vs gen 1). loop-1 (25k-game generations, warm starts): fast early gains, then cycling again (4-generation window). |
| 07 | [repeated evaluations, the cache](2026-09-26-07-repeated-evaluations-and-the-cache.md) | 50% of network rows repeat (33% within a game; 56/38% up to symmetry). An evaluation cache cannot pay while GPU wait is ~20%: parked. Graph search later. |
| 08 | [the endgame solver](2026-09-26-08-endgame-solver.md) | Exact solver module; 70x faster than the probe at P <= 24 (line-making moves first, branch-free bit tricks). P_game ~26-27, P_node ~19. |
| 09 | [network calls: queueing, groups, pipelining](2026-09-26-09-driver-queueing-and-pipelining.md) | One call at a time: self-play -7 to -11%. More groups: no gain. Copy/compute pipelining: correct but 9% slower (copies are only ~15% of a 512x4 call). |
| 10 | [solver move ordering study](2026-09-26-10-solver-move-ordering-study.md) | History heuristic (0.80x nodes at P <= 27). Per-move statistics (the developer's idea): 89% first-move cutoffs; every statistics-suggested reordering lost in controlled tests (observational bias). |
| 11 | [legal-move generation](2026-09-26-11-legal-move-generation.md) | pext in computeSpaceInfo and basicLegalMoves, branch-free lineBreakers: -13.5% instructions, bit-identical (digests, rulecheck, verify). Engine-wide. |
| 12 | [null-window solves](2026-09-26-12-null-window-solves.md) | "Win?" then "at least a draw?": 0.91-0.93x nodes at P <= 27-30, neutral at P <= 24. |
| 13 | [the solver's table](2026-09-26-13-solver-table.md) | 16-byte entries, 4-way buckets with one AVX2 compare, prefetching every child: P <= 30 -25%, P <= 27 -14%. |
| 14 | [the game solver in self-play](2026-09-26-14-game-solver-in-self-play.md) | A game at P <= 27 ends at once and is solved on spare threads (pausing it was 36-55% slower). P_game 27: self-play wall -7%, turns per game 29.9 -> 19.6; 28-29 cost wall time. |
| 15 | [node solver; games end when known](2026-09-26-15-node-solver-and-proven-endings.md) | Games end once proven or at P 27, checked every move. Node solver (P_node 15-19): slower, and paired matches (per-game, same seeds) show no gain, mild harm at 17-19: off. |

## The developer's ideas (2026-09-25), to work through

1. The game may be simple enough for a **smaller** network; find out how
   small can still match gen_93 (distillation on its own outputs).
2. The GPU now has headroom (entry 01), so a **bigger** network may be
   affordable; re-cost candidates under TensorRT + compact I/O.
3. **Illegal-move masking.** Training takes softmax over all 96 moves with a
   0 target on illegal ones; the search (`TrainMC::setProbs`) keeps only legal
   moves and renormalizes. So the network spends capacity on legality that
   play throws away. Fix: softmax over legal moves only; the mask must be
   recomputed from the 70-float state (a small C++ tool).
4. **8-fold symmetry** in the model rather than 8x stored data: at least
   unique rows + random symmetry on the GPU (1/8 memory); maybe canonical
   orientation or a symmetric architecture.
5. **Line features:** one input per space, "this space is part of a line"
   (16 inputs; the developer prefers this to one per line shape).
6. BatchNorm placement, residual blocks, LayerNorm, no norm (all fold or
   cost the same at inference).
7. **Supervised experiments** on `pipeline/runs/full-2` (gen_1's samples were
   played by gen_93; gens 2–4 by gen 1): which models fit best, how small.

## Data available

`pipeline/runs/full-2/gen_{1..4}/samples/`: 5.67M rows each, stored as 8
symmetries of 709k positions (2.84M unique positions in all), ~3.6 GB per
generation. States are multiples of 0.25; 16 spaces x 4 bits (base, column,
capital, frozen) + 6 reserves, player to move first. Old cloud generations
kept no samples.

## Where to resume

- No run is going (loop-1 stopped at gen 15; night-1 paused after it4,
  night-2 after it1; all resumable, all data kept).
- Self-play with the solver: `--solve-p 27` (entries 14-15); `--node-p`
  stays off. Not yet measured: a paired **training** run with and without
  `--solve-p 27` (queued driver, TensorRT, wider window).
- `pipeline/build/corintho_play` is rebuilt (queued calls, solver), and
  `arch/loop.sh` passes `--solve-p 27` by default (`SOLVE_P=0` turns it
  off). PR #169.
- Strength comparisons: paired matches (`game_scores.txt`,
  `arch/paired.py`, entry 15); the second player wins ~91% of test games,
  so plain totals are noisy.
- Open (entry 15): the network gets no data at P <= 27 but still evaluates
  search leaves at P 18-27 (value-only solver-labelled samples, or measure
  its error there first); shortest-win / longest-loss lines if ever needed.
- Parked: graph search (transpositions), evaluation cache, symmetry
  variance, solver SIMD, NNUE move ordering, proof-number search.
- The symmetry fix `873988e` should reach `main` (small PR).
