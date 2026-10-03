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
| 16 | [solver in self-play: a first training comparison (loss only)](2026-09-26-16-solver-in-training-loss-comparison.md) | `solve-27` vs `solve-0`, 3 generations each, same init. Value/policy loss higher with the solver, but not from label extremity (checked: ~97-99% \|v\|=1 either way) — data-volume and a widening gen-over-gen gap in `solve-27`'s val_value_mse are unexplained. No strength comparison yet (`paired.py` next). |
| 17 | [where repeats come from; one self-play tree](2026-09-26-17-shared-self-play-tree.md) | Repeated rows: 20.6% repeat the *other player's* tree, only 6% are transpositions within a search. `--shared-tree 1`: rows new to the game 69 -> 87% at the same cost (1600 new searches per move), root visits ~1.8x. Root visits now capped at 32,000 (int16_t). Training effect not yet measured. |
| 18 | [solver endings hurt training](2026-09-26-18-solver-endings-hurt-training.md) | Paired training to gen 5 from night-2 it1: solve-27 scores 0.24 vs its start (0.40 with both sides adjudicated at P 27), solve-0 scores 0.57. Endgame play drifts without data and the middlegame gets worse too. `loop.sh` defaults to `SOLVE_P=0`. |
| 19 | [the solver plays solved games out](2026-09-26-19-solver-plays-games-out.md) | `Solver::playOut` (winner: immediate win or the table's move; loser: hardest refutation; draw: most losing replies), 0.56 ms per line; labels 100% exact; self-play 12% faster than solver off. 5 gens: no collapse, but gen 5 loses to solve-0 gen 5 0.464 (z ~ -5); late-endgame values drift down. Solver lines are not the positions the search evaluates. Next: MCTS play with exact relabelling. |
| 20 | [exact relabelling; run-to-run noise](2026-09-26-20-exact-relabelling.md) | `--relabel-p 27`: games played out as usual, exact value labels at P <= 27 (identical positions and policies, 9% of values change, +6.7% time). Endgame values fine. Head to head vs solve-0 gen 5: 0.467; but a same-settings replicate (`SEED=1`) also scores 0.475 against it, and relabel beats the replicate 0.518. Run-to-run noise ~+-25 Elo at 5 gens: comparisons need several seeds per arm. |
| 21 | [small value networks for alpha-beta](2026-10-03-21-small-value-networks.md) | Distilled 70->H->32->1 value nets (AZ teacher above P 27, exact below) on 3.7M positions: h1024 matches the teacher as closely as a second AZ network does at P 23-48 (0.050 vs 0.055 at P 28-35), at 0.7-2.6 us per position in plain float (AZ: ~11.5 us). Still improving with more data and steps. Next: AB engine vs MCTS at equal time. |
| 22 | [alpha-beta vs MCTS, preliminary](2026-10-03-22-alpha-beta-vs-mcts.md) | `ab_match`: AB (small net, ID negamax, TT, PVS) vs MCTS (AZ solve-0 g5), one CPU thread each, ~2 s per move, 200 games with random openings, adjudicated at P 27: **54-6-140 (0.285, ~-160 Elo)**, opening pairs 2-48. AB reaches depth ~5.5 at 147k nodes/s. Then the developer's suggestions: solver move ordering, line extension, late-move reductions, using the full time: depth 8.1, **0.445** at equal time (pairs 12-22-66). |
| 23 | [smaller and 8-bit small networks](2026-10-03-23-int8-small-network.md) | Smaller is at least as good (h256 float 0.468, depth 9.0). int8 with AVX-VNNI (`small_net.h`): 4.4x faster than float for h256 (0.48 us) after fixing aliasing; layer-2 weights clipped to +-1.98 in training so scale 64 is lossless. h256 int8 vs MCTS at equal time: 96-2-102 (0.485), 1.12M nodes/s. Incremental first layer (exact): 1.46M nodes/s, depth 10.0, **104-3-93 (0.527)**. |
| 24 | [symmetry-tied and reserve-monotone small networks](2026-10-03-24-symmetric-and-monotone-small-networks.md) | Code for the next distillation sweep. `--tie`: weights tied across the 8 symmetries, exactly invariant, 3.3k free parameters instead of 26k, same dense network at inference. `--mono-m/--mono-m2`: board-only and reserve-aware units, value provably monotone in reserves. `export` writes `small_net.h`'s file. Tied layer 2 needs lr / 8 (it saturated for good in 1 of 6 seeds). `arch/nnue_sweep.sh` runs 3 seeds of each. |

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
  night-2 after it1; solve-27 and solve-0 stopped after gen 5; all
  resumable, all data kept).
- Entries 19-20: solver playout (`SOLVE_P=27`) and exact relabelling
  (`RELABEL_P=27`) are both within run-to-run noise of no solver at gen 5
  (~+-25 Elo between same-settings runs, `SEED`). Comparing training
  treatments needs several seeds per arm. Defaults stay off.
- **Ending self-play games by solver at P 27 hurts training (entry 18):**
  solve-27 gen 5 scores 0.24 against its own starting network, solve-0 gen
  5 scores 0.57. `arch/loop.sh` now defaults to `SOLVE_P=0`. Suspected
  cause: no training data at P <= 27. `--node-p` stays off.
- `pipeline/build/corintho_play` is rebuilt (queued calls, solver). PR
  #169. `build-shared/` has `--shared-tree` (entry 17).
- Strength comparisons: paired matches (`game_scores.txt`,
  `arch/paired.py`, entry 15); the second player wins ~91% of test games,
  so plain totals are noisy.
- Open (entry 15): the network gets no data at P <= 27 but still evaluates
  search leaves at P 18-27 (value-only solver-labelled samples, or measure
  its error there first); shortest-win / longest-loss lines if ever needed.
- Paired training run going (`arch/paired_loop.sh`): solve-27 vs solve-0 from
  night-2 it1, alternating generations to 20 each. Compare gen_N networks
  with paired matches.
- `--shared-tree 1` (entry 17) is built and checked but off; loop.sh does
  not pass it. Next: its own paired training run.
- Parked: graph search (6% of rows, entry 17), evaluation cache, symmetry
  variance, solver SIMD, NNUE move ordering, proof-number search.
- The symmetry fix `873988e` should reach `main` (small PR).
