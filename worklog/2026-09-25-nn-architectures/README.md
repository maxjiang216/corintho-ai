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

- Supervised test bench (idea 7) with masking (3), symmetry (4) and line
  features (5) as switches; the current architecture as reference.
- Re-run `pipeline/arch/cost.py` candidates under `trt:` with compact I/O.
