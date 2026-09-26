# 04 — Training recipe, size, first matches, and the overnight loop

2026-09-25 evening. Commits `84464ce` (block order, sweep 3), `ae47343`
(exporter, matches), `417418f` (overnight loop). Data: `data/sweep3-*`,
`data/sweep4-*`, `data/first-matches.txt`. Research note:
`research-other-games.md`.

## Decisions by the developer

- Randomize orientation (random symmetry per sample); ignore the value gain
  of "no symmetries" (entry 03), which is a quirk of this data's lineage.
- Keep the inputs pure (the 70 state inputs): line and legal inputs helped
  little on the bigger nets.
- The classic ResNet block order (Linear -> BN -> ReLU -> Linear -> BN, add,
  ReLU), though pre-activation was 0.005 better in one comparison.
- "Performance is paramount": the residual 512x4.

## Sweep 3: simple choices (residual 256x4, 20 epochs)

- Seed noise: loss 0.9205 / 0.9193 / 0.9187 (spread ~0.002).
- Classic order +0.005 vs pre-activation. Weight decay 1e-3 vs 1e-4: none.
- **Learning rate is the big lever:** (classic order) 2e-3 0.9246, 4e-3
  0.9132, 8e-3 0.9022, 1.6e-2 0.8928, **3.2e-2 0.8861**, 6.4e-2 0.8871,
  1.28e-1 0.8970. With 60 epochs at 3.2e-2: 0.8303 (value MSE 0.499, top-1
  63.3%), train = val throughout: far from converged at 20 epochs.

## Sweep 4: size x inputs (classic order, lr 3.2e-2, 20 epochs)

loss (plain / +16 lines / +96 legal): 256x4 0.8861 / 0.8834 / 0.8737;
256x8 0.8683 / 0.8660 / 0.8583; 384x4 0.8482 / 0.8446 / 0.8405; 512x4
**0.8197** / 0.8193 / 0.8157. Width beats depth at equal parameters (384x4
1.25M beats 256x8 1.1M). Extra inputs shrink toward noise with width.

## Cost under TensorRT, compact I/O (16k-row call)

fp32 / fp16 ms: gen_93 1.89 / 1.74; 256x4 3.84 / 2.25; 384x4 6.02 / 2.75;
512x4 8.73 / 3.66; 768x4 16.2 / 5.97; 256x8 6.29 / 3.17. Wide nets need fp16
(tensor cores). fp16 vs TensorRT fp32 on the trained 512x4: top move differs
on ~0.5% of rows, max value difference 0.06.

## First matches (1600 games, 1600 searches each, vs gen 1 = the old best)

Trained 20 epochs from scratch on runs/full-2: 256x4 53.6% of decisive
games (+11% self-play time), 384x4 59.8% (+22%), **512x4 65.1% (+40%)**.
Engine threads with the 512x4: 20 -> 56.7 s, **14 -> 54.3 s**, 10 -> 58.0
s (4000 games): 14 physical cores are as fast as 20 threads.

## Why the old loop plateaued (hypothesis, from the research)

The old pipeline halved the learning rate on every failed promotion:
1e-3 (gens 1-50) -> 6.25e-5 (~70) -> 5e-6 (90+). A gain too small for the
1600-game gate at 95% fails it, the rate halves, the next network moves
less: a spiral. No comparable project anneals on rejection; they use
steady or scheduled rates, a sliding window of recent games, often no gate,
and weight averaging. Also every generation had 25% wrong policy targets
(entry 03), no masking, and trained only on the latest generation.

## The overnight loop (`pipeline/arch/overnight.sh`)

Iteration i: self-play 10 x 25k games with the current network (fp16,
14 threads) -> dataset of the newest 14 sample directories (runs/full-2
gens 1-4 are the oldest until pushed out) -> a fresh 512x4 trained 40
epochs at lr 3.2e-2, validated on the newest directory -> matches for the
record vs the previous iteration (Elo chain), iteration 0 (the 20-epoch
supervised 512x4) and gen 1. No gate: the new network always plays next.

Robustness: each step's output is its marker (reruns resume; a
half-written self-play directory is redone); `overnight_supervise.sh`
reruns after failures (5 in a row at most) under a lock. Raw samples that
leave the window are deleted (logs kept); datasets two iterations old are
deleted. Tested at tiny size, including a kill mid-self-play. Memory
tested: a 10M-position dataset peaks at 4.4 GB GPU, 2.5 GB RAM, ~30 s per
epoch for the 512x4.

Estimated ~1.5 h per iteration: self-play ~57 min, dataset ~3, training
~20-25, matches ~5. Launched 2026-09-25 23:10 as `runs/night-1`, 8
iterations.

**Reading the results:** `pipeline/runs/night-1/summary.tsv` (one row per
iteration: self-play time, dataset size, validation metrics on the newest
data, decisive win rates vs previous / it0 / gen 1, Elo chain) and
`progress.log`. Validation metrics are on each iteration's own newest data,
so they are not comparable across iterations; the matches are. A plateau
shows as vs-previous near 0.5 and a flat Elo chain.
