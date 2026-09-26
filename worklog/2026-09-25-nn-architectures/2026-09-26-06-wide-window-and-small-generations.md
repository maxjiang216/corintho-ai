# 06 — The wide window breaks the plateau; small generations

2026-09-26 morning. Commits `cbe4064` (window options), `7de0495` (loop.sh),
`e94b42f` (game logs). Data: `data/night-2-*`, `data/loop-1-*`.

## night-2: one change, a wider window

night-1 (entry 05) trained each fresh network on the newest 14 sample
directories (~1.4 iterations of games) and cycled. night-2 changed only the
window: the newest 40 directories (~4 iterations), keeping a random 35% of
each directory's positions (`dataset.py --keep`) to stay at ~10M positions.
It started from night-1 it4; its first window had 28 directories (7.1M
positions).

night-2 it1 (3200-game anchors): 62.3% vs night-1 it0, 61.5% vs night-1 it1,
**70.2% vs gen 1** (night-1's best against gen 1 was 63%). About +80 Elo over
anything night-1 produced, against fixed opponents: the plateau was the
narrow window. Paused after it1 for the next design.

## Training-budget test (a side question)

Is iteration 1's jump (night-1) from its new data or from its ~9x larger
training budget? A fresh 512x4 on the old data only (runs/full-2), 180
epochs: its validation loss fell to 0.65 (0.82 at 20 epochs), yet it lost
40.1% to night-1 it0 and 38.3% to it1. More training on old data made the
network imitate weaker players more closely: the gain came from the data,
and supervised loss on old data is a poor proxy for strength.

## Research: the old loop's learning rate

The old pipeline halved the learning rate on every failed promotion (1e-3
for gens 1-50, 6.25e-5 by ~70, 5e-6 from 90): a spiral, since a smaller
rate makes the next network more alike, so it fails again. See
`research-other-games.md`: no comparable project anneals on rejection.

## loop-1: many small generations (the developer's design)

`pipeline/arch/loop.sh`: 25k games per generation with the current network
(fp16 TensorRT, 14 threads); the generation's data compacted (checked; ~0.2
GB kept per generation, raw samples deleted); the previous network
continued (warm start) for 2 passes over the last 4 generations at lr 1e-2
with 2% held out; anchor matches every 5 generations; no gate. ~7.5 min per
generation. Warm-start rate from a check on night-2 data: none 0.6938, 1e-3
0.6861, 3e-3 0.6819, **1e-2 0.6774**, 3.2e-2 0.6834 (loss on a held-out set
after 2 passes).

Results (decisive win rates, 1600 games): gen 5 vs its start (night-2 it1)
57.9%, vs night-1 it0 67.1%, vs gen 1 70.5%; gen 10 vs start 58.3%, **vs gen
5 60.3%**, vs night-1 it0 64.5%, vs gen 1 68.9%. So the first 5
generations (~40 min) gained about as much as a 1.5 h night-2 iteration,
but gen 10 beats gen 5 without improving on the fixed anchors: the cycling
pattern again, likely because 4 generations x 25k games (100k games) is a
narrow window again (night-2's was ~1M games, subsampled). Next time: a
window of ~20-40 generations, subsampled.

Stopped at gen 15 (the developer: no ongoing run while the engine keeps
changing). All per-generation data and networks are kept.
