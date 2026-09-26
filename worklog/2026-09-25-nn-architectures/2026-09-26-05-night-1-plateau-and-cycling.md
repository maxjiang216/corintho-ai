# 05 — night-1: one real step, then a plateau with cycling

2026-09-26, overnight, the developer asleep. Data: `data/night-1-summary.tsv`,
`data/night-1-anchors-3200.tsv`, `data/night-1-progress.log`. Scripts:
`pipeline/arch/overnight.sh`, `night_extra.sh` (commit after `211f2ab`).

## What ran

`runs/night-1` (entry 04's loop): 4 iterations of 250k self-play games with
the current residual 512x4 (fp16), a fresh network trained 40 epochs at lr
3.2e-2 on the newest 14 sample directories (~10.3M positions, ~1.4
iterations of games), no gate. ~92 min per iteration: self-play 62, dataset
4, training 24, matches 2. Every dataset check passed; all new self-play
data matches the corrected symmetry tables (the fix works end to end).

## Results (decisive win rates)

| iter | vs previous (1600) | vs it0 (3200) | vs it1 (3200) | vs gen 1 (3200) | Elo chain |
|---|---|---|---|---|---|
| 1 | 0.568 | 0.547 | - | 0.633 | +47 |
| 2 | 0.502 | 0.592 | 0.505 | 0.606 | +48 |
| 3 | 0.564 | 0.586 | 0.490 | 0.570 | +93 |
| 4 | 0.550 | 0.535 | 0.519 | 0.594 | +128 |

it0 (the 20-epoch supervised 512x4) vs gen 1: 0.644 (1600 games, this
loop's settings; 0.651 earlier). Extra 1600-game check: it3 vs it1 0.503.

## Reading

- Iteration 1 is a real step (+~35-50 Elo over it0).
- After that: flat. it2-it4 are level with it1 (0.49-0.52), and against
  it0 and gen 1 they are no better, even slightly worse by it4.
- Yet each iteration beats its predecessor 55-57%: **non-transitive
  cycling.** The chain (+128) overstates progress; fixed anchors are the
  honest measure (the loop's own 1600-game anchor matches agree).

## Likely cause (hypothesis)

The training window held ~1.4 iterations of games, and every network was
trained from scratch. So each network mostly learns to beat the previous
network's play and forgets earlier opponents' lines: the classic cause of
cycling in self-play, which the wide replay windows of AlphaZero/KataGo
prevent.

## Next: night-2, one change

`runs/night-2`: identical except the window is 40 sample directories
(~4 iterations), 35% of each directory's positions (to stay ~10M positions
in GPU memory), starting from night-1 it4 with the 18 raw directories still
on disk. Extra anchors: night-1 it0 and it1, gen 1. night-1 was paused
after it4 (resume with `NAME=night-1 ITERS=8 arch/overnight_supervise.sh`).

If night-2 still cycles, the next suspects are training from scratch
(try warm starts plus weight averaging) and the search settings (the
policy targets may already match what 1600 searches can improve).
