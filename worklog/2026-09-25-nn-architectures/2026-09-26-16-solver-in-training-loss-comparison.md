# 16 — Solver in self-play: a first training comparison (loss only)

2026-09-26. Runs `pipeline/runs/solve-27` and `pipeline/runs/solve-0`
(`arch/loop.sh`), both `NAME=... GENS=3 INIT=runs/night-2/it1/net.pt`,
otherwise default config (`GAMES=25000`, `WINDOW=4`, `EPOCHS=2`, `LR=1e-2`).
`solve-27`: `SOLVE_P=27` (default, entry 14). `solve-0`: `SOLVE_P=0` (off).
Paired only on INIT and gen count; no strength comparison yet (see below).

## Results

**solve-27** (games end by exact solution at P <= 27):

| gen | selfplay_s | positions | val_value_mse | val_policy_ce | val_top1 | turns/game | first_player_score |
|---|---|---|---|---|---|---|---|
| 1 | 274 | 483,091 | 0.3916 | 1.5670 | 0.6220 | 19.32 | 0.4985 |
| 2 | 526 | 953,956 | 0.5439 | 1.3842 | 0.6728 | 18.83 | 0.5010 |
| 3 | 279 | 1,426,269 | 0.5543 | 1.3119 | 0.6828 | 18.89 | 0.5076 |

**solve-0** (solver off, games played to natural/turn-limit end):

| gen | selfplay_s | positions | val_value_mse | val_policy_ce | val_top1 | turns/game | first_player_score |
|---|---|---|---|---|---|---|---|
| 1 | 353 | 746,245 | 0.3686 | 1.2333 | 0.6719 | 29.85 | 0.4974 |
| 2 | 363 | 1,495,196 | 0.3748 | 1.2056 | 0.6811 | 29.96 | 0.5030 |
| 3 | 351 | 2,240,843 | 0.3852 | 1.1850 | 0.6852 | 29.83 | 0.4998 |

(`positions` is `train_positions + val_positions` summed over the trailing
WINDOW=4 generations, per `loop.sh`'s summary row, not the current
generation alone.)

first_player_score ~0.50 throughout both — no first-move edge in Corintho
at this network strength, solver on or off.

## The loss gap: not label extremity

First guess: the solver hands positions a sharper/more "extreme" value
label (exact win/loss) than an AlphaZero game outcome would, inflating
val_value_mse, and a forced best move at solved positions similarly
sharpens the policy target, inflating val_policy_ce. Checked directly
against `data.npz` — wrong:

```
solve-27 gen1: mean|v|=0.9862 extreme%=98.62 zero%=1.38
solve-27 gen2: mean|v|=0.9812 extreme%=98.12 zero%=1.88
solve-27 gen3: mean|v|=0.9811 extreme%=98.11 zero%=1.89
solve-0  gen1: mean|v|=0.9635 extreme%=96.35 zero%=3.65
solve-0  gen2: mean|v|=0.9637 extreme%=96.37 zero%=3.63
solve-0  gen3: mean|v|=0.9639 extreme%=96.39 zero%=3.61
```

Both are ~96-99% |v|=1 either way (draws are rare regardless of the
solver); the value label is the same kind of target (final game outcome,
backed up to every position) in both conditions. The 2-point extreme%
gap does not account for a 40%+ relative gap in val_value_mse.

## What's real, and what's still open

- Shorter solved games (19 vs 30 turns) log proportionally fewer
  positions per 25k-game batch (~480k vs ~745k per generation, ratio
  matching the turns-per-game ratio almost exactly). Less data per
  generation for `solve-27` plausibly explains some of the gap, but not
  the trend.
- Unexplained: `solve-27`'s val_value_mse *worsens* gen over gen
  (0.39 -> 0.55) while `solve-0`'s stays flat (0.37 -> 0.39), even though
  per-condition position count is roughly flat across its own
  generations. Data volume alone doesn't cover this. Candidates not yet
  checked: LR/warm-start interaction with the more truncated-game data
  distribution; something compounding through the WINDOW=4 rolling
  dataset once solved endgames repeat across generations. Needs a look
  at actual per-position predictions vs targets on `solve-27` gen 3, not
  further guessing.
- Loss is not a strength signal here either way, given the differing
  data distributions and (for `solve-27`) worsening value fit — a paired
  match (`arch/paired.py`) between the two gen_3 networks is the next
  step before drawing any conclusion about whether the solver helps or
  hurts final playing strength.

## A process note

Mid-run, a duplicate `arch/loop.sh` was accidentally started on top of
the still-running original (a `nohup ...  &` combined with the harness's
own backgrounding double-backgrounded it, and the process was
misdiagnosed as dead). Both instances wrote into `solve-27/gen_3/samples`
concurrently; `dataset.py`'s symmetry check caught it
(`AssertionError: copies match neither table`). Fix: kill the duplicate,
clear `gen_3`'s samples/dataset/selfplay output, rerun `loop.sh` (resumes
from `gen.done` markers, so only gen_3 redid). `solve-27/matches.tsv` and
`summary.tsv` for gens 1-2 still carry one harmless duplicate row each
from the race window (both rows identical; not corrupted, just doubled).
