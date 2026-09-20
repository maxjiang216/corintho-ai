# 2026-09-20 — Enabling LTO, and a hole in the harness

Branch: `perf/training-overhaul`. First actual optimization on this branch.

## TL;DR

- **`-flto` added to `corintho_ai/python/setup.py`.** One line, no source
  changes, digests unchanged. Measured **−23.3%** and **−24.6%** on two
  interleaved runs on battery, and **−23.2%** re-measured on AC. Power state
  changes precision, not the ratio. At 14 threads it is **−18.8%**, since the
  parallel run is more memory-bound.
- **Instructions fell 32.7% but time fell only 23%** — the removed work was
  call/return overhead, the cheapest instructions in the program. Branch
  mispredicts even rose 2.2%.
- **Found and fixed a hole in the benchmark harness.** An identical build
  measured 77% slower than its own baseline forty minutes earlier, because the
  laptop had moved from AC to battery. Nothing recorded this.
- **Could not verify the real build**: no cython or tensorflow installed
  locally. Verified the main LTO risk separately instead.

## The change

`corintho_ai/python/setup.py` is the build script for the training pipeline: it
compiles the six engine `.cpp` files together with `main.pyx` into the
`corintho` Python extension module that `wrapper.py` imports.
`scripts/bash/build.sh` invokes it. Its flags were `-O3 -std=c++17 -fopenmp
-DNDEBUG`.

Added `-flto` to `extra_compile_args`, and `-flto -O3` to `extra_link_args`.
Both are required: with LTO the optimization happens at link time, so passing
the flag only to the compiler yields none of the benefit — a common and silent
mistake.

**Why it helps this code specifically.** `node.h` declares the one-line
accessors (`num_legal_moves` line 63, `move_id` 66, `probability` 67) but their
bodies live in `node.cpp`. `TrainMC::chooseNext` calls all three per edge, in a
loop. Across the translation-unit boundary the compiler cannot see the bodies,
so each is a real call, and it must assume memory may have changed across it.
Callgrind puts those accessors at **10.7% of all instructions** without LTO.

## Measurement

Interleaved A/B via the new `bench/ab.sh`, 7 reps per arm, single-threaded,
50 games × 1600 searches:

```
  no-lto      median  9.941s   min  9.837  max 10.626  spread 7.9%
  with-lto    median  7.628s   min  7.329  max  7.675  spread 4.5%

  with-lto vs no-lto: -23.3%
  Delta exceeds the worst arm spread (7.9%), so it is real.
```

### Correction: the battery hypothesis was wrong

An earlier version of this entry claimed the battery measurement was inflated,
reasoning that a lower clock makes memory stalls a smaller share of time and so
exaggerates the benefit of removing compute. **That was wrong.** Re-measured on
AC with the browser closed:

| Condition | Threads | no-LTO | with-LTO | Delta | Worst spread |
|---|---|---|---|---|---|
| Battery, browser playing video | 1 | 9.941s | 7.628s | −23.3% | 7.9% |
| Battery, browser playing video | 1 | 9.812s | 7.401s | −24.6% | 10.9% |
| **AC, quiet** | **1** | **5.169s** | **3.973s** | **−23.2%** | **1.8%** |
| **AC, quiet** | **14** | **2.758s** | **2.240s** | **−18.8%** | **2.6%** |

**Power state does not change the ratio.** It changes *precision*: spreads fell
from 8–11% to 0.4–2.6%, and the re-recorded baseline now reproduces
`st_engine_seconds` to the fourth decimal. Plugging in does not buy a different
answer, it buys a *resolvable* one.

**Thread count is what explains the discrepancy.** The earlier ad-hoc −18.1%
that prompted the wrong hypothesis was a 200-game, 14-thread run, compared
against `ab.sh`'s 50-game single-threaded default. It matches the −18.8%
measured properly at 14 threads. The number was right; the attribution was not.

The real finding: **multi-threaded captures 81% of the single-threaded win.**
The parallel run is closer to memory-bandwidth-bound, so removing compute pays
less there. Both numbers are worth recording — ST because it is the precise
instrument, MT because training runs multi-threaded.

**Headline figures for this change: −23.2% single-threaded, −18.8% at 14
threads.**

### Original caveat, retained: taken on battery, with a browser playing video

Both A/B runs were made with the laptop **on battery** (`ac=0`, `powersave`
governor) and Brave playing Netflix. Max confirmed this after the fact.

Replicated to check stability:

| Run | Reps | no-LTO median | with-LTO median | Delta | Worst spread |
|---|---|---|---|---|---|
| 1 | 7 | 9.941s | 7.628s | **−23.3%** | 7.9% |
| 2 | 9 | 9.812s | 7.401s | **−24.6%** | 10.9% |

Reproducible to ~1.3 points, so the result is not an artifact of one unlucky
sample. Interleaving handles the slow drift (battery depleting, machine
heating) and the median over 7–9 reps absorbs the video-decode bursts, which is
what the 8–11% spreads are.

**But the ratio itself is probably inflated.** Lower clock means each
instruction takes longer in wall time while DRAM latency stays fixed in
nanoseconds, so at low clock memory stalls are a *smaller* share of total time
and compute is a *larger* one. LTO removes compute — call setup, jump, return —
so its benefit should look bigger on battery than on AC.

The earlier ad-hoc flag comparison, taken while the machine was on AC (3 reps,
not interleaved) gave `3.067s → 2.512s` = **−18.1%**. That is the predicted
direction and roughly the predicted magnitude.

**Best estimate on AC: −18% to −20%.** The headline −23.3% should be read as an
upper bound. Re-measure on AC with the browser quiet:

```
cd bench && ./ab.sh no-lto "-std=c++17 -O3 -fopenmp -DNDEBUG" \
                    with-lto "-std=c++17 -O3 -fopenmp -DNDEBUG -flto" 7
```

None of this touches the counters below: they come from simulation and have no
dependence on clock speed or contention. The *mechanism* is established
regardless; only its wall-clock translation is uncertain.

Exact counters, from simulation and therefore unaffected by clock speed
(`bench/results/before-lto.tsv`, `bench/results/after-lto.tsv`):

| Counter | no-LTO | with-LTO | Δ |
|---|---|---|---|
| `exact_instructions` | 1,519,890,125 | 1,022,704,411 | **−32.71%** |
| `exact_branches` | 138,947,465 | 128,087,654 | −7.82% |
| `exact_branch_mispredicts` | 11,636,232 | 11,896,444 | **+2.24%** |
| `exact_d1_read_miss` | 2,504,600 | 2,500,033 | −0.18% |
| `exact_alloc_blocks` | 87,010 | 87,010 | same |

`digest_game` and `digest_engine` both **unchanged**. This matters more than
usual: LTO optimizes more aggressively and can expose latent undefined behaviour
that survived while the compiler could not see across files. The digests holding
says there was none to expose here.

### The interesting part: the two tables disagree

**Instructions fell 32.7%; time fell 23.3%.** The gap is the finding.

The instructions LTO removed are call setup, jump, return, and the reloads
forced by an opaque call boundary. Those are the *cheapest* instructions in the
program — perfectly predicted, fully pipelined, often dual-issued. Removing a
third of the instruction count bought only a quarter of the time because the
removed third was below-average cost.

Branch mispredicts actually rose 2.24%: inlining changed code layout and the
predictor does slightly worse on the result. It was still a clear net win.

This is a useful calibration for the bitboard predictions in entry 02, which
claim the **opposite** — that time will improve *more* than instructions,
because that change removes expensive mispredicting branches rather than cheap
calls. LTO is now a worked example of the mechanism in reverse.

## The harness hole

Sanity-checking the first (non-interleaved) LTO measurement, the *unchanged*
default build measured `st_engine_seconds` 7.22 against 4.08 in
`bench/results/baseline.tsv` forty minutes earlier. **77% slower, same code,
same flags.**

Cause: `/sys/class/power_supply/AC*/online` was `0` — **the laptop was on
battery**. With `intel_pstate` on the `powersave` governor, battery caps turbo
hard on a 12700H. A browser was also consuming roughly 80% of a core across
three processes.

Package temperature was 59 °C, so this was not thermal throttling; checking that
first was the wrong guess.

Nothing in any record captured power state or load, so the drift was invisible.
Every cross-session comparison in `bench/results/` was exposed to it — including
the baseline that entry 02's predictions are anchored to.

**Fixes** (committed separately, before the optimization itself):

- `run_suite.sh` records `#ac_power`, `#governor`, `#loadavg`.
- **`ab.sh`** builds two arms once into separate directories and **alternates**
  runs A,B,A,B,…, so slow drift lands on both arms equally instead of entirely
  on the comparison. It reports each arm's spread and refuses to endorse a delta
  smaller than the worst arm's own spread.

**Revised guidance:** `run_suite.sh` for recording a labelled point and for the
exact counters; `ab.sh` for deciding whether one build is faster than another.
Absolute timings are valid within a session only. Counters are simulated and
stay comparable across sessions.

## What is not verified

**The real build was never run.** The system Python has neither cython nor
tensorflow, so `python3 setup.py build_ext --inplace` cannot execute here. The
flags are syntactically valid Python and the C++ builds fine with `-flto`, but
the actual extension module has not been produced.

The specific risk with LTO in a Python extension is that aggressive
whole-program optimization drops or mangles the exported `PyInit_corintho`
entry point. That was tested directly: a stand-in `PyInit_corintho` with
`visibility("default")` was compiled together with all seven engine sources at
`-flto -fPIC -shared`, and the symbol is present in the dynamic table
(`nm -D` shows `T PyInit_corintho`) and resolves through `dlopen`.

That retires the main concern but is not the same as a successful real build.
**Before the next training run, confirm `scripts/bash/build.sh` completes and
the module imports.** If LTO causes trouble there, removing the two flags is a
complete revert.

Build time and link-time memory both rise. For six sources it is seconds.
`-flto=auto` would parallelize the link if it ever becomes annoying; plain
`-flto` was chosen for compatibility.

## Carried forward

From `2026-09-20-02`:

**Done since**

- ~~Add `-flto` to `setup.py`~~ — done, this entry, measured at −23.3%.

**Still live**

- **Move the `Node` accessors into `node.h`.** Still worth doing: it makes the
  win robust without depending on LTO, and it is a good test of understanding —
  the prediction is that on an LTO build it changes *nothing*, because LTO
  already inlines them. If it does change something, our model is wrong.
- **The spreadsheet's last column is still unidentified.** Unchanged and still
  the highest-value unknown in the project.
- **The stub evaluator's tree shape is uncalibrated** (18.2 turns/game vs 28.4).
- **`to_eval_` oversizing** gates local runs at realistic scale.
- **Distinct-position instrumentation** not built.
- **S3** not set up.
- **New:** the standing `baseline.tsv` was recorded on AC and cannot be compared
  against records taken on battery. Re-record it on AC before using it as the
  anchor for the bitboard predictions.

## Next steps

1. Verify `build.sh` works with LTO on a machine that has cython.
2. Move the `Node` accessors into `node.h` — and check the prediction that it
   changes nothing on an LTO build.
3. Build the `verify` mode and `getLegalMovesReference` safety net.
4. Bitboard stage 1.
