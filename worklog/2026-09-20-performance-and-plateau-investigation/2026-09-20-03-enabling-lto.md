# 2026-09-20 — Enabling LTO, and a hole in the harness

Branch: `perf/training-overhaul`. First actual optimization on this branch.

## TL;DR

- **`-flto` added to `corintho_ai/python/setup.py`.** Worth a measured **23%**
  of engine time, one line, no source changes, digests unchanged.
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
