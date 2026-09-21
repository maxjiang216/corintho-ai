# to_eval_ was 100x too big, and it was resident

*2026-09-21 — branch `perf/engine-optimizations`, commit `abf45e1`*

## The buffer

`to_eval_` stages one batch of neural network inputs. During a turn MCTS
descends the tree repeatedly; each time it reaches a node needing evaluation it
writes that position's 70 floats into `to_eval_` and records the node in
`searched_`. When the batch is full, `SelfPlayer::writeRequests` copies the
filled prefix out:

```cpp
int32_t count = kGameStateSize * players_[to_play_].num_requests();
std::copy(to_eval_.get(), to_eval_.get() + count, game_states);
```

Python runs the network, `receiveEval` propagates results up the tree, and the
buffer is **refilled from offset 0**.

## The defect

```cpp
to_eval_{std::make_unique<float[]>(kGameStateSize * max_searches)}   // selfplayer.cpp:24
```

`max_searches` (1600) is the searches in a whole *turn*. The buffer drains every
batch, so the live count is `searches_per_eval` (16).

| | bytes |
|---|---|
| one position | 70 floats x 4 = 280 |
| needed | 280 x 16 = 4,480 (4.4 KiB) |
| allocated | 280 x 1600 = 448,000 (437.5 KiB) |

**100x**, exactly `max_searches / searches_per_eval`.

### Why it was resident, not merely reserved

This is what turns a curiosity into a blocker. `std::make_unique<float[]>(n)`
**value-initializes** — it zero-fills the array. Every page is touched at
construction, so the OS physically backs all of it. Had the tail been left
untouched, Linux would never have committed those pages.

Measured, single-threaded:

| games | peak RSS | `to_eval_` share | actually used |
|---|---|---|---|
| 200 | 127 MB | 85 MB | 0.85 MB |
| 800 | 487 MB | 342 MB | 3.42 MB |

About **70% of the process** was this buffer, 99% of it never touched again
after being zeroed.

## Why it blocked the plan

The waste scales with games *and* with `max_searches`, and the plan was to raise
both.

| run | `to_eval_` before | after |
|---|---|---|
| 3,000 games | 1.25 GiB | 12.8 MiB |
| 25,000 games | **10.4 GiB** | 107 MiB |

A 25,000-game run was impossible on a 15 GB machine: 10.4 GiB gone before a
single tree node. Raising `max_searches` above 1600 made it worse linearly —
the waste grew exactly where the headroom was wanted.

## The fix

One variable, but two sites, and the second is not a pure substitution.

```cpp
// selfplayer.cpp
to_eval_{std::make_unique<float[]>(kGameStateSize * searches_per_eval)}

// match.cpp -- max over searches_per_eval, NOT over max_searches
to_eval_{std::make_unique<float[]>(
    kGameStateSize *
    std::max(player1.searches_per_eval, player2.searches_per_eval))}
```

`Match`'s two players may be configured differently, and it is the batch size
that bounds the writes, so the max must be taken over the right field.

`SelfPlayer` hands **both** `TrainMC` players the same pointer and both fill
from offset 0. That is safe because only the player to move searches. Sizing to
`searches_per_eval` preserves that assumption exactly rather than changing it.

## Evidence the bound is right

Five independent checks, because shrinking a buffer is the kind of change where
being wrong is a silent heap corruption:

1. **All four write sites bounded.** Three write at offset 0, each only when
   `searched_` is empty (first iteration, new root from opponent). The fourth
   writes at `searched_.size() * kGameStateSize` inside a loop guarded by
   `searched_.size() < searches_per_eval_`.
2. **Read side agrees.** `writeRequests` copies `num_requests()` ==
   `searched_.size()` entries — same bound.
3. **`dockermc.cpp:16` already did it correctly**: `searches_per_eval *
   kGameStateSize`. One of the three allocation sites was always right, which is
   what makes this an oversight rather than a deliberate safety margin.
4. **Asserts enabled.** The normal build is `-DNDEBUG`, so every
   `assert(searched_.size() <= searches_per_eval_)` is compiled out and proves
   nothing in production. Rebuilt with them on: 914,223 requests over 40 games,
   no failure.
5. **valgrind memcheck**, 3-game run: `0 errors from 0 contexts`, no invalid
   write or read.

Point 5 is the gate that actually matters here. `digest_game` and
`digest_engine` are unchanged, but a digest cannot catch an out-of-bounds store
that lands in another allocation — it would change behaviour somewhere else
entirely, or not at all until it did. Memcheck can.

## Results

Peak RSS, single-threaded:

| games | old | new | factor |
|---|---|---|---|
| 200 | 127 MB | 48 MB | 2.6x |
| 800 | 487 MB | 174 MB | 2.8x |
| 2000 | 1069 MB | 357 MB | 3.0x |

Setup time at 200 games: **0.0260 s -> 0.0010 s (-96%)**, which is the zero-fill
disappearing. It scales with games x `max_searches`, so at 3,000 games it is
roughly 0.39 s -> 0.015 s. Small absolutely, but free.

Engine time measured **-1.2%** on min-of-3 at 200 games. **Not established** —
three repetitions with a browser running cannot support a 1% claim, and it is
not claimed. Recorded here only so a future session does not re-derive it and
mistake it for a result. If someone wants the real number, it needs the full
interleaved treatment on a quiet machine; the plausible mechanism is fewer
resident pages and so less TLB and page-fault pressure across the parallel loop,
which would show up multi-threaded rather than single-threaded.

## Note on the earlier estimate

Entries 05-08 carried this as "448 KB/game vs 4.5 KB needed". That was right.
What those entries did not say, and what actually mattered, is that the excess
was **resident rather than reserved**. Without that, the fix is a tidiness
change; with it, it is the difference between a 25,000-game run being possible
and not.

## Status

- [x] `vector<bool>` race (06)
- [x] `doMove` (07)
- [x] `writeGameState` (08)
- [x] `to_eval_` sizing — 3x peak RSS
- [ ] `setup.py` has no `-march`; decide deliberately (see 08)
- [ ] `scripts/bash/build.sh` never build-verified with `-flto`
- [ ] Strength impact of the line-breaking fix — needs a fixed-weights match
- [ ] `web/engine.js`, `web/line_breakers.js` still carry every line-breaking defect
- [ ] **The spreadsheet's last column**, open since entry 01
