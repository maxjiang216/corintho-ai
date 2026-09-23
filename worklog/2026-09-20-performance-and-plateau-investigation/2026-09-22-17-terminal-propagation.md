# 17 — Terminal-result propagation: drawn positions were deduced as losses

Correctness fix, not performance. Max asked to look at the correctness and
performance of W/L/D propagation. Performance turned out to be negligible. The
correctness bug is the one audit item CPP-1 (entry 01) had already listed, and
it was still unfixed.

## The rule

Results are from the point of view of the player to move:

- **won** if any move leads to a lost position;
- **lost** if every move leads to a won position;
- **drawn** if every move leads to a won or drawn position, and at least one to
  a drawn one;
- otherwise unknown.

`TrainMC::propagateTerminal` applies this upward from each new terminal leaf.
It stops at the first ancestor that stays unknown, because nothing above that
can change.

## The bug

In the loop over a node's children, the draw test read the node being decided
instead of the child:

```cpp
if (cur->drawn())        // was: cur is the parent, still kResultNone here
if (cur_child->drawn())  // now
```

`has_draw` could therefore never become true. Every fully resolved node with no
winning move was marked `kDeducedLoss`, including those with a drawing move. The
next iteration then marked its parent `kDeducedWin`, and so on upward.
`kDeducedDraw` was never produced (`ded_draw=0` below). `web/trainmc.js`
already had the correct test. This looks like a typo in the C++.

## Why training labels were not directly affected

`SelfPlayer` always plays a game to its terminal position (`endGame` asserts
it), and value labels come from that real outcome. The damage was elsewhere:

- **Move choice at a wrongly known root.** A false `lost` root stops searching
  and plays the most-visited move, which may be a known loss. A false `won` root
  plays its "winning" move, actually a draw or unknown, with no further search.
- **The policy target.** A known root writes a one-hot `prob_sample`.
- **Search pruning.** Known won and lost children are skipped by `chooseNext`,
  so a false result removes a move from search, and from `chooseMoveNormal`'s
  candidates.

Match play (`match.cpp`) uses the same `TrainMC`, so tournaments and ratings
were affected too.

## How it was checked: a full-tree audit

`data/terminal-propagation/audit-instrumentation.patch` is an instrumented
`trainmc.cpp`, not committed to the engine. The `TERM_FIX=0|1` switch selects
the old or the fixed test.

At every `chooseMove`, the audit recursively recomputes every node's result
from the terminal leaves using the rule above, and compares it with the stored
result:

- **wrong**: the stored result is known but differs;
- **missed**: stored unknown, but the recomputed result is known. This checks
  that propagation is complete, not just sound.

Because the audit derives everything from the leaves, a clean audit shows the
tree's results are exactly what game theory gives from what has been searched.

Results, 200 games, 1600 searches, 8 threads (full output in
`data/terminal-propagation/audit-runs.txt`):

| | stub, old | stub, fixed | model_93, old | model_93, fixed |
|---|---|---|---|---|
| move decisions audited | 3,732 | 3,741 | 5,630 | 5,632 |
| **root result wrong** | **33** | 0 | **57** | 0 |
| roots with a known result | 815 | 821 | 1,151 | 1,139 |
| wrong interior nodes (summed over audits) | 755 | 0 | 8,209 | 0 |
| missed deductions | 0 | 0 | 0 | 0 |
| deduced draws | 0 | 395 | 0 | 5,170 |

With the real network, the 57 wrong roots break down as:

| stored | true | count |
|---|---|---|
| lost | still unknown | 32 |
| lost | draw | 5 |
| won | still unknown | 19 |
| won | draw | 1 |

So 1.0% of move decisions, and 5% of decisions at a known root, were made from
a wrong result. A false result can be "still unknown" and not only "draw"
because an error propagates: a false loss makes its parent a false win, whose
own parent may then be a false loss, and so on.

Assert-enabled runs of the fixed build (stub 200 games, real network 100
games, different seeds) were clean with zero wrong. Game length did not change:
28.15 → 28.16 turns per game with the real network.

## The golden engine digest cannot see this

After the fix `golden engine` is still `d7581196f9d30591`. Its workload (20
games, 200 searches) never deduces a draw. So an unchanged digest here is
**not** evidence of unchanged behaviour: the 1600-search runs above play
different games. Any future change to terminal handling needs the audit, not
the digest.

## Performance: nothing to do

`propagateTerminal` is 0.02% of instructions: 139,639 Ir, 979 calls per 3 stub
games (`mirror-lines-ir.txt`), inlined into `search`. With the real network it
examines ~2.7 children per call. Not worth touching.

## Noted, not changed

Both are behaviour choices that would need a strength test, not a benchmark.

1. **Proven draws stop gaining visits.** Terminal draws are `all_visited` from
   construction. A deduced-draw node has only known children, so one descent
   ends in `kNone`, marks it `all_visited`, and it is skipped from then on. The
   `kDrawnChild` score (`P·c·√N`, with no `/(n+1)`) therefore buys at most that
   one wasted descent. At move choice a draw is scored as 0 but compared by
   visits, so a proven draw with frozen visits can lose to a more-visited move
   whose value is worse than 0. Before the fix this path was unreachable (no
   `kDeducedDraw` existed). It is reachable now.
2. **Proven results do not update ancestors' values.** A parent's accumulated
   evaluation still averages searches of a child now proven lost for it. An
   MCTS-Solver-style exact backup would fix that. It could help endgame
   strength, but would need an Elo test.

The comment in `search` says the terminal node "may be a drawn node that has
been searched before". It cannot be: terminal draws are skipped by the
`all_visited` flag. Left alone.
