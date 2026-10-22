# Quiet-move ordering from unbiased labels

Worklog 2026-09-25-nn-architectures, entry 24.

Statistics gathered under the solver's own move order are biased: the move
that cuts off is, by construction, tried after the moves that failed. So
these tools label every quiet move of sampled positions, searched with the
same window.

1. `oracle_labels.patch` adds the labelling to `Solver::search`: at one in
   eight decisions with P >= 16 where only quiet moves remain, it searches
   every quiet child and writes each one's features, whether it cuts and
   its cost to `$CAND_OUT`. Results stay identical; nodes rise.

       git apply bench/ordering/oracle_labels.patch   # on a scratch copy
       CAND_OUT=oracle.bin build/solver_bench results/solver-positions.bin 27 2000000000 1

2. `fit_oracle.py` (run next to `oracle.bin`) scores any order exactly
   offline: per held-out decision, the cost of the moves ahead of its first
   cutter plus that cutter. It fits tiny decision trees and logistic models
   for P(cut). `fit_ratio.py` ranks by P(cut) over a tree's predicted
   cost.

What it found: the current order is beaten offline by P(cut)/cost trees
(0.86x) and by one rule. Placements that would leave the mover one or no
pieces of that type go last. That rule alone took P <= 27 from 29.8M to
24.7M nodes in the real solver.
