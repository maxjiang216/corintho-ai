# 10 — Solver move ordering: statistics, and why only experiments count

2026-09-26. Commits `4408971` (history heuristic), `d3907bd` (statistics
mode). Data: `data/solver/ordering-stats-p24.txt`,
`data/solver/bench-9-history-p27.tsv`.

## History heuristic (adopted)

Quiet cutoffs are credited P^2 (per side to move and move; P = 2 x
reserves + occupied spaces); quiet moves are ordered by it within each rank
group (stack moves, base, column, capital). P <= 24: nodes 0.86x; P <= 27:
0.80x (summed time 9.62 -> 7.99 s). History overriding the rank groups:
1.36x / 1.47x more nodes.

## The developer's idea: find what correlates with a good ordering

Random permutations would be noisy; instead the solver records every move
it tries (`-DSOLVER_STATS`, `bench/solver_stats.py`): 3.49M moves at P <=
24.

- 89.2% of cutoffs come from the first move tried (chess engines aim for
  ~90%); 79% of all cutoffs are immediate wins found by classification.
- Cutoff rate by feature (quiet): stack moves 24.3%, base 13.2%, column
  11.0%, capital 4.9%; onto an existing stack 16.1% vs empty 7.1%;
  column-topped stack moves 49.4% vs capital-topped 19.3%; destination
  centre 14.2% vs edge/corner ~10.5%; distance to the frozen square: none.
  Line-making: more opponent replies, higher cutoff rate (1 reply 19.1% ...
  5 replies 32.2%).
- With each move's subtree cost, cutoffs per 1000 positions spent: line
  1 reply 110.6 falling to 11.0 at 5; quiet stack 25.0, capital 10.0,
  column 5.5, base 3.2; onto a stack 20.7 vs 5.6; capital-topped stack
  moves 32.7 vs column-topped 17.1; height 2 43.3 vs 23.6.

## Controlled tests of what the statistics suggested (all gated, 6 threads)

| change | P <= 24 nodes | P <= 27 nodes |
|---|---|---|
| line-making: most replies first | 1.11x | 1.14x |
| line-making: unordered | 1.03x | 1.04x |
| placements capital, column, base (efficiency order) | 1.56x | 1.67x (2 fewer solved) |
| quiet: onto a stack first within the group | 1.006x | 1.005x |
| stack moves: capital-topped, taller first | 1.008x | 1.010x |
| both | 1.014x | 1.018x |

None beat the current order. The statistics are observational: a move's
measured cost and cutoff rate depend on when the current order tries it
(capital placements come last, mostly in positions where everything else
failed, with narrow windows and a warm table, so they look cheap). They
suggest candidates; only reordering and counting nodes decides. An
unbiased dataset would need an "oracle" run (search every move of sampled
positions with the same window). The current order (line-making fewest
replies first; stack moves, base, column, capital; history within groups)
is a local optimum for these simple features.
