# 17 — Where repeated evaluations come from; one tree for both self-play sides

2026-09-26. Commits `d157f21` (root-visit limit), `5f0652d`
(`--shared-tree`). Tool `pipeline/arch/dup_split.py`. Data:
`data/transpositions/repeat-split.txt`.

## The question

The developer wanted to revisit transpositions (graph search, parked in
entry 07). Entry 07 found 32.6% of network rows repeat a position already
evaluated in the same game, but did not split out where the first
evaluation happened. Graph search only removes one kind: transpositions
within one search.

## The split

`CORINTHO_DUP_DUMP` (entry 07) records each row's game and search root
horizon P. P falls every move, so (game, root P) identifies a search, and
the parity of its rank within the game identifies the player (the two
sides alternate). `arch/dup_split.py` classifies each row. 500 self-play
games, loop-1 gen 15 (trt16), `--solve-p 27`, seed 7, 15.1M rows:

| first evaluated | exact | up to symmetry |
|---|---|---|
| nowhere (new to the game) | 69.1% | 63.2% |
| earlier in the same search (a transposition) | 6.0% | 12.0% |
| in the same player's earlier search | 4.2% | 4.6% |
| only in the other player's tree | **20.6%** | 20.2% |

Transpositions within a search grow with distance from the root (5.1% of
rows at P drop 3-4, 7.5% at 10+). The largest part is not transpositions:
self-play kept **a tree per player** (`SelfPlayer::players_[2]`) although
both use the same network, so each side re-evaluated positions the other
side's tree already held.

## What the search budget means

Each move gets `max_searches` (1600) **new** searches: `searches_done_`
resets to 0 when the root moves down, and visits kept from earlier
searches add on top (a root ends at V + 1600). So removing repeats does not
reduce network calls; the calls go to new positions and the tree grows.

The alternative, a budget of total root visits (KataGo-style, stop at
1600 including the V inherited), would cut rows only by searching less
per move. The developer: "that means it's doing fewer searches. let's not
touch that for now."

## One tree for both sides

The developer asked whether sharing is unfair. In self-play it is not:
both sides get the same (each inherits the other's search from one ply
back, where today each inherits its own from two plies back), the network
is the same, the game has no hidden information, and self-play produces
data rather than deciding a contest. AlphaGo Zero's self-play also uses
one tree. Matches between different networks keep a tree per network.

`--shared-tree 1` (train mode; off by default): both sides use
`players_[0]` through `SelfPlayer::tree(p)`. `chooseMove` already moves the
root down to the chosen child (keeping its subtree), so the other side
continues searching from it; `receiveOpponentMove` is skipped. Everything
else (solver endings, samples, logs) reads the tree through `tree()`.

Same 500 games (seed 7), `--shared-tree 1`:

| | two trees | shared |
|---|---|---|
| network rows | 15.12M | 15.05M |
| rows new to the game | 69.1% | **86.9%** |
| repeats of the other side's tree | 20.6% | 4.5% |
| transpositions within a search | 6.0% | 5.6% |
| root visits at move choice, mean | 2,699 | 4,932 |
| root visits, p99 / max | 6,258 / 7,722 | 13,977 / 16,570 |
| first-player score | 0.503 | 0.500 |

The remaining 4.5% are positions in branches the other side searched but
did not play (dropped with the siblings of the chosen move).

Not yet measured: whether the bigger trees make better training data
(sharper policy targets, from ~1.8x the root visits). That needs a paired
training run, like solve-27 vs solve-0 (entry 16).

## A limit on root visits

`Node::visits_` is `int16_t` and nothing stopped a search at 32,767.
Shared trees carry more visits over: with the solver off (longer games),
root visits reached 20,380 (100 logged games of 500). The search now also
stops once the root has 32,000 visits (`TrainMC::budgetLeft`, commit
`d157f21`). Checks:
- never reached today, so the 300-game sample digest is unchanged
  (2a77d17cf2451dbb, `--solve-p 27`);
- a throwaway build with the limit at 3,000 stopped roots at exactly
  3,000 and every game finished;
- unit tests pass (50, GCC and clang).

## Still open

- Graph search for the 6% (12% up to symmetry, where the network is only
  approximately invariant, entry 07) of rows that are transpositions
  within a search. A large change (a node has one parent and a child list;
  backup through several parents).
- The shared tree's effect on training (paired run).
