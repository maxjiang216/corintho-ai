// First/second player balance, with no neural network involved.
//
// Worklog entry 10 found that the second player wins ~74% of training
// self-play games across all 95 recorded generations. Entry 11 uses this to
// ask where that comes from. It plays either uniformly random games (sims 0)
// or plain UCT with random rollouts (sims > 0), so it needs no trained model
// and no Python.
//
//   ./build/parity <games> <sims> [seed]      sims 0 = pure random play
//
// To compare rule sets across versions, build this file against another tree
// rather than against getLegalMovesReference. game_reference.cpp is NOT the
// pre-fix implementation -- it was re-frozen by the fix commit (5b2afb6) so
// that bench/verify keeps passing, so comparing against it compares the fixed
// rules with themselves. Use a worktree instead:
//
//   git worktree add --detach /tmp/prefix 5b2afb6^
//   then compile this file with -I/tmp/prefix/corintho_ai/cpp/include and
//   /tmp/prefix/corintho_ai/cpp/src/{util,move,game}.cpp
#include <cstdint>
#include <bitset>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "game.h"
#include "util.h"

static inline uint64_t splitmix64(uint64_t &x) {
  uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// Result from the point of view of the player to move at the terminal node:
// with lines on the board and no legal moves, that player has lost.
// Returns score for the player to move at the ROOT of the rollout.
static double rollout(Game g, uint64_t &rng) {
  std::bitset<kNumMoves> legal;
  int ply = 0;
  for (;;) {
    bool lines = g.getLegalMoves(legal);
    size_t c = legal.count();
    if (c == 0) {
      if (!lines) return 0.5;
      // Player to move lost. ply even => that is the rollout's root player.
      return (ply % 2 == 0) ? 0.0 : 1.0;
    }
    size_t pick = splitmix64(rng) % c;
    int32_t mv = 0;
    for (int32_t i = 0; i < kNumMoves; ++i)
      if (legal[i]) { if (pick == 0) { mv = i; break; } --pick; }
    g.doMove(mv);
    ++ply;
  }
}

struct N {
  Game game;
  std::vector<int32_t> moves;
  std::vector<int32_t> child;   // index into pool, -1 = unexpanded
  double w = 0; int32_t n = 0;
  bool terminal = false; double tscore = 0.5;
};

static int32_t makeNode(std::vector<N> &pool, const Game &g) {
  pool.emplace_back();
  N &node = pool.back();
  node.game = g;
  std::bitset<kNumMoves> legal;
  bool lines = g.getLegalMoves(legal);
  if (legal.count() == 0) {
    node.terminal = true;
    node.tscore = lines ? 0.0 : 0.5;   // player to move loses, or draw
    return (int32_t)pool.size() - 1;
  }
  for (int32_t i = 0; i < kNumMoves; ++i)
    if (legal[i]) node.moves.push_back(i);
  node.child.assign(node.moves.size(), -1);
  return (int32_t)pool.size() - 1;
}

// Returns score for the player to move at `idx`.
static double search(std::vector<N> &pool, int32_t idx, uint64_t &rng, double c_puct) {
  N *node = &pool[idx];
  if (node->terminal) { node->n++; node->w += node->tscore; return node->tscore; }
  // pick child by UCT
  int32_t best = -1; double bestv = -1e18;
  double logN = std::log((double)std::max(1, node->n) + 1.0);
  for (size_t i = 0; i < node->moves.size(); ++i) {
    int32_t ci = node->child[i];
    double v;
    if (ci < 0) v = 1e17 + (double)(splitmix64(rng) % 1000);  // unvisited first
    else {
      const N &cn = pool[ci];
      double q = cn.n ? 1.0 - (cn.w / cn.n) : 0.5;   // child's value is opponent's
      v = q + c_puct * std::sqrt(logN / (1.0 + cn.n));
    }
    if (v > bestv) { bestv = v; best = (int32_t)i; }
  }
  double score;
  if (node->child[best] < 0) {
    Game g = node->game; g.doMove(node->moves[best]);
    int32_t ci = makeNode(pool, g);
    node = &pool[idx];                 // pool may have reallocated
    node->child[best] = ci;
    N &cn = pool[ci];
    double s = cn.terminal ? cn.tscore : rollout(cn.game, rng);
    cn.n++; cn.w += s;
    score = 1.0 - s;
  } else {
    score = 1.0 - search(pool, node->child[best], rng, c_puct);
  }
  node = &pool[idx];
  node->n++; node->w += score;
  return score;
}

int main(int argc, char **argv) {
  const long long games = argc > 1 ? atoll(argv[1]) : 400;
  const int32_t sims = argc > 2 ? atoi(argv[2]) : 200;
  uint64_t rng = argc > 3 ? strtoull(argv[3], nullptr, 10) : 7;
  const double c_puct = 1.4;
  long long p1 = 0, p2 = 0, dr = 0, plies = 0;
  std::bitset<kNumMoves> legal;
  for (long long g = 0; g < games; ++g) {
    Game game; int ply = 0;
    for (;;) {
      bool lines = game.getLegalMoves(legal);
      if (legal.count() == 0) {
        plies += ply;
        if (!lines) ++dr; else if (ply % 2 == 0) ++p2; else ++p1;
        break;
      }
      int32_t chosen;
      if (sims == 0) {
        // Pure random play. Must be handled separately: with no simulations
        // there are no visit counts, so the most-visited rule below would
        // always pick move 0 and replay one deterministic game.
        size_t c = legal.count();
        size_t pick = splitmix64(rng) % c;
        chosen = 0;
        for (int32_t i = 0; i < kNumMoves; ++i)
          if (legal[i]) { if (pick == 0) { chosen = i; break; } --pick; }
      } else {
        std::vector<N> pool; pool.reserve(sims * 2 + 8);
        int32_t root = makeNode(pool, game);
        for (int32_t s = 0; s < sims; ++s) search(pool, root, rng, c_puct);
        int32_t bi = 0, bn = -1;
        for (size_t i = 0; i < pool[root].moves.size(); ++i) {
          int32_t ci = pool[root].child[i];
          int32_t v = ci < 0 ? 0 : pool[ci].n;
          if (v > bn) { bn = v; bi = (int32_t)i; }
        }
        chosen = pool[root].moves[bi];
      }
      game.doMove(chosen);
      ++ply;
    }
  }
  double n = (double)games, P1 = p1/n, P2 = p2/n, D = dr/n, S = P2 + 0.5*D;
  double var = P2*(1-S)*(1-S) + D*(0.5-S)*(0.5-S) + P1*S*S;
  printf("sims %5d   P1 %.4f  draw %.4f  P2 %.4f   P2score %.4f +- %.4f   plies %.2f\n",
         sims, P1, D, P2, S, sqrt(var/n), plies/n);
  return 0;
}
