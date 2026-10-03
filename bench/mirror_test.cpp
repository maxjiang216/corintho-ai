// Does a 180-degree mirror strategy work for the second player?
//
// On a 4x4 board, rotation by 180 degrees pairs every square with a different,
// non-adjacent square. The board has no colours and reserves start equal, so
// after the second player mirrors, the position is point-symmetric again and
// the mirrored reply is always available -- unless the first player made a
// line, which must be broken instead.
//
// P1: exact win/draw/loss alpha-beta to depth D1 (0 = uniform random), heuristic
//     value 0 beyond the horizon, random tie-breaks among equal best moves.
// P2: the mirror move when it is legal (legal moves already account for the
//     line-breaking rule); otherwise the same search as P1 at depth D2.
//
// Usage: mirror_test [games] [D1] [D2] [threads]

#include <omp.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "game.h"
#include "move.h"
#include "util.h"

namespace {

int g_mirror[kNumMoves];

void initMirror() {
  auto rot = [](int s) { return 15 - s; };  // (r,c) -> (3-r,3-c)
  for (int m = 0; m < kNumMoves; ++m) {
    const MoveInfo &mi = kMoveTable[m];
    g_mirror[m] = -1;
    for (int n = 0; n < kNumMoves; ++n) {
      const MoveInfo &ni = kMoveTable[n];
      if (ni.is_place != mi.is_place || ni.piece != mi.piece) continue;
      if (ni.to != rot(mi.to)) continue;
      if (!mi.is_place && ni.from != rot(mi.from)) continue;
      g_mirror[m] = n;
    }
  }
}

// Exact value for side to move within depth: +1 win, -1 loss, 0 unknown/draw.
int search(const Game &g, int depth, int alpha, int beta) {
  MoveMask m;
  const bool lines = g.getLegalMoves(m);
  if (!m.any()) return lines ? -1 : 0;
  if (depth == 0) return 0;
  int best = -2;
  bool cut = false;
  forEachMove(m, [&](int mv) {
    if (cut) return;
    Game c = g;
    c.doMove(mv);
    const int s = -search(c, depth - 1, -beta, -alpha);
    best = std::max(best, s);
    alpha = std::max(alpha, s);
    if (alpha >= beta) cut = true;
  });
  return best;
}

int chooseSearch(const Game &g, const MoveMask &m, int depth,
                 std::mt19937_64 &rng) {
  std::vector<int> moves;
  forEachMove(m, [&](int mv) { moves.push_back(mv); });
  if (depth == 0)
    return moves[std::uniform_int_distribution<size_t>(0, moves.size() - 1)(rng)];
  std::shuffle(moves.begin(), moves.end(), rng);
  int best = -2, choice = moves[0];
  for (int mv : moves) {
    Game c = g;
    c.doMove(mv);
    const int s = -search(c, depth - 1, -1, 1);
    if (s > best) {
      best = s;
      choice = mv;
      if (best == 1) break;
    }
  }
  return choice;
}

struct Out {
  int winner;  // 0 P1, 1 P2, -1 draw
  int plies;
  int p2_moves, p2_mirrors, first_deviation;
};

Out play(int d1, int d2, uint64_t seed) {
  std::mt19937_64 rng(seed);
  Game g;
  Out o{-1, 0, 0, 0, -1};
  int last = -1;
  for (int ply = 0;; ++ply) {
    MoveMask m;
    const bool lines = g.getLegalMoves(m);
    if (!m.any()) {
      o.plies = ply;
      o.winner = lines ? 1 - ply % 2 : -1;
      return o;
    }
    int mv;
    if (ply % 2 == 0) {
      mv = chooseSearch(g, m, d1, rng);
    } else {
      ++o.p2_moves;
      const int mir = g_mirror[last];
      if (m.test(mir)) {
        mv = mir;
        ++o.p2_mirrors;
      } else {
        if (o.first_deviation < 0) o.first_deviation = ply;
        mv = chooseSearch(g, m, d2, rng);
      }
    }
    g.doMove(mv);
    last = mv;
  }
}

}  // namespace

int main(int argc, char **argv) {
  const int games = argc > 1 ? std::atoi(argv[1]) : 1000;
  const int d1 = argc > 2 ? std::atoi(argv[2]) : 4;
  const int d2 = argc > 3 ? std::atoi(argv[3]) : 4;
  const int threads = argc > 4 ? std::atoi(argv[4]) : 20;
  omp_set_num_threads(threads);
  initMirror();
  for (int m = 0; m < kNumMoves; ++m)
    if (g_mirror[m] < 0) {
      std::fprintf(stderr, "no mirror for move %d\n", m);
      return 1;
    }
  std::vector<Out> out(games);
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < games; ++i) out[i] = play(d1, d2, 0x5eed + 7919ULL * i);
  int w[3] = {0, 0, 0}, dev_games = 0;
  double plies = 0, mirror_rate = 0, dev_ply = 0;
  for (auto &o : out) {
    w[o.winner + 1]++;
    plies += o.plies;
    mirror_rate += double(o.p2_mirrors) / std::max(1, o.p2_moves);
    if (o.first_deviation >= 0) {
      ++dev_games;
      dev_ply += o.first_deviation;
    }
  }
  std::printf("P1 depth %d vs mirror-P2 (fallback depth %d), %d games\n", d1, d2,
              games);
  std::printf("  P1 %.3f  draw %.3f  P2 %.3f   mean plies %.1f\n",
              double(w[1]) / games, double(w[0]) / games, double(w[2]) / games,
              plies / games);
  std::printf("  P2 mirrored %.1f%% of its moves; %d games needed a deviation, "
              "first at ply %.1f on average\n",
              100 * mirror_rate / games, dev_games,
              dev_games ? dev_ply / dev_games : 0);
  return 0;
}
