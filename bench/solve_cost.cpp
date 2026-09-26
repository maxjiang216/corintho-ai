// Endgame solver cost on real self-play positions, by the horizon
// P = 2 * reserves + occupied squares, which bounds the plies left (every move
// lowers it by at least one). Worklog 2026-09-25-nn-architectures, entry 07.
//
//   solve_cost POSITIONS.bin CAP THREADS > out.tsv
//
// POSITIONS.bin: rows of 71 bytes, the 70 network inputs x 4 (uint8, as in
// arch/dataset.py) then the game-result value target (int8, side to move).
// Each position is solved cold (a fresh transposition table), with
// ab_probe.cpp's solver: exact win/draw/loss alpha-beta, TT move first,
// move-ID order otherwise. Output TSV: P, reserves, occupied, solved (0/1),
// result (-1/0/1, side to move), value target, nodes, microseconds.

#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "game.h"
#include "move.h"
#include "util.h"

namespace {

static_assert(sizeof(Game) == 16, "key layout assumes a 16-byte Game");

struct Key {
  uint64_t a{0}, b{0};
  bool operator==(const Key &o) const { return a == o.a && b == o.b; }
};

Key rawKey(const Game &g) {
  uint64_t w[2];
  std::memcpy(w, &g, 16);
  w[1] &= 0x00FFFFFFFFFFFFFFULL;  // byte 15 is padding
  return {w[0], w[1]};
}

enum Bound : int8_t { kExact = 0, kLower = 1, kUpper = 2 };

struct Entry {
  Key key;
  int8_t score, bound, move, used;
};

struct Table {
  std::vector<Entry> e;
  uint64_t mask;
  explicit Table(int log2)
      : e(size_t{1} << log2), mask((uint64_t{1} << log2) - 1) {}
  void clear() { std::fill(e.begin(), e.end(), Entry{}); }
  static uint64_t h(const Key &k) {
    uint64_t x = k.a * 0x9E3779B97F4A7C15ULL ^ (k.b + 0x632BE59BD9B4E019ULL);
    x ^= x >> 29;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 32;
    return x;
  }
  Entry *probe(const Key &k) {
    Entry &x = e[h(k) & mask];
    return (x.used && x.key == k) ? &x : nullptr;
  }
  void store(const Key &k, int score, int bound, int move) {
    e[h(k) & mask] = Entry{k, static_cast<int8_t>(score),
                           static_cast<int8_t>(bound),
                           static_cast<int8_t>(move), 1};
  }
};

struct Solver {
  Table tt{18};
  uint64_t nodes = 0, cap = 0;
  bool aborted = false;

  // -1/0/+1 for the side to move
  int solve(const Game &g, int alpha, int beta) {
    if (++nodes > cap) {
      aborted = true;
      return 0;
    }
    MoveMask m;
    const bool lines = g.getLegalMoves(m);
    if (!m.any())
      return lines ? -1 : 0;
    const Key k = rawKey(g);
    const int a0 = alpha;
    int tt_move = -1;
    if (Entry *e = tt.probe(k)) {
      const int s = e->score;
      if (e->bound == kExact || (e->bound == kLower && s >= beta) ||
          (e->bound == kUpper && s <= alpha))
        return s;
      if (e->bound == kLower)
        alpha = std::max(alpha, s);
      if (e->bound == kUpper)
        beta = std::min(beta, s);
      tt_move = e->move;
    }
    int moves[kNumMoves];
    int n = 0;
    forEachMove(m, [&](int mv) { moves[n++] = mv; });
    if (tt_move >= 0)
      for (int i = 0; i < n; ++i)
        if (moves[i] == tt_move) {
          std::rotate(moves, moves + i, moves + i + 1);
          break;
        }
    int best = -2, best_mv = moves[0];
    for (int i = 0; i < n; ++i) {
      Game c = g;
      c.doMove(moves[i]);
      const int s = -solve(c, -beta, -alpha);
      if (aborted)
        return 0;
      if (s > best) {
        best = s;
        best_mv = moves[i];
      }
      alpha = std::max(alpha, s);
      if (alpha >= beta)
        break;
    }
    tt.store(k, best, best <= a0 ? kUpper : best >= beta ? kLower : kExact,
             best_mv);
    return best;
  }
};

}  // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: solve_cost POSITIONS.bin CAP THREADS\n");
    return 1;
  }
  FILE *f = std::fopen(argv[1], "rb");
  if (!f)
    return 1;
  std::vector<uint8_t> raw;
  uint8_t buf[71];
  while (std::fread(buf, 1, 71, f) == 71)
    raw.insert(raw.end(), buf, buf + 71);
  std::fclose(f);
  const int64_t n = static_cast<int64_t>(raw.size() / 71);
  const uint64_t cap = std::strtoull(argv[2], nullptr, 10);
  omp_set_num_threads(std::atoi(argv[3]));
  std::vector<std::array<int64_t, 8>> out(static_cast<size_t>(n));
#pragma omp parallel
  {
    Solver solver;
#pragma omp for schedule(dynamic, 1)
    for (int64_t r = 0; r < n; ++r) {
      const uint8_t *s = raw.data() + r * 71;
      int32_t board[4 * kBoardSize];
      int occupied = 0;
      for (int32_t i = 0; i < 4 * kBoardSize; ++i)
        board[i] = s[i] != 0;
      for (int32_t sp = 0; sp < kBoardSize; ++sp)
        occupied += (s[4 * sp] | s[4 * sp + 1] | s[4 * sp + 2]) != 0;
      int32_t pieces[6];
      int reserves = 0;
      for (int32_t i = 0; i < 6; ++i)
        reserves += pieces[i] = s[4 * kBoardSize + i];
      const Game game{board, 0, pieces};
      solver.tt.clear();
      solver.nodes = 0;
      solver.cap = cap;
      solver.aborted = false;
      const auto t0 = std::chrono::steady_clock::now();
      const int result = solver.solve(game, -1, 1);
      const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
      out[static_cast<size_t>(r)] = {2 * reserves + occupied,
                                     reserves,
                                     occupied,
                                     solver.aborted ? 0 : 1,
                                     solver.aborted ? 0 : result,
                                     static_cast<int8_t>(s[70]),
                                     static_cast<int64_t>(solver.nodes),
                                     us};
    }
  }
  std::printf("P\treserves\toccupied\tsolved\tresult\tvalue\tnodes\tus\n");
  for (const auto &o : out)
    std::printf("%ld\t%ld\t%ld\t%ld\t%ld\t%ld\t%ld\t%ld\n", o[0], o[1], o[2],
                o[3], o[4], o[5], o[6], o[7]);
  return 0;
}
