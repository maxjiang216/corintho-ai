// Monte Carlo estimate of how many Corintho positions are real.
//
// The exact upper bound (see bound.py in the ab-probe results) counts every
// board whose stacks are legal (7 states per square) and whose piece totals fit
// the 8-per-type supply, times the reserve splits (0-4 per player per type),
// times the frozen square (exactly one, always occupied), colour-folded: the
// board has no colours, so a position is (board, frozen, mover's reserves,
// opponent's reserves) with the side to move implicit. That set is ~1.6e15;
// the 8 board symmetries divide it by ~8 more.
//
// This tool draws positions uniformly from that set and applies filters in
// order, so pass rate x bound estimates each reduced count:
//
//   lines     every line on the board passes through the frozen square. The
//             last move changed only its destination (the source of a stack
//             move becomes empty), so a line elsewhere would have had to be
//             broken by that move.
//   live      the side to move has a legal move (not terminal).
//   pred      some legal position leads here in one move. Checked with the
//             engine itself: candidate predecessor, getLegalMoves, doMove,
//             compare bytes.
//   reach     a backward search finds a path to the empty starting board
//             within a node budget. "unknown" when the budget runs out.
//
// Usage: state_sample [samples] [reach_samples] [threads]
// Environment: REACH_BUDGET (default 200000 backward nodes per sample).

#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <unordered_set>
#include <vector>

#include "game.h"
#include "move.h"
#include "util.h"

namespace {

static_assert(sizeof(Game) == 16, "byte layout assumed below");

// Stack states as piece bits (bit 0 base, 1 column, 2 capital).
constexpr int kStates[7] = {0, 1, 2, 4, 3, 6, 7};  // -, B, C, K, BC, CK, BCK

// W[i][m][b][c][k]: ways to fill squares i..15 with exactly m occupied squares
// and b/c/k pieces of each type.
double W[17][17][9][9][9];

void buildCounts() {
  std::memset(W, 0, sizeof(W));
  W[16][0][0][0][0] = 1;
  for (int i = 15; i >= 0; --i)
    for (int m = 0; m <= 16; ++m)
      for (int b = 0; b <= 8; ++b)
        for (int c = 0; c <= 8; ++c)
          for (int k = 0; k <= 8; ++k) {
            double s = 0;
            for (int st : kStates) {
              const int sb = st & 1, sc = (st >> 1) & 1, sk = (st >> 2) & 1;
              const int occ = st != 0;
              if (m < occ || b < sb || c < sc || k < sk) continue;
              s += W[i + 1][m - occ][b - sb][c - sc][k - sk];
            }
            W[i][m][b][c][k] = s;
          }
}

int splits(int n) { return std::min(n, 8 - n) + 1; }

struct Class {
  int m, b, c, k;
};
std::vector<Class> g_classes;
std::vector<double> g_cum;
double g_total = 0;

void buildClasses() {
  for (int m = 1; m <= 16; ++m)
    for (int b = 0; b <= 8; ++b)
      for (int c = 0; c <= 8; ++c)
        for (int k = 0; k <= 8; ++k) {
          const double w = W[0][m][b][c][k];
          if (w == 0) continue;
          const double weight = w * splits(b) * splits(c) * splits(k) * m;
          g_total += weight;
          g_classes.push_back({m, b, c, k});
          g_cum.push_back(g_total);
        }
}

// Raw fields of a position, independent of Game's private layout.
struct Pos {
  uint64_t board;   // 4 bits per square: piece bits 0-2, frozen bit 3
  int8_t mine[3];   // side to move's reserves
  int8_t theirs[3];
};

Game toGame(const Pos &p, int to_play) {
  int32_t bits[64];
  for (int i = 0; i < 64; ++i) bits[i] = (p.board >> i) & 1;
  int32_t pieces[6];
  for (int t = 0; t < 3; ++t) {
    pieces[to_play * 3 + t] = p.mine[t];
    pieces[(1 - to_play) * 3 + t] = p.theirs[t];
  }
  return Game(bits, to_play, pieces);
}

void gameBytes(const Game &g, uint64_t w[2]) {
  std::memcpy(w, &g, 16);
  w[1] &= 0x00FFFFFFFFFFFFFFULL;
}

Pos samplePos(std::mt19937_64 &rng) {
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const double x = u(rng) * g_total;
  const size_t ci =
      std::lower_bound(g_cum.begin(), g_cum.end(), x) - g_cum.begin();
  Class cl = g_classes[std::min(ci, g_classes.size() - 1)];
  Pos p{};
  int m = cl.m, b = cl.b, c = cl.c, k = cl.k;
  int occupied[16];
  int no = 0;
  for (int i = 0; i < 16; ++i) {
    double tot = W[i][m][b][c][k];
    double y = u(rng) * tot;
    int chosen = 0;
    for (int st : kStates) {
      const int sb = st & 1, sc = (st >> 1) & 1, sk = (st >> 2) & 1;
      const int occ = st != 0;
      if (m < occ || b < sb || c < sc || k < sk) continue;
      const double w = W[i + 1][m - occ][b - sb][c - sc][k - sk];
      chosen = st;
      if (y < w) break;
      y -= w;
    }
    const int sb = chosen & 1, sc = (chosen >> 1) & 1, sk = (chosen >> 2) & 1;
    m -= chosen != 0;
    b -= sb;
    c -= sc;
    k -= sk;
    p.board |= static_cast<uint64_t>(chosen) << (4 * i);
    if (chosen) occupied[no++] = i;
  }
  const int f = occupied[std::uniform_int_distribution<int>(0, no - 1)(rng)];
  p.board |= uint64_t{1} << (4 * f + 3);
  const int on[3] = {cl.b, cl.c, cl.k};
  for (int t = 0; t < 3; ++t) {
    const int rem = 8 - on[t];
    const int lo = std::max(0, rem - 4), hi = std::min(4, rem);
    const int r = std::uniform_int_distribution<int>(lo, hi)(rng);
    p.mine[t] = static_cast<int8_t>(r);
    p.theirs[t] = static_cast<int8_t>(rem - r);
  }
  return p;
}

int topType(uint64_t board, int s) {
  const int st = (board >> (4 * s)) & 7;
  return st & 4 ? 2 : st & 2 ? 1 : st & 1 ? 0 : -1;
}

int frozenSquare(uint64_t board) {
  for (int s = 0; s < 16; ++s)
    if ((board >> (4 * s + 3)) & 1) return s;
  return -1;
}

// Every line shape currently topped by one type passes through the frozen
// square. With no frozen square (start position) there must be no line.
bool linesThroughFrozen(uint64_t board) {
  const int f = frozenSquare(board);
  for (const LineShape &ls : kLineShapes) {
    const int t = topType(board, ls.cells[0]);
    if (t < 0) continue;
    bool line = true, has_f = false;
    for (int j = 0; j < ls.count; ++j) {
      line &= topType(board, ls.cells[j]) == t;
      has_f |= ls.cells[j] == f;
    }
    if (line && !has_f) return false;
  }
  return true;
}

int reservesLeft(const Pos &p) {
  int r = 0;
  for (int t = 0; t < 3; ++t) r += p.mine[t] + p.theirs[t];
  return r;
}

// All legal one-move predecessors of p, colour-folded (the predecessor's
// mover is p's "theirs"). Verified with the engine.
template <typename F>
void forEachPredecessor(const Pos &p, F fn) {
  const int f = frozenSquare(p.board);
  if (f < 0) return;  // start position has no predecessor
  const uint64_t unfrozen = p.board & kUnfrozenMask;
  const int fst = (unfrozen >> (4 * f)) & 7;
  uint64_t target[2];
  gameBytes(toGame(p, 0), target);

  auto tryPred = [&](uint64_t pred_board, const int8_t pred_mover[3],
                     const int8_t pred_other[3], int move_id) {
    // Enumerate the predecessor's frozen square: any occupied square, or none
    // when the predecessor board is empty (the start).
    const uint64_t occ_mask = pred_board;
    bool any = false;
    for (int q = -1; q < 16; ++q) {
      if (q >= 0 && ((occ_mask >> (4 * q)) & 7) == 0) continue;
      if (q < 0 && pred_board != 0) continue;
      any = true;
      Pos pp{};
      pp.board = pred_board | (q >= 0 ? uint64_t{1} << (4 * q + 3) : 0);
      std::memcpy(pp.mine, pred_mover, 3);
      std::memcpy(pp.theirs, pred_other, 3);
      Game g = toGame(pp, 1);
      MoveMask legal;
      g.getLegalMoves(legal);
      if (!legal.test(move_id)) continue;
      g.doMove(move_id);
      uint64_t got[2];
      gameBytes(g, got);
      if (got[0] == target[0] && got[1] == target[1]) fn(pp);
    }
    (void)any;
  };

  // (a) Unplace the top piece of the frozen stack; the previous mover gets it
  // back.
  {
    const int t = fst & 4 ? 2 : fst & 2 ? 1 : 0;
    int8_t mover[3], other[3];
    std::memcpy(mover, p.theirs, 3);
    std::memcpy(other, p.mine, 3);
    if (mover[t] < 4) {
      ++mover[t];
      const uint64_t pred = unfrozen & ~(uint64_t{1} << (4 * f + t));
      const int move_id = 48 + t * 16 + f;
      tryPred(pred, mover, other, move_id);
    }
  }
  // (b) Un-merge: the frozen stack split into a lower part that stays and an
  // upper part that came from an adjacent square which is now empty.
  {
    int8_t mover[3], other[3];
    std::memcpy(mover, p.theirs, 3);
    std::memcpy(other, p.mine, 3);
    // piece bits present, in order base < column < capital
    int pieces[3], np = 0;
    for (int t = 0; t < 3; ++t)
      if (fst & (1 << t)) pieces[np++] = t;
    for (int split = 1; split < np; ++split) {
      int lower = 0, upper = 0;
      for (int j = 0; j < split; ++j) lower |= 1 << pieces[j];
      for (int j = split; j < np; ++j) upper |= 1 << pieces[j];
      // merged stacks are contiguous: upper's bottom is lower's top + 1
      if (pieces[split] != pieces[split - 1] + 1) continue;
      for (int m = 0; m < 48; ++m) {
        const MoveInfo &mi = kMoveTable[m];
        if (mi.to != f) continue;
        const int a = mi.from;
        if (((unfrozen >> (4 * a)) & 7) != 0) continue;
        uint64_t pred = unfrozen & ~(uint64_t{7} << (4 * f));
        pred |= static_cast<uint64_t>(lower) << (4 * f);
        pred |= static_cast<uint64_t>(upper) << (4 * a);
        tryPred(pred, mover, other, m);
      }
    }
  }
}

struct KeyHash {
  size_t operator()(const std::pair<uint64_t, uint64_t> &k) const {
    uint64_t x = k.first * 0x9E3779B97F4A7C15ULL ^ k.second;
    x ^= x >> 31;
    return x * 0xBF58476D1CE4E5B9ULL;
  }
};

// 1 reachable, 0 proven unreachable, -1 budget exhausted
int reachable(const Pos &start, uint64_t budget) {
  std::unordered_set<std::pair<uint64_t, uint64_t>, KeyHash> seen;
  std::vector<Pos> stack{start};
  uint64_t nodes = 0;
  auto key = [](const Pos &p) {
    uint64_t hi = 0;
    for (int t = 0; t < 3; ++t)
      hi = hi << 8 | static_cast<uint8_t>(p.mine[t]);
    for (int t = 0; t < 3; ++t)
      hi = hi << 8 | static_cast<uint8_t>(p.theirs[t]);
    return std::make_pair(p.board, hi);
  };
  seen.insert(key(start));
  while (!stack.empty()) {
    if (++nodes > budget) return -1;
    Pos cur = stack.back();
    stack.pop_back();
    if (cur.board == 0) return 1;  // empty board, full reserves by counting
    // Push un-merges first and unplaces last, so unplaces pop first: they
    // head straight for the empty board.
    std::vector<Pos> preds;
    forEachPredecessor(cur, [&](const Pos &pp) { preds.push_back(pp); });
    std::sort(preds.begin(), preds.end(), [](const Pos &x, const Pos &y) {
      return reservesLeft(x) < reservesLeft(y);
    });
    for (const Pos &pp : preds)
      if (seen.insert(key(pp)).second) stack.push_back(pp);
  }
  return 0;
}

struct Tally {
  long n = 0, lines = 0, live = 0, pred = 0;
  long reach_n = 0, reach_yes = 0, reach_no = 0, reach_unknown = 0;
  void add(const Tally &o) {
    n += o.n; lines += o.lines; live += o.live; pred += o.pred;
    reach_n += o.reach_n; reach_yes += o.reach_yes; reach_no += o.reach_no;
    reach_unknown += o.reach_unknown;
  }
};

void report(const char *label, double frac, long hits, long n, double base) {
  const double p = n ? double(hits) / n : 0;
  const double se = n ? std::sqrt(p * (1 - p) / n) : 0;
  std::printf("  %-34s %8ld / %-8ld  %.5f +- %.5f   => %.3e positions (/8 sym %.3e)\n",
              label, hits, n, frac, 1.96 * se, base * frac, base * frac / 8);
}

}  // namespace

int main(int argc, char **argv) {
  const long samples = argc > 1 ? std::atol(argv[1]) : 1000000;
  const long reach_samples = argc > 2 ? std::atol(argv[2]) : 20000;
  const int threads = argc > 3 ? std::atoi(argv[3]) : 20;
  const char *rb = std::getenv("REACH_BUDGET");
  const uint64_t budget = rb ? std::strtoull(rb, nullptr, 10) : 200000;
  omp_set_num_threads(threads);

  buildCounts();
  buildClasses();
  std::printf("colour-folded bound: %.4e positions (/8 symmetries: %.4e)\n",
              g_total, g_total / 8);

  constexpr int kRBuckets = 25;
  std::vector<std::vector<Tally>> per(threads, std::vector<Tally>(kRBuckets));
#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    std::mt19937_64 rng(12345 + 7919 * tid);
#pragma omp for schedule(dynamic, 256)
    for (long i = 0; i < samples; ++i) {
      const Pos p = samplePos(rng);
      Tally &t = per[tid][reservesLeft(p)];
      ++t.n;
      if (!linesThroughFrozen(p.board)) continue;
      ++t.lines;
      Game g = toGame(p, 0);
      MoveMask legal;
      g.getLegalMoves(legal);
      if (!legal.any()) continue;
      ++t.live;
      bool has_pred = false;
      forEachPredecessor(p, [&](const Pos &) { has_pred = true; });
      if (!has_pred) continue;
      ++t.pred;
    }
  }
  // Second pass: keep drawing until reach_samples positions that pass every
  // cheap filter have had the backward search. Tallied by R separately.
#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    std::mt19937_64 rng(999331 + 104729 * tid);
#pragma omp for schedule(dynamic, 1)
    for (long i = 0; i < reach_samples; ++i) {
      for (;;) {
        const Pos p = samplePos(rng);
        if (!linesThroughFrozen(p.board)) continue;
        Game g = toGame(p, 0);
        MoveMask legal;
        g.getLegalMoves(legal);
        if (!legal.any()) continue;
        bool has_pred = false;
        forEachPredecessor(p, [&](const Pos &) { has_pred = true; });
        if (!has_pred) continue;
        Tally &t = per[tid][reservesLeft(p)];
        ++t.reach_n;
        const int r = reachable(p, budget);
        (r == 1 ? t.reach_yes : r == 0 ? t.reach_no : t.reach_unknown)++;
        break;
      }
    }
  }
  std::vector<Tally> byR(kRBuckets);
  Tally all;
  for (auto &th : per)
    for (int r = 0; r < kRBuckets; ++r) byR[r].add(th[r]);
  for (auto &t : byR) all.add(t);

  std::printf("\nsamples %ld\n", all.n);
  report("lines through frozen square", double(all.lines) / all.n, all.lines,
         all.n, g_total);
  report("... and not terminal", double(all.live) / all.n, all.live, all.n,
         g_total);
  report("... and has a legal predecessor", double(all.pred) / all.n, all.pred,
         all.n, g_total);
  if (all.reach_n) {
    const double pp = double(all.pred) / all.n;
    std::printf("\nreachability, on %ld samples that passed every filter above "
                "(budget %llu nodes):\n",
                all.reach_n, static_cast<unsigned long long>(budget));
    std::printf("  reachable %ld, proven unreachable %ld, unknown %ld\n",
                all.reach_yes, all.reach_no, all.reach_unknown);
    const double lo = double(all.reach_yes) / all.reach_n;
    const double hi = double(all.reach_yes + all.reach_unknown) / all.reach_n;
    std::printf("  reachable fraction in [%.4f, %.4f]  =>  %.3e .. %.3e positions "
                "(/8 sym %.3e .. %.3e)\n",
                lo, hi, g_total * pp * lo, g_total * pp * hi,
                g_total * pp * lo / 8, g_total * pp * hi / 8);
  }

  std::printf("\nby reserve pieces left (both players):\n");
  std::printf("  R  | samples  | lines   | live    | pred    | reach y/n/? \n");
  for (int r = 0; r < kRBuckets; ++r) {
    const Tally &t = byR[r];
    if (!t.n) continue;
    std::printf("  %2d | %8ld | %.5f | %.5f | %.5f | %ld/%ld/%ld\n", r, t.n,
                double(t.lines) / t.n, double(t.live) / t.n,
                double(t.pred) / t.n, t.reach_yes, t.reach_no, t.reach_unknown);
  }
  return 0;
}
