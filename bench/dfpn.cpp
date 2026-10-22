// df-pn prototype for Corintho's exact result, against the alpha-beta solver
//
// Worklog 2026-09-25-nn-architectures, entry 26: not faster than the
// alpha-beta solver; kept as the record of that experiment.
//
// Question 1: does the side to move at the root win? Negamax df-pn (Nagai
// 2002): each node has (phi, delta), the proof and disproof numbers of
// "the side to move here achieves its goal". The root side's goal is a
// win; the other side's goal is to stop that (a win or a draw for it).
// Draws are rare, so every result records whether its proof leaned on a
// draw: a "no" whose proof never did is already a proof of a loss, and the
// second question (at least a draw?) runs only when one did.
//
//   dfpn POSITIONS.bin LIST [eps] [init_moves]  (DFPN_VERBOSE=1: per position)
//   LIST: lines "index result ab_nodes" (the alpha-beta solver's answers)

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>

#include "game.h"
#include "solver.h"
#include "util.h"

namespace {

constexpr uint32_t kInf = 1U << 30;

uint32_t addCap(uint32_t a, uint32_t b) {
  const uint64_t s = uint64_t{a} + b;
  return s >= kInf ? kInf : static_cast<uint32_t>(s);
}

struct Entry {
  uint32_t phi = 1, delta = 1;
  bool drawn = false;  // the proof or disproof leans on a draw
};

struct Key {
  uint64_t board, rest;
  bool operator==(const Key &o) const {
    return board == o.board && rest == o.rest;
  }
};
struct KeyHash {
  size_t operator()(const Key &k) const {
    uint64_t x = k.board * 0x9E3779B97F4A7C15ULL ^ (k.rest + 0x632BE59BD9B4E019ULL);
    x ^= x >> 29;
    x *= 0xBF58476D1CE4E5B9ULL;
    return x ^ (x >> 32);
  }
};

class Dfpn {
 public:
  // root_wins_draws: false asks "does the root side win?", true asks "does
  // the root side at least draw?" (a draw then counts for the root side)
  Dfpn(int root_side, bool draw_for_root, double eps, bool init_moves)
      : root_side_(root_side), draw_for_root_(draw_for_root), eps_(eps),
        init_moves_(init_moves) {}

  uint64_t expansions = 0;
  // Keep every solved position whose proof or disproof never touched a
  // draw: it holds whichever side a draw counts for
  void inheritDrawFree(const Dfpn &other) {
    for (const auto &[k, e] : other.table_)
      if ((e.phi == 0 || e.delta == 0) && !e.drawn)
        table_.emplace(k, e);
  }

  Entry solve(const Game &root) {
    uint32_t th_phi = kInf - 1, th_delta = kInf - 1;
    mid(root, th_phi, th_delta);
    return lookup(root);
  }

 private:
  Key keyOf(const Game &g) const {
    Key k;
    g.key(k.board, k.rest);
    return k;
  }
  int sideOf(const Key &k) const { return static_cast<int>(k.rest & 1); }

  // A position's entry, initialised on first sight (terminal positions get
  // their final values)
  Entry lookup(const Game &g) {
    const Key k = keyOf(g);
    auto it = table_.find(k);
    if (it != table_.end())
      return it->second;
    Entry e;
    MoveMask legal;
    const bool lines = g.getLegalMoves(legal);
    if (!legal.any()) {
      // The side to move here has no move: lost if a line stands, else drawn
      const bool mover_is_root = sideOf(k) == root_side_;
      bool goal_met;
      if (lines)
        goal_met = false;  // the mover lost
      else
        goal_met = mover_is_root ? draw_for_root_ : !draw_for_root_;
      e.phi = goal_met ? 0 : kInf;
      e.delta = goal_met ? kInf : 0;
      e.drawn = !lines;
    } else if (init_moves_) {
      // df-pn+: refuting a position means refuting every move in it
      e.delta = static_cast<uint32_t>(legal.count());
    }
    table_.emplace(k, e);
    return e;
  }

  void mid(const Game &g, uint32_t th_phi, uint32_t th_delta) {
    Entry e = lookup(g);
    if (e.phi == 0 || e.delta == 0)
      return;
    ++expansions;
    MoveMask legal;
    g.getLegalMoves(legal);
    std::vector<Game> children;
    children.reserve(legal.count());
    forEachMove(legal, [&](int32_t m) {
      children.push_back(g);
      children.back().doMove(m);
    });
    const Key k = keyOf(g);
    for (;;) {
      // phi = min child delta, delta = sum child phi
      uint32_t phi = kInf, delta = 0, d2 = kInf;
      int best = -1;
      bool drawn_proof = false, drawn_disproof = false;
      for (size_t i = 0; i < children.size(); ++i) {
        const Entry c = lookup(children[i]);
        delta = addCap(delta, c.phi);
        if (c.phi == 0)
          drawn_disproof |= c.drawn;  // every child must be proven
        if (c.delta < phi) {
          d2 = phi;
          phi = c.delta;
          best = static_cast<int>(i);
        } else if (c.delta < d2) {
          d2 = c.delta;
        }
      }
      // A proof here: a child disproven (pick a draw-free one if any)
      if (phi == 0) {
        bool any_clean = false;
        for (const Game &c : children) {
          const Entry ce = lookup(c);
          if (ce.delta == 0 && !ce.drawn)
            any_clean = true;
        }
        drawn_proof = !any_clean;
      }
      Entry &slot = table_[k];
      slot.phi = phi;
      slot.delta = delta;
      slot.drawn = phi == 0 ? drawn_proof : delta == 0 ? drawn_disproof : false;
      if (phi >= th_phi || delta >= th_delta)
        return;
      const Entry c = lookup(children[best]);
      // Child thresholds (negamax df-pn), with the 1 + eps trick against
      // thrashing between siblings (Pawlewicz and Lew 2007)
      const uint32_t c_th_phi =
          static_cast<uint32_t>(std::min<uint64_t>(
              kInf - 1, uint64_t{th_delta} + c.phi - delta));
      const uint64_t d2e =
          d2 >= kInf ? kInf : static_cast<uint64_t>(d2 * (1.0 + eps_)) + 1;
      const uint32_t c_th_delta =
          static_cast<uint32_t>(std::min<uint64_t>(th_phi, d2e));
      mid(children[best], c_th_phi, c_th_delta);
    }
  }

  int root_side_;
  bool draw_for_root_;
  double eps_;
  bool init_moves_;
  std::unordered_map<Key, Entry, KeyHash> table_;
};

}  // namespace

int main(int argc, char **argv) {
  FILE *f = std::fopen(argv[1], "rb");
  std::vector<uint8_t> raw;
  uint8_t buf[71];
  while (std::fread(buf, 1, 71, f) == 71)
    raw.insert(raw.end(), buf, buf + 71);
  std::fclose(f);
  const double eps = argc > 3 ? std::atof(argv[3]) : 0.25;
  const bool init_moves = argc > 4 && std::atoi(argv[4]) != 0;
  const bool verbose = std::getenv("DFPN_VERBOSE") != nullptr;
  FILE *list = std::fopen(argv[2], "r");
  long idx, ab_result, ab_nodes;
  long n = 0, wrong = 0, second = 0, df_better = 0;
  double ab_s = 0, df_s = 0;
  uint64_t ab_total = 0, df_total = 0;
  Solver solver{20};
  while (std::fscanf(list, "%ld %ld %ld", &idx, &ab_result, &ab_nodes) == 3) {
    const uint8_t *s = raw.data() + idx * 71;
    int32_t board[64], pieces[6];
    for (int i = 0; i < 64; ++i)
      board[i] = s[i] != 0;
    for (int i = 0; i < 6; ++i)
      pieces[i] = s[64 + i];
    const Game g(board, 0, pieces);

    auto t0 = std::chrono::steady_clock::now();
    solver.clear();
    const int ab = solver.solve(g, 1ULL << 40);
    auto t1 = std::chrono::steady_clock::now();
    int result;
    uint64_t exp = 0;
    bool ran_second = false;
    {
      Dfpn win(0, false, eps, init_moves);
      const Entry e = win.solve(g);
      exp += win.expansions;
      if (e.phi == 0) {
        result = 1;
      } else if (!e.drawn) {
        result = -1;  // a draw-free "no win": the opponent wins
      } else {
        ++second;
        ran_second = true;
        Dfpn draw(0, true, eps, init_moves);
        draw.inheritDrawFree(win);
        const Entry e2 = draw.solve(g);
        exp += draw.expansions;
        result = e2.phi == 0 ? 0 : -1;
      }
    }
    auto t2 = std::chrono::steady_clock::now();
    const double a = std::chrono::duration<double>(t1 - t0).count();
    const double d = std::chrono::duration<double>(t2 - t1).count();
    ab_s += a;
    df_s += d;
    ab_total += solver.nodes();
    df_total += exp;
    df_better += d < a;
    if (verbose)
      std::printf("%6ld result %+d  alpha-beta %9llu nodes %8.3f s  df-pn %9llu exp %8.3f s%s\n",
                  idx, ab, (unsigned long long)solver.nodes(), a,
                  (unsigned long long)exp, d, ran_second ? "  (2nd question)" : "");
    if (result != ab) {
      ++wrong;
      std::printf("MISMATCH index %ld: alpha-beta %d, df-pn %d\n", idx, ab, result);
    }
    ++n;
  }
  std::printf("%ld positions, %ld mismatches, second question on %ld\n", n,
              wrong, second);
  std::printf("alpha-beta %.2f s, %llu nodes; df-pn %.2f s, %llu expansions; "
              "df-pn faster on %ld\n",
              ab_s, (unsigned long long)ab_total, df_s,
              (unsigned long long)df_total, df_better);
}
