// Probe: what would alpha-beta search look like for Corintho, using the
// AlphaZero network (model_93) as the evaluation?
//
// Three phases, all single-binary, no Python:
//
//   A. Distributions. Play games with the network's policy (temperature 1 for
//      the first OPEN_PLIES plies, then greedy) and with uniform random moves.
//      Per ply bucket: legal moves, how often lines are on the board, policy
//      sharpness (top-1 mass, entropy, moves needed for 90% mass), the value
//      head by side to move, and whether the policy's top move agrees with the
//      best child by one-ply value lookahead.
//
//   B. Alpha-beta trees. Negamax alpha-beta with a transposition table,
//      iterative deepening, network value at the leaves, from positions sampled
//      out of the policy games. Two orderings: move-ID order and policy order
//      (the interior node's network call gives both value and policy). Reports
//      nodes, network calls and effective branching factor per depth.
//
//   C. Solver reach. Pure win/draw/loss alpha-beta to the end of the game, no
//      network, with and without symmetry-folded transposition keys. From
//      positions at several plies. Also scores the value head against every
//      proven result.
//
// Usage: ab_probe <model.mlp> [games] [threads]
// Environment: OPEN_PLIES (default 6), AB_MAXDEPTH (default 6),
//   AB_POS (positions per ply bucket in B, default 12),
//   SOLVE_POS (positions per ply bucket in C, default 24),
//   SOLVE_CAP (node cap per solve, default 20000000).
//
// The transposition key reads Game's bytes directly (static_assert on its
// size): 8 bytes of board, 6 reserve counts, side to move, one padding byte
// which is masked off. It is exact -- no hashing collisions in the key itself.

#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "game.h"
#include "mlp.h"
#include "move.h"
#include "util.h"

namespace {

static_assert(sizeof(Game) == 16, "key layout assumes a 16-byte Game");

int envInt(const char *name, int def) {
  const char *v = std::getenv(name);
  return v ? std::atoi(v) : def;
}

double now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

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

// ---- Symmetry ------------------------------------------------------------

// kPerm[t][s]: where space s goes under transform t (D4 on the 4x4 board).
std::array<std::array<int, 16>, 8> kPerm;
// kMovePerm[t][m]: the move ID that move m becomes under transform t.
std::array<std::array<int, kNumMoves>, 8> kMovePerm;
// kInvT[t]: inverse transform index
std::array<int, 8> kInvT;

void initSymmetry() {
  for (int t = 0; t < 8; ++t) {
    for (int s = 0; s < 16; ++s) {
      int r = s / 4, c = s % 4;
      int rr = r, cc = c;
      for (int k = 0; k < (t & 3); ++k) {  // rotate 90 degrees k times
        int nr = cc, nc = 3 - rr;
        rr = nr;
        cc = nc;
      }
      if (t & 4) cc = 3 - cc;  // then reflect
      kPerm[t][s] = rr * 4 + cc;
    }
  }
  for (int t = 0; t < 8; ++t) {
    for (int m = 0; m < kNumMoves; ++m) {
      const MoveInfo &mi = kMoveTable[m];
      int from = mi.is_place ? -1 : kPerm[t][mi.from];
      int to = kPerm[t][mi.to];
      kMovePerm[t][m] = -1;
      for (int n = 0; n < kNumMoves; ++n) {
        const MoveInfo &ni = kMoveTable[n];
        if (ni.is_place == mi.is_place && ni.to == to &&
            ni.piece == mi.piece && (mi.is_place || ni.from == from)) {
          kMovePerm[t][m] = n;
        }
      }
      if (kMovePerm[t][m] < 0) {
        std::fprintf(stderr, "symmetry: no image for move %d under %d\n", m, t);
        std::exit(1);
      }
    }
    for (int u = 0; u < 8; ++u) {
      bool inv = true;
      for (int s = 0; s < 16; ++s) inv &= kPerm[u][kPerm[t][s]] == s;
      if (inv) kInvT[t] = u;
    }
  }
}

uint64_t permuteBoard(uint64_t b, int t) {
  uint64_t out = 0;
  for (int s = 0; s < 16; ++s)
    out |= ((b >> (4 * s)) & 0xFULL) << (4 * kPerm[t][s]);
  return out;
}

// Canonical key: smallest board word over the 8 transforms. Returns the
// transform used, so a stored move can be mapped in and out.
Key canonKey(const Game &g, int &t_out) {
  Key k = rawKey(g);
  uint64_t best = k.a;
  t_out = 0;
  for (int t = 1; t < 8; ++t) {
    uint64_t p = permuteBoard(k.a, t);
    if (p < best) {
      best = p;
      t_out = t;
    }
  }
  return {best, k.b};
}

// ---- Transposition table ---------------------------------------------------

enum Bound : int8_t { kExact = 0, kLower = 1, kUpper = 2 };

struct TTEntry {
  Key key;
  float score;
  int8_t depth;
  int8_t bound;
  int8_t move;
  int8_t used;
};

struct TT {
  std::vector<TTEntry> e;
  uint64_t mask;
  explicit TT(int log2) : e(size_t{1} << log2), mask((uint64_t{1} << log2) - 1) {}
  void clear() { std::fill(e.begin(), e.end(), TTEntry{}); }
  static uint64_t h(const Key &k) {
    uint64_t x = k.a * 0x9E3779B97F4A7C15ULL ^ (k.b + 0x632BE59BD9B4E019ULL);
    x ^= x >> 29;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 32;
    return x;
  }
  TTEntry *probe(const Key &k) {
    TTEntry &x = e[h(k) & mask];
    return (x.used && x.key == k) ? &x : nullptr;
  }
  void store(const Key &k, float score, int depth, int bound, int move) {
    TTEntry &x = e[h(k) & mask];
    x = TTEntry{k, score, static_cast<int8_t>(depth), static_cast<int8_t>(bound),
                static_cast<int8_t>(move), 1};
  }
};

// ---- Network ---------------------------------------------------------------

const Mlp *g_net = nullptr;

// Value for the side to move (sign verified in phase A), and legal-masked,
// renormalized policy.
float netEval(const Game &g, const MoveMask &legal, float *pol) {
  float state[kGameStateSize];
  float probs[kNumMoves];
  float v;
  g.writeGameState(state);
  g_net->evaluate(state, 1, &v, probs);
  if (pol) {
    float sum = 0;
    for (int m = 0; m < kNumMoves; ++m) {
      pol[m] = legal.test(m) ? probs[m] : 0.0F;
      sum += pol[m];
    }
    if (sum > 0)
      for (int m = 0; m < kNumMoves; ++m) pol[m] /= sum;
  }
  return v;
}

// ---- Phase A ---------------------------------------------------------------

struct PosRec {
  Game g;
  int ply;
  int game;
};

constexpr int kBuckets = 10;  // ply / 5, last bucket 45+
int bucketOf(int ply) { return std::min(ply / 5, kBuckets - 1); }

struct Acc {
  double n = 0, legal = 0, lines = 0, top1 = 0, ent = 0, w90 = 0, v = 0,
         vabs = 0, agree = 0, rank = 0, vbest_mass = 0, v_p1 = 0, n_p1 = 0,
         v_p2 = 0, n_p2 = 0, forced1 = 0;
  void add(const Acc &o) {
    n += o.n; legal += o.legal; lines += o.lines; top1 += o.top1; ent += o.ent;
    w90 += o.w90; v += o.v; vabs += o.vabs; agree += o.agree; rank += o.rank;
    vbest_mass += o.vbest_mass; v_p1 += o.v_p1; n_p1 += o.n_p1;
    v_p2 += o.v_p2; n_p2 += o.n_p2; forced1 += o.forced1;
  }
};

// Score of child position from the parent's perspective, using the network
// for non-terminal children. Terminal: side to move in child has no moves.
float childScore(const Game &child, float sign) {
  MoveMask m;
  const bool lines = child.getLegalMoves(m);
  if (!m.any()) return lines ? 1.0F : 0.0F;  // child to move lost / draw
  return -sign * netEval(child, m, nullptr);
}

struct GameOut {
  std::vector<PosRec> pos;
  int winner = -1;  // 0 first player, 1 second, -1 draw
  int plies = 0;
};

// Play one game. mode 0: policy (T=1 for open plies, then greedy);
// mode 1: uniform random.
GameOut playGame(int mode, uint64_t seed, int open_plies, int gi) {
  std::mt19937_64 rng(seed);
  GameOut out;
  Game g;
  float pol[kNumMoves];
  for (int ply = 0;; ++ply) {
    MoveMask m;
    const bool lines = g.getLegalMoves(m);
    if (!m.any()) {
      out.plies = ply;
      // side to move = ply % 2
      out.winner = lines ? 1 - (ply % 2) : -1;
      return out;
    }
    out.pos.push_back({g, ply, gi});
    int choice = -1;
    if (mode == 1) {
      int k = std::uniform_int_distribution<int>(0, m.count() - 1)(rng);
      forEachMove(m, [&](int mv) {
        if (k-- == 0) choice = mv;
      });
    } else {
      netEval(g, m, pol);
      if (ply < open_plies) {
        std::discrete_distribution<int> d(pol, pol + kNumMoves);
        choice = d(rng);
      } else {
        choice = static_cast<int>(std::max_element(pol, pol + kNumMoves) - pol);
      }
    }
    g.doMove(choice);
  }
}

// Evaluate one position's statistics into acc.
// sign converts the network's output to side-to-move perspective.
void positionStats(const PosRec &p, float sign, Acc &acc, Acc &sign_check) {
  MoveMask m;
  const bool lines = p.g.getLegalMoves(m);
  float pol[kNumMoves];
  const float v = sign * netEval(p.g, m, pol);
  const int nl = m.count();
  acc.n += 1;
  acc.legal += nl;
  acc.lines += lines;
  acc.forced1 += (nl == 1);
  acc.v += v;
  acc.vabs += std::fabs(v);
  if (p.ply % 2 == 0) {
    acc.v_p1 += v;
    acc.n_p1 += 1;
  } else {
    acc.v_p2 += v;
    acc.n_p2 += 1;
  }
  // policy sharpness
  std::vector<std::pair<float, int>> order;
  forEachMove(m, [&](int mv) { order.push_back({pol[mv], mv}); });
  std::sort(order.begin(), order.end(),
            [](auto &x, auto &y) { return x.first > y.first; });
  acc.top1 += order[0].first;
  double ent = 0;
  for (auto &o : order)
    if (o.first > 0) ent -= o.first * std::log(o.first);
  acc.ent += ent;
  double cum = 0;
  int w = 0;
  for (auto &o : order) {
    cum += o.first;
    ++w;
    if (cum >= 0.9) break;
  }
  acc.w90 += w;
  // one-ply value lookahead
  float best = -1e9F;
  int best_mv = -1;
  bool has_win = false;
  forEachMove(m, [&](int mv) {
    Game c = p.g;
    c.doMove(mv);
    MoveMask cm;
    const bool cl = c.getLegalMoves(cm);
    if (!cm.any() && cl) has_win = true;
    const float s = childScore(c, sign);
    if (s > best) {
      best = s;
      best_mv = mv;
    }
  });
  acc.agree += (order[0].second == best_mv);
  for (size_t i = 0; i < order.size(); ++i)
    if (order[i].second == best_mv) {
      acc.rank += static_cast<double>(i + 1);
      acc.vbest_mass += order[i].first;
    }
  if (has_win) {
    sign_check.n += 1;
    sign_check.v += v;
  }
}

// ---- Phase B: alpha-beta with the network -----------------------------------

constexpr float kWin = 10.0F;
constexpr int kMaxPly = 64;

struct AB {
  TT tt{20};
  bool policy_order = false;
  bool line_ext = true;
  float sign = 1.0F;
  uint64_t nodes = 0, leaf_evals = 0, inner_evals = 0, tt_cuts = 0;
  uint64_t cap = 0;
  bool aborted = false;

  float search(const Game &g, int depth, float alpha, float beta, int ply) {
    ++nodes;
    if (cap && leaf_evals + inner_evals > cap) {
      aborted = true;
      return 0;
    }
    MoveMask m;
    const bool lines = g.getLegalMoves(m);
    if (!m.any()) return lines ? -(kWin - ply * 0.01F) : 0.0F;
    if (depth <= 0 && !(line_ext && lines && ply < kMaxPly)) {
      ++leaf_evals;
      return sign * netEval(g, m, nullptr);
    }
    const int d = depth <= 0 ? 0 : depth;  // extension keeps depth at 0
    const Key k = rawKey(g);
    const float a0 = alpha;
    int tt_move = -1;
    if (TTEntry *e = tt.probe(k)) {
      tt_move = e->move;
      if (e->depth >= d) {
        if (e->bound == kExact ||
            (e->bound == kLower && e->score >= beta) ||
            (e->bound == kUpper && e->score <= alpha)) {
          ++tt_cuts;
          return e->score;
        }
      }
    }
    int moves[kNumMoves];
    int n = 0;
    if (policy_order) {
      float pol[kNumMoves];
      ++inner_evals;
      netEval(g, m, pol);
      forEachMove(m, [&](int mv) { moves[n++] = mv; });
      std::sort(moves, moves + n,
                [&](int x, int y) { return pol[x] > pol[y]; });
    } else {
      forEachMove(m, [&](int mv) { moves[n++] = mv; });
    }
    if (tt_move >= 0)
      for (int i = 0; i < n; ++i)
        if (moves[i] == tt_move) {
          std::rotate(moves, moves + i, moves + i + 1);
          break;
        }
    float best = -1e9F;
    int best_mv = moves[0];
    for (int i = 0; i < n; ++i) {
      Game c = g;
      c.doMove(moves[i]);
      const float s = -search(c, d - 1, -beta, -alpha, ply + 1);
      if (aborted) return 0;
      if (s > best) {
        best = s;
        best_mv = moves[i];
      }
      if (s > alpha) alpha = s;
      if (alpha >= beta) break;
    }
    const int bound = best <= a0 ? kUpper : best >= beta ? kLower : kExact;
    tt.store(k, best, d, bound, best_mv);
    return best;
  }
};

// ---- Phase C: solver -----------------------------------------------------------

struct Solver {
  TT tt{21};
  bool sym = false;
  uint64_t nodes = 0, cap = 0;
  bool aborted = false;

  // Returns -1/0/+1 for side to move. Scores are exact game values, so the
  // table stores bounds on {-1,0,1}. Depth field unused (always to the end).
  int solve(const Game &g, int alpha, int beta) {
    ++nodes;
    if (cap && nodes > cap) {
      aborted = true;
      return 0;
    }
    MoveMask m;
    const bool lines = g.getLegalMoves(m);
    if (!m.any()) return lines ? -1 : 0;
    int t = 0;
    const Key k = sym ? canonKey(g, t) : rawKey(g);
    const int a0 = alpha;
    int tt_move = -1;
    if (TTEntry *e = tt.probe(k)) {
      const int s = static_cast<int>(e->score);
      if (e->bound == kExact || (e->bound == kLower && s >= beta) ||
          (e->bound == kUpper && s <= alpha))
        return s;
      if (e->bound == kLower) alpha = std::max(alpha, s);
      if (e->bound == kUpper) beta = std::min(beta, s);
      // stored move is in canonical frame; map back to this position's frame
      tt_move = (sym && e->move >= 0) ? kMovePerm[kInvT[t]][e->move] : e->move;
    }
    int moves[kNumMoves];
    int n = 0;
    // Move-ID order, TT move first. No other ordering: this measures the
    // floor of what a solver needs.
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
      if (aborted) return 0;
      if (s > best) {
        best = s;
        best_mv = moves[i];
      }
      if (s > alpha) alpha = s;
      if (alpha >= beta) break;
    }
    const int bound = best <= a0 ? kUpper : best >= beta ? kLower : kExact;
    const int stored_mv = sym ? kMovePerm[t][best_mv] : best_mv;
    tt.store(k, static_cast<float>(best), 0, bound, stored_mv);
    return best;
  }
};

void printAcc(const char *title, const std::vector<Acc> &b) {
  std::printf("\n%s\n", title);
  std::printf(
      "plies  | positions | legal | lines%% | forced1%% | top1  | entropy | "
      "w90  | pol=1ply%% | rank(1ply-best) | pmass(1ply-best) | v(stm)  | "
      "|v|  | v P1-to-move | v P2-to-move\n");
  Acc tot;
  for (int i = 0; i < kBuckets; ++i) {
    const Acc &a = b[i];
    tot.add(a);
    if (a.n == 0) continue;
    std::printf(
        "%2d-%-3s | %9.0f | %5.1f | %5.1f | %7.1f | %.3f | %7.3f | %4.1f | "
        "%8.1f | %15.2f | %16.3f | %+.3f | %.3f | %+12.3f | %+12.3f\n",
        i * 5, i == kBuckets - 1 ? "+" : std::to_string(i * 5 + 4).c_str(),
        a.n, a.legal / a.n, 100 * a.lines / a.n, 100 * a.forced1 / a.n,
        a.top1 / a.n, a.ent / a.n, a.w90 / a.n, 100 * a.agree / a.n,
        a.rank / a.n, a.vbest_mass / a.n, a.v / a.n, a.vabs / a.n,
        a.n_p1 ? a.v_p1 / a.n_p1 : 0, a.n_p2 ? a.v_p2 / a.n_p2 : 0);
  }
  const Acc &a = tot;
  std::printf(
      "all    | %9.0f | %5.1f | %5.1f | %7.1f | %.3f | %7.3f | %4.1f | "
      "%8.1f | %15.2f | %16.3f | %+.3f | %.3f | %+12.3f | %+12.3f\n",
      a.n, a.legal / a.n, 100 * a.lines / a.n, 100 * a.forced1 / a.n,
      a.top1 / a.n, a.ent / a.n, a.w90 / a.n, 100 * a.agree / a.n,
      a.rank / a.n, a.vbest_mass / a.n, a.v / a.n, a.vabs / a.n,
      a.n_p1 ? a.v_p1 / a.n_p1 : 0, a.n_p2 ? a.v_p2 / a.n_p2 : 0);
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model.mlp> [games] [threads]\n", argv[0]);
    return 1;
  }
  const int num_games = argc > 2 ? std::atoi(argv[2]) : 2000;
  const int threads = argc > 3 ? std::atoi(argv[3]) : 20;
  const int open_plies = envInt("OPEN_PLIES", 6);
  const int ab_maxdepth = envInt("AB_MAXDEPTH", 6);
  const int ab_pos = envInt("AB_POS", 12);
  const int solve_pos = envInt("SOLVE_POS", 24);
  const uint64_t solve_cap = static_cast<uint64_t>(envInt("SOLVE_CAP", 20000000));
  omp_set_num_threads(threads);
  initSymmetry();

  Mlp net(argv[1]);
  g_net = &net;
  std::printf("model %s: input %d, moves %d\n", argv[1], net.inputSize(),
              net.numMoves());

  // Single-row network cost, for scale.
  {
    Game g;
    MoveMask m;
    g.getLegalMoves(m);
    const double t0 = now();
    float s = 0;
    for (int i = 0; i < 20000; ++i) s += netEval(g, m, nullptr);
    std::printf("single-row network eval: %.2f us (checksum %.1f)\n",
                (now() - t0) / 20000 * 1e6, s);
    const double t1 = now();
    uint64_t c = 0;
    for (int i = 0; i < 2000000; ++i) {
      MoveMask mm;
      g.getLegalMoves(mm);
      c += mm.lo;
      g = Game{};
    }
    std::printf("getLegalMoves (start position): %.1f ns (checksum %llu)\n",
                (now() - t1) / 2000000 * 1e9,
                static_cast<unsigned long long>(c & 0xFF));
  }

  // ---- Phase A ----
  std::vector<GameOut> games[2];
  for (int mode = 0; mode < 2; ++mode) {
    games[mode].resize(num_games);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_games; ++i)
      games[mode][i] = playGame(mode, 1000003ULL * (i + 1) + mode, open_plies, i);
  }
  for (int mode = 0; mode < 2; ++mode) {
    int w[3] = {0, 0, 0};
    double len = 0;
    std::vector<int> hist(50, 0);
    for (auto &go : games[mode]) {
      w[go.winner + 1]++;
      len += go.plies;
      hist[std::min(go.plies, 49)]++;
    }
    std::printf("\n%s games: %d, P1 %.3f draw %.3f P2 %.3f, mean plies %.2f\n",
                mode == 0 ? "policy" : "random", num_games,
                double(w[1]) / num_games, double(w[0]) / num_games,
                double(w[2]) / num_games, len / num_games);
    std::printf("  length histogram:");
    for (int i = 0; i < 50; ++i)
      if (hist[i]) std::printf(" %d:%d", i, hist[i]);
    std::printf("\n");
  }

  // Sign: over positions where the side to move has an immediate win,
  // the network's raw value should be strongly positive if it is from the
  // side to move's perspective.
  float sign = 1.0F;
  {
    std::vector<Acc> sc(threads);
    std::vector<Acc> dm(threads);
    auto &pos0 = games[0];
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_games; ++i)
      for (auto &p : pos0[i].pos)
        positionStats(p, 1.0F, dm[omp_get_thread_num()], sc[omp_get_thread_num()]);
    Acc s;
    for (auto &x : sc) s.add(x);
    std::printf("\nsign check: %0.f positions with an immediate win, mean raw "
                "value %+.3f\n", s.n, s.n ? s.v / s.n : 0);
    sign = (s.n && s.v < 0) ? -1.0F : 1.0F;
    std::printf("=> value is from the %s's perspective\n",
                sign > 0 ? "side to move" : "side that just moved");
  }

  for (int mode = 0; mode < 2; ++mode) {
    std::vector<std::vector<Acc>> per(threads, std::vector<Acc>(kBuckets));
    std::vector<Acc> sc(threads);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_games; ++i)
      for (auto &p : games[mode][i].pos)
        positionStats(p, sign, per[omp_get_thread_num()][bucketOf(p.ply)],
                      sc[omp_get_thread_num()]);
    std::vector<Acc> b(kBuckets);
    for (auto &t : per)
      for (int i = 0; i < kBuckets; ++i) b[i].add(t[i]);
    printAcc(mode == 0 ? "PHASE A: positions from policy games"
                       : "PHASE A: positions from random games",
             b);
  }

  // Sample positions for B and C from policy games, by ply.
  auto sample = [&](int ply, int count, uint64_t seed) {
    std::vector<PosRec> out;
    std::mt19937_64 rng(seed);
    std::vector<const PosRec *> cands;
    for (auto &go : games[0])
      for (auto &p : go.pos)
        if (p.ply == ply) cands.push_back(&p);
    std::shuffle(cands.begin(), cands.end(), rng);
    for (int i = 0; i < count && i < static_cast<int>(cands.size()); ++i)
      out.push_back(*cands[i]);
    return out;
  };

  // ---- Phase B ----
  std::printf("\nPHASE B: alpha-beta, network value at leaves, TT, line "
              "extension, iterative deepening to depth %d\n", ab_maxdepth);
  std::printf("(per depth: mean over positions of cumulative nodes, network "
              "calls, ms; EBF = calls(d)/calls(d-1))\n");
  const int ab_plies[] = {0, 4, 8, 12, 16, 20};
  for (int order = 0; order < 2; ++order) {
    std::printf("\n  ordering: %s\n", order ? "policy" : "move ID");
    std::printf("  ply | pos | depth | nodes      | net calls  | ms/pos   | EBF  "
                "| best=policy-top%%\n");
    for (int ply : ab_plies) {
      auto ps = sample(ply, ab_pos, 77 + ply);
      if (ps.empty()) continue;
      const int np = static_cast<int>(ps.size());
      std::vector<std::vector<double>> nodes(np), calls(np), ms(np);
      std::vector<std::vector<int>> agree(np);
#pragma omp parallel for schedule(dynamic)
      for (int i = 0; i < np; ++i) {
        AB ab;
        ab.policy_order = order == 1;
        ab.sign = sign;
        ab.cap = 30000000;
        MoveMask m;
        ps[i].g.getLegalMoves(m);
        float pol[kNumMoves];
        netEval(ps[i].g, m, pol);
        const int ptop = static_cast<int>(std::max_element(pol, pol + kNumMoves) - pol);
        const double t0 = now();
        for (int d = 1; d <= ab_maxdepth; ++d) {
          ab.search(ps[i].g, d, -1e9F, 1e9F, 0);
          if (ab.aborted) break;
          nodes[i].push_back(double(ab.nodes));
          calls[i].push_back(double(ab.leaf_evals + ab.inner_evals));
          ms[i].push_back((now() - t0) * 1e3);
          TTEntry *e = ab.tt.probe(rawKey(ps[i].g));
          agree[i].push_back(e && e->move == ptop);
        }
      }
      for (int d = 1; d <= ab_maxdepth; ++d) {
        double sn = 0, sc = 0, sm = 0, sa = 0, prev = 0;
        int cnt = 0;
        for (int i = 0; i < np; ++i) {
          if (static_cast<int>(nodes[i].size()) < d) continue;
          ++cnt;
          sn += nodes[i][d - 1];
          sc += calls[i][d - 1];
          sm += ms[i][d - 1];
          sa += agree[i][d - 1];
          prev += d > 1 ? calls[i][d - 2] : 1;
        }
        if (!cnt) break;
        std::printf("  %3d | %3d | %5d | %10.0f | %10.0f | %8.1f | %4.1f | %5.1f\n",
                    ply, cnt, d, sn / cnt, sc / cnt, sm / cnt,
                    d > 1 ? sc / prev : 0.0, 100.0 * sa / cnt);
      }
    }
  }

  // ---- Phase C ----
  std::printf("\nPHASE C: exact solver (win/draw/loss to game end), node cap "
              "%llu\n", static_cast<unsigned long long>(solve_cap));
  std::printf("  ply | sym | pos | solved | mean nodes (solved) | max nodes | "
              "ms (solved) | W/D/L stm | net sign agrees%% | net |v| on solved\n");
  const int solve_plies[] = {32, 28, 24, 20, 16, 12, 8};
  for (int ply : solve_plies) {
    auto ps = sample(ply, solve_pos, 991 + ply);
    if (ps.empty()) continue;
    for (int sym = 0; sym < 2; ++sym) {
      const int np = static_cast<int>(ps.size());
      std::vector<int> res(np), ok(np);
      std::vector<double> nodes(np), ms(np);
      std::vector<float> nv(np);
#pragma omp parallel for schedule(dynamic)
      for (int i = 0; i < np; ++i) {
        Solver s;
        s.sym = sym == 1;
        s.cap = solve_cap;
        const double t0 = now();
        res[i] = s.solve(ps[i].g, -1, 1);
        ms[i] = (now() - t0) * 1e3;
        ok[i] = !s.aborted;
        nodes[i] = double(s.nodes);
        MoveMask m;
        ps[i].g.getLegalMoves(m);
        nv[i] = sign * netEval(ps[i].g, m, nullptr);
      }
      int solved = 0, w = 0, d = 0, l = 0, agree = 0, decisive = 0;
      double sn = 0, sm = 0, mx = 0, sv = 0;
      for (int i = 0; i < np; ++i) {
        mx = std::max(mx, nodes[i]);
        if (!ok[i]) continue;
        ++solved;
        sn += nodes[i];
        sm += ms[i];
        sv += std::fabs(nv[i]);
        (res[i] > 0 ? w : res[i] < 0 ? l : d)++;
        if (res[i] != 0) {
          ++decisive;
          agree += (res[i] > 0) == (nv[i] > 0);
        }
      }
      std::printf("  %3d | %3s | %3d | %6d | %19.0f | %9.0f | %11.1f | %2d/%d/%-2d  "
                  "| %15.1f | %.3f\n",
                  ply, sym ? "yes" : "no", np, solved,
                  solved ? sn / solved : 0, mx, solved ? sm / solved : 0, w, d, l,
                  decisive ? 100.0 * agree / decisive : 0,
                  solved ? sv / solved : 0);
      std::fflush(stdout);
    }
  }

  // ---- Phase E: depth-limited proof search ----
  // How many plies ahead can alpha-beta prove a forced result? Win/loss search
  // to a fixed depth (0 = not proven within the depth, including draws),
  // iterative deepening, TT, TT move first, move-ID order otherwise. Proven
  // wins and losses are stored as depth-independent.
  const int e_pos = envInt("E_POS", 0);
  if (e_pos > 0) {
    const int e_maxd = envInt("E_MAXD", 24);
    const double e_cap_ms = envInt("E_CAP_MS", 10000);
    struct Prover {
      TT tt{22};
      uint64_t nodes = 0;
      double deadline = 0;
      bool aborted = false;
      int search(const Game &g, int depth, int alpha, int beta) {
        ++nodes;
        if ((nodes & 0xFFFF) == 0 && now() > deadline) aborted = true;
        if (aborted) return 0;
        MoveMask m;
        const bool lines = g.getLegalMoves(m);
        if (!m.any()) return lines ? -1 : 0;
        if (depth == 0) return 0;
        const Key k = rawKey(g);
        const int a0 = alpha;
        int tt_move = -1;
        if (TTEntry *e = tt.probe(k)) {
          tt_move = e->move;
          const int s = static_cast<int>(e->score);
          if (e->depth >= depth &&
              (e->bound == kExact || (e->bound == kLower && s >= beta) ||
               (e->bound == kUpper && s <= alpha)))
            return s;
        }
        int moves[kNumMoves], n = 0;
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
          const int s = -search(c, depth - 1, -beta, -alpha);
          if (aborted) return 0;
          if (s > best) {
            best = s;
            best_mv = moves[i];
          }
          if (s > alpha) alpha = s;
          if (alpha >= beta) break;
        }
        const int bound = best <= a0 ? kUpper : best >= beta ? kLower : kExact;
        // A proven win (lower bound 1) or loss (upper bound -1) holds at any
        // depth; everything else only at this depth.
        const bool proven = (best == 1 && bound != kUpper) ||
                            (best == -1 && bound != kLower);
        tt.store(k, static_cast<float>(best), proven ? 127 : depth, bound,
                 best_mv);
        return best;
      }
    };
    std::printf("\nPHASE E: depth-limited proof search (win/loss within depth), "
                "cap %.0f ms per position\n", e_cap_ms);
    std::printf("  ply | depth | completed | proven by now | median ms (completed) "
                "| max ms | mean nodes\n");
    for (int ply : {4, 8, 12, 16, 20}) {
      auto ps = sample(ply, e_pos, 31337 + ply);
      const int np = static_cast<int>(ps.size());
      if (!np) continue;
      // per position: time and nodes at each completed depth, depth proven
      std::vector<std::vector<double>> tms(np), tnodes(np);
      std::vector<int> proven_at(np, -1);
#pragma omp parallel for schedule(dynamic)
      for (int i = 0; i < np; ++i) {
        Prover pr;
        const double t0 = now();
        pr.deadline = t0 + e_cap_ms / 1e3;
        for (int d = 1; d <= e_maxd; ++d) {
          const int v = pr.search(ps[i].g, d, -1, 1);
          if (pr.aborted) break;
          tms[i].push_back((now() - t0) * 1e3);
          tnodes[i].push_back(double(pr.nodes));
          if (v != 0) {
            proven_at[i] = d;
            break;
          }
        }
      }
      for (int d = 1; d <= e_maxd; ++d) {
        int done = 0, proven = 0;
        std::vector<double> t;
        double mx = 0, sn = 0;
        for (int i = 0; i < np; ++i) {
          if (proven_at[i] >= 1 && proven_at[i] <= d) {
            ++proven;
            ++done;
            continue;
          }
          if (static_cast<int>(tms[i].size()) >= d) {
            ++done;
            t.push_back(tms[i][d - 1]);
            mx = std::max(mx, tms[i][d - 1]);
            sn += tnodes[i][d - 1];
          }
        }
        if (t.empty() && proven == done) {
          if (done < np) break;
          continue;
        }
        std::sort(t.begin(), t.end());
        std::printf("  %3d | %5d | %5d/%-3d | %13d | %21.1f | %6.0f | %10.0f\n",
                    ply, d, done, np, proven, t.empty() ? 0 : t[t.size() / 2], mx,
                    t.empty() ? 0 : sn / t.size());
        if (t.empty()) break;
      }
      std::fflush(stdout);
    }
  }

  // ---- Phase D: solver by reserve pieces left ----
  // Every move lowers 2 * reserves + occupied squares by at least one, so
  // that quantity bounds the plies left. Bucket by reserves rather than ply.
  const int res_pos = envInt("RES_POS", 24);
  if (res_pos > 0) {
    auto reservesLeft = [](const Game &g) {
      int8_t p[6];
      std::memcpy(p, reinterpret_cast<const char *>(&g) + 8, 6);
      int r = 0;
      for (int i = 0; i < 6; ++i) r += p[i];
      return r;
    };
    auto occupied = [](const Game &g) {
      uint64_t b;
      std::memcpy(&b, &g, 8);
      int m = 0;
      for (int s = 0; s < 16; ++s) m += ((b >> (4 * s)) & kStackMask) != 0;
      return m;
    };
    std::printf("\nPHASE D: exact solver by reserve pieces left (both players), "
                "node cap %llu\n", static_cast<unsigned long long>(solve_cap));
    std::printf("  reserves | pos | solved | horizon 2R+m | plies left in game | "
                "median ms | p90 ms | max ms | mean nodes | W/D/L stm | net sign agrees%%\n");
    for (int R = 0; R <= 14; ++R) {
      std::vector<const PosRec *> cands;
      for (auto &go : games[0])
        for (auto &p : go.pos)
          if (reservesLeft(p.g) == R) cands.push_back(&p);
      if (cands.empty()) continue;
      std::mt19937_64 rng(4242 + R);
      std::shuffle(cands.begin(), cands.end(), rng);
      const int np = std::min(res_pos, static_cast<int>(cands.size()));
      std::vector<int> res(np), ok(np), hor(np), left(np);
      std::vector<double> nodes(np), ms(np);
      std::vector<float> nv(np);
#pragma omp parallel for schedule(dynamic)
      for (int i = 0; i < np; ++i) {
        const PosRec &p = *cands[i];
        Solver s;
        s.cap = solve_cap;
        const double t0 = now();
        res[i] = s.solve(p.g, -1, 1);
        ms[i] = (now() - t0) * 1e3;
        ok[i] = !s.aborted;
        nodes[i] = double(s.nodes);
        hor[i] = 2 * R + occupied(p.g);
        left[i] = games[0][p.game].plies - p.ply;
        MoveMask m;
        p.g.getLegalMoves(m);
        nv[i] = sign * netEval(p.g, m, nullptr);
      }
      if (const char *dump = std::getenv("RES_DUMP")) {
        if (FILE *df = std::fopen(dump, "a")) {
          for (int i = 0; i < np; ++i)
            std::fprintf(df, "%d %d %d %d %.0f %.3f %d\n", R, hor[i] - 2 * R,
                         hor[i], ok[i], nodes[i], ms[i], left[i]);
          std::fclose(df);
        }
      }
      std::vector<double> sms;
      int solved = 0, w = 0, d = 0, l = 0, agree = 0, decisive = 0;
      double sn = 0, sh = 0, sl = 0;
      for (int i = 0; i < np; ++i) {
        sh += hor[i];
        sl += left[i];
        if (!ok[i]) continue;
        ++solved;
        sms.push_back(ms[i]);
        sn += nodes[i];
        (res[i] > 0 ? w : res[i] < 0 ? l : d)++;
        if (res[i] != 0) {
          ++decisive;
          agree += (res[i] > 0) == (nv[i] > 0);
        }
      }
      std::sort(sms.begin(), sms.end());
      auto q = [&](double f) {
        return sms.empty() ? 0.0 : sms[std::min(sms.size() - 1, size_t(f * sms.size()))];
      };
      std::printf("  %8d | %3d | %6d | %12.1f | %18.1f | %9.2f | %6.1f | %6.1f | "
                  "%10.0f | %2d/%d/%-2d | %6.1f\n",
                  R, np, solved, sh / np, sl / np, q(0.5), q(0.9),
                  sms.empty() ? 0.0 : sms.back(), solved ? sn / solved : 0, w, d, l,
                  decisive ? 100.0 * agree / decisive : 0);
      std::fflush(stdout);
    }
  }
  return 0;
}
