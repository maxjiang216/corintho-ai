// Preliminary match: alpha-beta with a small value network against MCTS
// with an AZ network, on the CPU, one thread per game for both sides
// (worklog 2026-09-25-nn-architectures, entry 22).
//
//   ab_match --small NET.bin --az MODEL.onnx [--games 300] [--threads 18]
//            [--ab-seconds 2] [--mcts-searches 1600] [--opening 4]
//            [--solve-p 27] [--seed 1]
//
// Openings: --opening random legal moves from the start, each opening played
// twice with colours swapped. A game is adjudicated by exact solution once
// its horizon P is at most --solve-p, for both sides alike. MCTS: the
// engine's TrainMC (testing mode, c_puct 3, noise 0.25, as in test
// matches), a fixed number of searches per move, the network through ONNX
// Runtime on the CPU (one intra-op thread). Alpha-beta: iterative
// deepening negamax with a transposition table, --ab-seconds per move, the
// small network at the leaves.

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "game.h"
#include "move.h"
#include "solver.h"
#include "trainmc.h"
#include "util.h"

#ifdef AB_PROFILE
#include <x86intrin.h>
// Cycles by part of an alpha-beta node (worklog entry 22): move generation,
// network evaluation, move ordering, child setup; and counts
thread_local uint64_t g_prof[8];
#define PROF_START(v) const uint64_t v = __rdtsc()
#define PROF_ADD(i, v) (g_prof[i] += __rdtsc() - (v))
#else
#define PROF_START(v) ((void)0)
#define PROF_ADD(i, v) ((void)0)
#endif

namespace {

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) {
  return std::chrono::duration<double>(Clock::now() - t).count();
}

/// 70 -> H -> 32 -> 1, clipped ReLU, tanh (arch/nnue_distill.py Small)
struct SmallNet {
  int32_t h{0};
  // Input-major ([70][h], [h][32]): each input updates independent outputs,
  // which vectorizes without reassociating sums
  std::vector<float> w1, b1, w2, b2, w3;
  float b3{0};

  explicit SmallNet(const std::string &path) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (f == nullptr || std::fread(&h, 4, 1, f) != 1) {
      std::fprintf(stderr, "cannot read %s\n", path.c_str());
      std::exit(1);
    }
    auto read = [&](std::vector<float> &v, size_t n) {
      v.resize(n);
      if (std::fread(v.data(), 4, n, f) != n)
        std::exit(1);
    };
    read(w1, 70 * static_cast<size_t>(h));
    read(b1, h);
    std::vector<float> w2_rows;  // stored [32][h]
    read(w2_rows, 32 * static_cast<size_t>(h));
    w2.resize(w2_rows.size());
    for (int32_t k = 0; k < 32; ++k)
      for (int32_t j = 0; j < h; ++j)
        w2[static_cast<size_t>(j) * 32 + k] =
            w2_rows[static_cast<size_t>(k) * h + j];
    read(b2, 32);
    read(w3, 32);
    if (std::fread(&b3, 4, 1, f) != 1)
      std::exit(1);
    std::fclose(f);
  }

  /// Value for the side to move
  float eval(const Game &game) const {
    float x[kGameStateSize];
    game.writeGameState(x);
    std::vector<float> &a = scratch();
    a.assign(b1.begin(), b1.end());
    for (int32_t i = 0; i < 70; ++i) {
      if (x[i] == 0.0F)
        continue;
      const float xi = x[i];
      const float *w = w1.data() + static_cast<size_t>(i) * h;
      for (int32_t j = 0; j < h; ++j)
        a[j] += xi * w[j];
    }
    float h2[32];
    std::copy(b2.begin(), b2.end(), h2);
    for (int32_t j = 0; j < h; ++j) {
      const float aj = std::min(std::max(a[j], 0.0F), 1.0F);
      if (aj == 0.0F)
        continue;  // clipped ReLU: many activations are 0
      const float *w = w2.data() + static_cast<size_t>(j) * 32;
      for (int32_t k = 0; k < 32; ++k)
        h2[k] += aj * w[k];
    }
    float out = b3;
    for (int32_t k = 0; k < 32; ++k)
      out += std::min(std::max(h2[k], 0.0F), 1.0F) * w3[k];
    return std::tanh(out);
  }
  static std::vector<float> &scratch() {
    thread_local std::vector<float> v;
    return v;
  }
};

/// Iterative-deepening negamax with a transposition table
class AlphaBeta {
 public:
  AlphaBeta(const SmallNet &net, int32_t log2_table)
      : net_{net},
        table_(size_t{1} << log2_table), mask_{(uint64_t{1} << log2_table) -
                                               1} {}

  struct Result {
    int32_t move, depth;
    float score;
    uint64_t nodes;
  };

  Result choose(const Game &root, double seconds) {
    start_ = Clock::now();
    seconds_ = seconds;
    aborted_ = false;
    nodes_ = 0;
    std::memset(history_, 0, sizeof(history_));
    MoveMask legal;
    root.getLegalMoves(legal);
    std::vector<int32_t> moves;
    forEachMove(legal, [&](int32_t m) { moves.push_back(m); });
    Result best{moves[0], 0, 0.0F, 0};
    for (int32_t depth = 1; depth <= 64; ++depth) {
      float alpha = -kInf, best_score = -kInf;
      int32_t best_move = moves[0], done = 0;
      for (int32_t m : moves) {
        Game child = root;
        child.doMove(m);
        const float s = -search(child, depth - 1, -kInf, -alpha, 1);
        if (aborted_)
          break;
        ++done;
        if (s > best_score) {
          best_score = s;
          best_move = m;
        }
        alpha = std::max(alpha, s);
      }
      if (aborted_) {
        // Out of time mid-iteration: the previous best (searched first) and
        // any move that beat it at this depth are valid results
        if (done > 0 && best_move != best.move)
          best.move = best_move;
        break;
      }
      best = {best_move, depth, best_score, nodes_};
      // The best move first in the next iteration
      std::stable_partition(moves.begin(), moves.end(),
                            [&](int32_t m) { return m == best_move; });
      if (std::abs(best_score) > 1.5F)
        break;  // a forced result
    }
    best.nodes = nodes_;
    return best;
  }

 private:
  static constexpr float kInf = 1e9F;
  static constexpr float kMate = 2.0F;  // beyond any network value

  struct Entry {
    uint64_t board{0}, rest{~uint64_t{0}};
    float score{0};
    int8_t depth{-1}, flag{0}, move{-1};
  };
  enum : int8_t { kExact, kLower, kUpper };

  float search(const Game &game, int32_t depth, float alpha, float beta,
               int32_t ply) {
    if ((++nodes_ & 1023) == 0 && since(start_) > seconds_)
      aborted_ = true;
    if (aborted_)
      return 0.0F;
    PROF_START(t_gen);
    MoveMask legal;
    const bool lines = game.getLegalMoves(legal);
    PROF_ADD(0, t_gen);
    if (!legal.any())  // lost if a line stands, else drawn; prefer quick wins
      return lines ? -(kMate - 0.01F * static_cast<float>(ply)) : 0.0F;
    // A standing line forces the side to move to break it (like check):
    // keep searching instead of evaluating mid-sequence (AB_LINE_EXTEND=0
    // turns it off)
    if (depth <= 0 && !(lines && line_extend_)) {
      PROF_START(t_eval);
      const float v = net_.eval(game);
      PROF_ADD(1, t_eval);
#ifdef AB_PROFILE
      ++g_prof[4];  // leaves evaluated
#endif
      return v;
    }
    uint64_t board, rest;
    game.key(board, rest);
    Entry &e = table_[(board * 0x9E3779B97F4A7C15ULL ^ rest) & mask_];
    int32_t tt_move = -1;
    if (e.board == board && e.rest == rest) {
      tt_move = e.move;
      if (e.depth >= depth) {
        if (e.flag == kExact || (e.flag == kLower && e.score >= beta) ||
            (e.flag == kUpper && e.score <= alpha))
          return e.score;
      }
    }
    int32_t moves[kNumMoves], n = 0;
    forEachMove(legal, [&](int32_t m) { moves[n++] = m; });
    const int32_t side = static_cast<int32_t>(rest & 1);
    // Order (the solver's, entries 08 and 10): the table's move; moves
    // that make a line, fewest opponent replies first (none: an immediate
    // win); then quiet moves: move a stack, place a base, a column, a
    // capital, each group by history. AB_ORDER_DEPTH: from this depth up,
    // by the network's value of each child instead.
    uint64_t key[kNumMoves];
    PROF_START(t_order);
    for (int32_t i = 0; i < n; ++i) {
      const int32_t m = moves[i];
      if (m == tt_move) {
        key[m] = uint64_t{3} << 60;
        continue;
      }
      if (depth >= order_depth_) {
        Game child = game;
        child.doMove(m);
        key[m] = static_cast<uint64_t>((1.0F - net_.eval(child)) * 1e6F);
        key[m] = (uint64_t{2} << 60) - key[m];
        continue;
      }
      if (solver_order_) {
        Game child = game;
        child.doMove(m);
        if (child.hasLine()) {
          MoveMask replies;
          child.getLegalMoves(replies);
          const int32_t r = replies.count();
          if (r == 0)  // the opponent is stuck with a line: a win
            return kMate - 0.01F * static_cast<float>(ply + 1);
          key[m] =
              (uint64_t{2} << 60) + (static_cast<uint64_t>(100 - r) << 40);
          continue;
        }
        const MoveInfo &info = kMoveTable[m];
        const uint64_t rank = info.is_place ? 1 + info.piece : 0;
        key[m] =
            (uint64_t{1} << 60) + ((3 - rank) << 40) +
            std::min<uint64_t>(history_[side][m], (uint64_t{1} << 40) - 1);
        continue;
      }
      key[m] = static_cast<uint64_t>(history_[side][m]);
    }
    std::sort(moves, moves + n,
              [&](int32_t a, int32_t b) { return key[a] > key[b]; });
    PROF_ADD(2, t_order);
#ifdef AB_PROFILE
    ++g_prof[5];  // interior nodes expanded
#endif
    const float alpha0 = alpha;
    float best = -kInf;
    int32_t best_move = moves[0];
    for (int32_t i = 0; i < n; ++i) {
      PROF_START(t_child);
      Game child = game;
      child.doMove(moves[i]);
      PROF_ADD(3, t_child);
      // Principal variation search: later moves with a null window first
      float s;
      if (i == 0) {
        s = -search(child, depth - 1, -beta, -alpha, ply + 1);
      } else {
        // Late-move reductions: a late quiet move is first searched
        // shallower; only one that beats alpha gets the full depth
        const bool quiet = (key[moves[i]] >> 60) == 1;
        int32_t r = 0;
        if (lmr_ && quiet && depth >= 3 && i >= 3 && !lines)
          r = i >= 8 ? 2 : 1;
        s = -search(child, depth - 1 - r, -alpha - 1e-4F, -alpha, ply + 1);
        if (r > 0 && s > alpha && !aborted_)
          s = -search(child, depth - 1, -alpha - 1e-4F, -alpha, ply + 1);
        if (s > alpha && s < beta && !aborted_)
          s = -search(child, depth - 1, -beta, -alpha, ply + 1);
      }
      if (aborted_)
        return 0.0F;
      if (s > best) {
        best = s;
        best_move = moves[i];
      }
      if (s > alpha)
        alpha = s;
      if (alpha >= beta) {
        history_[side][moves[i]] += depth * depth;
        break;
      }
    }
    e.board = board;
    e.rest = rest;
    e.score = best;
    e.depth = static_cast<int8_t>(depth);
    e.move = static_cast<int8_t>(best_move);
    e.flag = best <= alpha0 ? kUpper : best >= beta ? kLower : kExact;
    return best;
  }

  const SmallNet &net_;
  bool lmr_{getenv("AB_LMR") == nullptr || atoi(getenv("AB_LMR")) != 0};
  bool line_extend_{getenv("AB_LINE_EXTEND") == nullptr ||
                    atoi(getenv("AB_LINE_EXTEND")) != 0};
  /// The solver's move ordering (AB_SOLVER_ORDER=0: table move and history)
  bool solver_order_{getenv("AB_SOLVER_ORDER") == nullptr ||
                     atoi(getenv("AB_SOLVER_ORDER")) != 0};
  /// Nodes with at least this depth order moves by the children's values
  int32_t order_depth_{
      getenv("AB_ORDER_DEPTH") ? atoi(getenv("AB_ORDER_DEPTH")) : 99};
  std::vector<Entry> table_;
  uint64_t mask_;
  int32_t history_[2][kNumMoves]{};
  Clock::time_point start_;
  double seconds_{0};
  bool aborted_{false};
  uint64_t nodes_{0};
};

/// The AZ network through ONNX Runtime on the CPU (a compact model: uint8
/// states in, fp32 value and fp16 policy out)
class AzNet {
 public:
  AzNet(Ort::Env &env, const std::string &path) {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = std::make_unique<Ort::Session>(env, path.c_str(), options);
  }
  void evaluate(const float *states, int32_t rows, float *values,
                float *probs) {
    std::vector<uint8_t> in(static_cast<size_t>(rows) * kGameStateSize);
    for (size_t i = 0; i < in.size(); ++i)
      in[i] = static_cast<uint8_t>(states[i] * 4.0F + 0.5F);
    const int64_t shape[2] = {rows, kGameStateSize};
    Ort::MemoryInfo info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<uint8_t>(info, in.data(),
                                                         in.size(), shape, 2);
    const char *in_names[] = {"states"};
    const char *out_names[] = {"value", "policy"};
    auto out = session_->Run(Ort::RunOptions{nullptr}, in_names, &input, 1,
                             out_names, 2);
    const float *v = out[0].GetTensorData<float>();
    std::copy(v, v + rows, values);
    const Ort::Float16_t *p = out[1].GetTensorData<Ort::Float16_t>();
    for (int32_t i = 0; i < rows * kNumMoves; ++i)
      probs[i] = p[i].ToFloat();
  }

 private:
  std::unique_ptr<Ort::Session> session_;
};

struct Args {
  std::map<std::string, std::string> kv;
  Args(int argc, char **argv) {
    for (int i = 1; i + 1 < argc; i += 2)
      kv[std::string(argv[i]).substr(2)] = argv[i + 1];
  }
  std::string str(const char *k, const char *d = "") const {
    auto it = kv.find(k);
    return it == kv.end() ? d : it->second;
  }
  double num(const char *k, double d) const {
    auto it = kv.find(k);
    return it == kv.end() ? d : std::atof(it->second.c_str());
  }
};

struct GameStats {
  float ab_score{0};  // 1 win, 0.5 draw, 0 loss, for the alpha-beta side
  int32_t ab_moves{0}, mcts_moves{0}, depth_sum{0};
  double ab_seconds{0}, mcts_seconds{0};
  uint64_t ab_nodes{0};
};

}  // namespace

int main(int argc, char **argv) {
  const Args a(argc, argv);
  const SmallNet small{a.str("small")};
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "ab_match"};
  const int32_t games = static_cast<int32_t>(a.num("games", 300));
  const int32_t threads = static_cast<int32_t>(a.num("threads", 18));
  const double ab_seconds = a.num("ab-seconds", 2.0);
  const int32_t mcts_searches =
      static_cast<int32_t>(a.num("mcts-searches", 1600));
  const int32_t opening = static_cast<int32_t>(a.num("opening", 4));
  const int32_t solve_p = static_cast<int32_t>(a.num("solve-p", 27));
  const uint32_t seed = static_cast<uint32_t>(a.num("seed", 1));
  std::vector<GameStats> stats(games);
  std::atomic<int32_t> next{0};
  std::mutex print_mutex;
  const auto t0 = Clock::now();

  auto worker = [&]() {
    AzNet az{env, a.str("az")};
    AlphaBeta ab{small, 20};
    Solver solver{20};
    std::vector<float> values(16), probs(16 * kNumMoves);
    for (int32_t g; (g = next++) < games;) {
      // Game pair g/2: the same opening, alpha-beta first in the even game
      std::mt19937 open_rng{seed * 100003U + static_cast<uint32_t>(g / 2)};
      Game game;
      int32_t depth = 0;
      for (int32_t i = 0; i < opening; ++i) {
        MoveMask legal;
        game.getLegalMoves(legal);
        std::vector<int32_t> moves;
        forEachMove(legal, [&](int32_t m) { moves.push_back(m); });
        game.doMove(moves[open_rng() % moves.size()]);
        ++depth;
      }
      const int32_t ab_side = g % 2;  // 0: alpha-beta moves first
      std::mt19937 mcts_rng{seed * 7919U + static_cast<uint32_t>(g)};
      std::vector<float> to_eval(16 * kGameStateSize);
      TrainMC mcts{&mcts_rng, to_eval.data(), mcts_searches, 16,
                   3.0F,      0.25F,          true};
      GameStats &st = stats[g];
      int32_t last_ab_move = -1;
      for (int32_t ply = 0;; ++ply) {
        MoveMask legal;
        const bool lines = game.getLegalMoves(legal);
        // The side to move at this ply relative to the opening's end
        const bool ab_to_move = (ply % 2) == ab_side;
        if (!legal.any()) {  // the side to move lost (line) or drew
          const float mover = lines ? 0.0F : 0.5F;
          st.ab_score = ab_to_move ? mover : 1.0F - mover;
          break;
        }
        if (game.horizon() <= solve_p) {
          int32_t r = solver.solve(game, 5000000);
          if (r == Solver::kUnknown)
            r = solver.solve(game, 100000000);
          const float mover =
              r == Solver::kUnknown ? 0.5F : (static_cast<float>(r) + 1) / 2;
          st.ab_score = ab_to_move ? mover : 1.0F - mover;
          break;
        }
        int32_t move;
        if (ab_to_move) {
          const auto t = Clock::now();
          const AlphaBeta::Result r = ab.choose(game, ab_seconds);
          st.ab_seconds += since(t);
          ++st.ab_moves;
          st.depth_sum += r.depth;
          st.ab_nodes += r.nodes;
          move = r.move;
          last_ab_move = move;
        } else {
          const auto t = Clock::now();
          bool done;
          if (mcts.uninitialized()) {
            mcts.createRoot(game, depth);
            done = mcts.doIteration();
          } else {
            const bool need =
                mcts.receiveOpponentMove(last_ab_move, game, depth);
            done = need ? false : mcts.doIteration();
          }
          while (!done) {
            const int32_t rows = mcts.num_requests();
            az.evaluate(to_eval.data(), rows, values.data(), probs.data());
            done = mcts.doIteration(values.data(), probs.data());
          }
          move = mcts.chooseMove();
          st.mcts_seconds += since(t);
          ++st.mcts_moves;
        }
        game.doMove(move);
        ++depth;
      }
      std::lock_guard<std::mutex> lock(print_mutex);
      std::fprintf(
          stderr,
          "game %d: ab %.1f (ab %s), %d+%d moves, ab %.2f s/move "
          "depth %.1f, mcts %.2f s/move, %.0f s\n",
          g, st.ab_score, ab_side == 0 ? "first" : "second", st.ab_moves,
          st.mcts_moves, st.ab_seconds / std::max(1, st.ab_moves),
          static_cast<double>(st.depth_sum) / std::max(1, st.ab_moves),
          st.mcts_seconds / std::max(1, st.mcts_moves), since(t0));
    }
  };
#ifdef AB_PROFILE
  std::atomic<uint64_t> prof_total[8]{};
  auto worker_profiled = [&]() {
    worker();
    for (int i = 0; i < 8; ++i)
      prof_total[i] += g_prof[i];
  };
#else
  auto &worker_profiled = worker;
#endif
  std::vector<std::thread> pool;
  for (int32_t i = 0; i < threads; ++i)
    pool.emplace_back(worker_profiled);
  for (std::thread &t : pool)
    t.join();

  double score = 0, ab_s = 0, mcts_s = 0, depth = 0;
  int32_t wins = 0, draws = 0, losses = 0, ab_m = 0, mcts_m = 0;
  for (const GameStats &s : stats) {
    score += s.ab_score;
    wins += s.ab_score == 1.0F;
    draws += s.ab_score == 0.5F;
    losses += s.ab_score == 0.0F;
    ab_s += s.ab_seconds;
    mcts_s += s.mcts_seconds;
    ab_m += s.ab_moves;
    mcts_m += s.mcts_moves;
    depth += s.depth_sum;
  }
  // Pairs: the same opening with colours swapped
  int32_t pairs_ab = 0, pairs_mcts = 0, pairs_split = 0;
  for (int32_t g = 0; g + 1 < games; g += 2) {
    const float s = stats[g].ab_score + stats[g + 1].ab_score;
    pairs_ab += s > 1.0F;
    pairs_mcts += s < 1.0F;
    pairs_split += s == 1.0F;
  }
  std::printf("alpha-beta vs MCTS: %d-%d-%d (score %.3f, decisive %.3f)\n",
              wins, draws, losses, score / games,
              static_cast<double>(wins) / std::max(1, wins + losses));
  std::printf("pairs won by alpha-beta %d, by MCTS %d, split %d\n", pairs_ab,
              pairs_mcts, pairs_split);
  uint64_t nodes = 0;
  for (const GameStats &s : stats)
    nodes += s.ab_nodes;
  std::printf("seconds per move: alpha-beta %.2f (mean depth %.1f, %.0fk "
              "nodes/s), MCTS %.2f (%d searches)\n",
              ab_s / std::max(1, ab_m), depth / std::max(1, ab_m),
              static_cast<double>(nodes) / std::max(1e-9, ab_s) / 1e3,
              mcts_s / std::max(1, mcts_m), mcts_searches);
  std::printf("wall %.0f s\n", since(t0));
#ifdef AB_PROFILE
  const double cycles = static_cast<double>(prof_total[0] + prof_total[1] +
                                            prof_total[2] + prof_total[3]);
  const char *names[4] = {"move generation", "network evaluation",
                          "move ordering", "child copy + doMove"};
  const double all_nodes = static_cast<double>(nodes);
  for (int i = 0; i < 4; ++i)
    std::printf("%-20s %5.1f%% of timed cycles, %6.0f cycles per node\n",
                names[i], 100.0 * prof_total[i] / cycles,
                prof_total[i] / all_nodes);
  std::printf("leaves evaluated %.1f%% of nodes, interior expanded %.1f%%\n",
              100.0 * prof_total[4] / all_nodes,
              100.0 * prof_total[5] / all_nodes);
  std::printf("timed cycles per node %.0f; all AB time per node %.2f us\n",
              cycles / all_nodes, 1e6 * ab_s / all_nodes);
#endif
  return 0;
}
