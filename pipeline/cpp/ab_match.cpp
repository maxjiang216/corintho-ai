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
#include "alphabeta.h"
#include "small_net.h"
#include "solver.h"
#include "trainmc.h"
#include "util.h"

namespace {

using ab::AlphaBeta;
using ab::Clock;
using ab::since;

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
  std::printf("  within ordering: replies of line-making moves %.0f, sort "
              "%.0f cycles per node\n",
              prof_total[6] / all_nodes, prof_total[7] / all_nodes);
  std::printf("leaves evaluated %.1f%% of nodes, interior expanded %.1f%%\n",
              100.0 * prof_total[4] / all_nodes,
              100.0 * prof_total[5] / all_nodes);
  std::printf("timed cycles per node %.0f; all AB time per node %.2f us\n",
              cycles / all_nodes, 1e6 * ab_s / all_nodes);
#endif
  return 0;
}
