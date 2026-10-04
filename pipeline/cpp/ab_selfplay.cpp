// Alpha-beta self-play: labelled positions for training the small value
// network (worklog 2026-09-25-nn-architectures, entry 25).
//
//   ab_selfplay --small NET.bin --out GEN.npy [--positions 200000]
//               [--threads 18] [--depth 5] [--opening 4] [--temp 0.05]
//               [--temp-plies 10] [--solve-p 27] [--min-p 20] [--seed 1]
//               [--report-seconds 30]
//
// Each game starts with --opening random legal moves (openings deduplicated
// up to the board symmetries), then both sides play a fixed-depth alpha-beta
// search with the small network. For the first --temp-plies moves after the
// opening the move is sampled: probabilities exp((s - best) / T) over the
// moves within 4T of the best, T falling linearly from --temp to 0. From
// P <= --solve-p every position is also solved exactly.
//
// Output: float32 .npy, one row per position with P >= --min-p:
//   0-69  network inputs x 4 (integers 0-4), the side to move first
//   70    search value: the root's fixed-depth score as a value (tanh of the
//         network's pre-tanh output; +-1 for a forced result)
//   71    outcome for the side to move: the exact result of the game's first
//         solved position if any, else the result as played
//   72    exact value (-1, 0, 1) if P <= --solve-p and solved, else NaN
//   73    horizon P
//   74    game number
//   75    ply

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "alphabeta.h"
#include "game.h"
#include "move.h"
#include "npy.h"
#include "small_net.h"
#include "solver.h"
#include "util.h"

namespace {

using ab::AlphaBeta;
using ab::Clock;
using ab::since;

constexpr int32_t kCols = 76;
constexpr int32_t kMaxPlies = 400;

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

/// The network inputs x 4 (integers)
void inputs(const Game &game, uint8_t out[kGameStateSize]) {
  float x[kGameStateSize];
  game.writeGameState(x);
  for (int32_t i = 0; i < kGameStateSize; ++i)
    out[i] = static_cast<uint8_t>(x[i] * 4.0F + 0.5F);
}

/// The position up to the 8 board symmetries: the smallest of its copies
std::string canonical(const uint8_t state[kGameStateSize]) {
  std::string best;
  for (int32_t k = 0; k < kNumSymmetries; ++k) {
    std::string s(kGameStateSize, '\0');
    for (int32_t j = 0; j < kGameStateSize; ++j) {
      const int32_t src =
          j < 64 ? space_symmetries[k][j / 4] * 4 + j % 4 : j;
      s[j] = static_cast<char>(state[src]);
    }
    if (k == 0 || s < best)
      best = s;
  }
  return best;
}

/// A search score as a value: the network's pre-tanh output is 8x the
/// score (SmallNet::kScoreScale); beyond +-1.5 the result is forced
float searchValue(float score) {
  if (std::abs(score) > 1.5F)
    return score > 0 ? 1.0F : -1.0F;
  return std::tanh(score / SmallNet::kScoreScale);
}

struct Position {
  uint8_t state[kGameStateSize];
  float search, exact;
  int32_t horizon, ply;
};

}  // namespace

int main(int argc, char **argv) {
  const Args a(argc, argv);
  const SmallNet net{a.str("small")};
  const std::string out_path = a.str("out");
  if (out_path.empty()) {
    std::fprintf(stderr, "--out is required\n");
    return 1;
  }
  const int64_t target = static_cast<int64_t>(a.num("positions", 200000));
  const int32_t threads = static_cast<int32_t>(a.num("threads", 18));
  const int32_t depth = static_cast<int32_t>(a.num("depth", 5));
  // A progress line every this many seconds (stderr)
  const double report_seconds = a.num("report-seconds", 30);
  const int32_t opening = static_cast<int32_t>(a.num("opening", 4));
  const float temp = static_cast<float>(a.num("temp", 0.05));
  const int32_t temp_plies = static_cast<int32_t>(a.num("temp-plies", 10));
  const int32_t solve_p = static_cast<int32_t>(a.num("solve-p", 27));
  const int32_t min_p = static_cast<int32_t>(a.num("min-p", 20));
  const uint32_t seed = static_cast<uint32_t>(a.num("seed", 1));

  NpyWriter writer{out_path, kCols};
  std::mutex mutex;  // the writer, the opening set and the counters
  std::set<std::string> openings;
  std::atomic<int64_t> written{0};
  std::atomic<int32_t> next_game{0};
  int64_t games = 0, plies = 0, solved_games = 0, duplicate_openings = 0;
  int64_t results[3] = {0, 0, 0};  // first player: loss, draw, win
  uint64_t nodes = 0;
  double search_seconds = 0;
  const auto t0 = Clock::now();
  double last_report = 0;

  auto worker = [&]() {
    AlphaBeta search{net, 20};
    Solver solver{20};
    std::vector<Position> positions;
    std::vector<float> rows;
    for (int32_t g; written.load() < target && (g = next_game++) >= 0;) {
      std::mt19937 rng{seed * 1000003U + static_cast<uint32_t>(g)};
      // Opening: random moves, a new position up to symmetry (a few tries)
      Game game;
      for (int32_t attempt = 0;; ++attempt) {
        game = Game{};
        bool ended = false;
        for (int32_t i = 0; i < opening && !ended; ++i) {
          MoveMask legal;
          game.getLegalMoves(legal);
          std::vector<int32_t> moves;
          forEachMove(legal, [&](int32_t m) { moves.push_back(m); });
          if (moves.empty()) {
            ended = true;
            break;
          }
          game.doMove(moves[rng() % moves.size()]);
        }
        uint8_t state[kGameStateSize];
        inputs(game, state);
        std::lock_guard<std::mutex> lock(mutex);
        if (openings.insert(canonical(state)).second || attempt >= 20)
          break;
        ++duplicate_openings;
      }
      positions.clear();
      int32_t ply = opening, first_solved = -1, first_value = 0;
      int32_t end_value = 0;  // for the side to move at ply `ply` when over
      uint64_t game_nodes = 0;
      double game_seconds = 0;
      for (;; ++ply) {
        MoveMask legal;
        const bool lines = game.getLegalMoves(legal);
        if (!legal.any() || ply >= opening + kMaxPlies) {
          end_value = legal.any() ? 0 : (lines ? -1 : 0);
          break;
        }
        Position pos;
        inputs(game, pos.state);
        pos.horizon = game.horizon();
        pos.ply = ply;
        pos.exact = NAN;
        if (pos.horizon <= solve_p) {
          const int32_t r = solver.solve(game, 5000000);
          if (r != Solver::kUnknown) {
            pos.exact = static_cast<float>(r);
            if (first_solved < 0) {
              first_solved = ply;
              first_value = r;
            }
          }
        }
        const auto t = Clock::now();
        AlphaBeta::Result best;
        int32_t move;
        const int32_t t_ply = ply - opening;
        if (t_ply < temp_plies && temp > 0) {
          const float T = temp * (1.0F - static_cast<float>(t_ply) /
                                             static_cast<float>(temp_plies));
          const auto scores = search.rootScores(game, depth, 4.0F * T, best);
          std::vector<double> w;
          for (const auto &ms : scores)
            w.push_back(ms.second <= -AlphaBeta::kInf / 2
                            ? 0.0
                            : std::exp((ms.second - best.score) / T));
          std::discrete_distribution<size_t> pick(w.begin(), w.end());
          move = scores[pick(rng)].first;
        } else {
          best = search.choose(game, 1e30, depth);
          move = best.move;
        }
        game_seconds += since(t);
        game_nodes += best.nodes;
        pos.search = searchValue(best.score);
        if (pos.horizon >= min_p)
          positions.push_back(pos);
        game.doMove(move);
      }
      // The outcome for each position's side to move: from the first solved
      // position if any (exact even if the game was misplayed after it),
      // else the result as played; the sign flips every ply
      const int32_t ref_ply = first_solved >= 0 ? first_solved : ply;
      const int32_t ref_value = first_solved >= 0 ? first_value : end_value;
      rows.clear();
      for (const Position &p : positions) {
        const float outcome = static_cast<float>(
            (p.ply - ref_ply) % 2 == 0 ? ref_value : -ref_value);
        for (int32_t i = 0; i < kGameStateSize; ++i)
          rows.push_back(static_cast<float>(p.state[i]));
        rows.push_back(p.search);
        rows.push_back(outcome);
        rows.push_back(p.exact);
        rows.push_back(static_cast<float>(p.horizon));
        rows.push_back(static_cast<float>(g));
        rows.push_back(static_cast<float>(p.ply));
      }
      // The first player's result (ply 0's side to move)
      const int32_t first =
          (0 - ref_ply) % 2 == 0 ? ref_value : -ref_value;
      std::lock_guard<std::mutex> lock(mutex);
      writer.append(rows.data(), static_cast<int64_t>(positions.size()));
      written += static_cast<int64_t>(positions.size());
      ++games;
      plies += ply;
      solved_games += first_solved >= 0;
      ++results[first + 1];
      nodes += game_nodes;
      search_seconds += game_seconds;
      const double now = since(t0);
      if (now - last_report >= report_seconds) {
        last_report = now;
        const double done = static_cast<double>(written.load());
        const double rate = done / std::max(1e-9, now);
        std::fprintf(stderr,
                     "%.0f s: %lld/%lld positions (%.0f/s, about %.0f s "
                     "left), %lld games, first player %lld-%lld-%lld\n",
                     now, static_cast<long long>(written.load()),
                     static_cast<long long>(target), rate,
                     std::max(0.0, static_cast<double>(target) - done) /
                         std::max(1e-9, rate),
                     static_cast<long long>(games),
                     static_cast<long long>(results[2]),
                     static_cast<long long>(results[1]),
                     static_cast<long long>(results[0]));
      }
    }
  };
  std::vector<std::thread> pool;
  for (int32_t i = 0; i < threads; ++i)
    pool.emplace_back(worker);
  for (std::thread &t : pool)
    t.join();
  writer.close();
  std::printf(
      "%s: %lld positions from %lld games (%.1f plies each, %.1f%% reached "
      "a solved position), first player %lld-%lld-%lld (W-D-L), %lld "
      "duplicate openings redrawn\n",
      out_path.c_str(), static_cast<long long>(writer.rows()),
      static_cast<long long>(games),
      static_cast<double>(plies) / std::max<int64_t>(1, games),
      100.0 * static_cast<double>(solved_games) / std::max<int64_t>(1, games),
      static_cast<long long>(results[2]), static_cast<long long>(results[1]),
      static_cast<long long>(results[0]),
      static_cast<long long>(duplicate_openings));
  std::printf("depth %d, %.0fk nodes/s per thread, wall %.0f s\n", depth,
              static_cast<double>(nodes) / std::max(1e-9, search_seconds) /
                  1e3,
              since(t0));
  return 0;
}
