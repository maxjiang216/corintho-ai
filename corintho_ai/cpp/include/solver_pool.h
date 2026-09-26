#ifndef SOLVER_POOL_H
#define SOLVER_POOL_H

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "game.h"

/// @brief A position waiting for (or given) an exact result
/// @details result is kWaiting until a pool thread stores 1, 0, -1 (for
/// the side to move) or Solver::kUnknown (the node cap was reached).
struct SolveJob {
  static constexpr int8_t kWaiting = 3;
  Game game;
  std::atomic<int8_t> result{kWaiting};
};

/// @brief Threads that solve self-play positions off the engine's critical
/// path (worklog 2026-09-25-nn-architectures, entry 14)
/// @details A game whose position reaches the solve horizon submits it and
/// pauses; self-play iterations never wait for a solve. Each thread keeps
/// one Solver whose table is never cleared: stored results and bounds are
/// facts about positions, true in every game, so the table becomes a
/// growing endgame cache.
class SolverPool {
 public:
  SolverPool(int32_t num_threads, int32_t log2_table, uint64_t max_nodes);
  ~SolverPool();
  SolverPool(const SolverPool &) = delete;
  SolverPool &operator=(const SolverPool &) = delete;

  std::shared_ptr<SolveJob> submit(const Game &game);

  /// @brief Totals so far: solves finished, of which capped; seconds spent
  /// solving (summed over threads); the longest single solve
  uint64_t solves() const noexcept { return solves_.load(); }
  uint64_t capped() const noexcept { return capped_.load(); }
  double seconds() const noexcept;
  double max_seconds() const noexcept;

 private:
  void run(int32_t index);

  int32_t log2_table_;
  uint64_t max_nodes_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<std::shared_ptr<SolveJob>> queue_;
  bool stop_{false};
  std::atomic<uint64_t> solves_{0}, capped_{0}, nanos_{0}, max_nanos_{0};
  std::vector<std::thread> threads_;
};

#endif
