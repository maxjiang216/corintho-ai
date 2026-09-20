#ifndef BENCH_UTIL_H
#define BENCH_UTIL_H

// Shared helpers for the Corintho benchmark harness.
// Deliberately dependency-free: no Python, no TensorFlow, no gtest.

#include <cstdint>

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "util.h"

namespace bench {

using Clock = std::chrono::steady_clock;

inline double secondsSince(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

/// @brief Keeps the optimizer from deleting work whose result is unused.
template <typename T> inline void keep(T const &value) {
  asm volatile("" : : "r,m"(value) : "memory");
}

/// @brief SplitMix64. Used for both the stub evaluator and corpus generation.
/// @details Chosen so that every benchmark is reproducible from a seed and
/// contains no real RNG cost worth measuring.
inline uint64_t splitmix64(uint64_t &state) {
  uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

/// @brief Stand-in for the neural network during engine benchmarks.
/// @details The real pipeline calls Keras here. For engine work we only need
/// something deterministic, cheap, and varied enough that the search tree does
/// not degenerate into a uniform fan-out. Evaluations are derived by hashing
/// the game state, so identical positions always receive identical values --
/// which also makes this harness usable for measuring position-sharing.
inline void stubEvaluate(const float *game_states, int32_t num_requests,
                         float *evals, float *probs) {
  for (int32_t i = 0; i < num_requests; ++i) {
    const float *state = game_states + static_cast<size_t>(i) * kGameStateSize;
    // Hash the position.
    uint64_t h = 0xCBF29CE484222325ULL;
    for (int32_t j = 0; j < kGameStateSize; ++j) {
      uint32_t bits = 0;
      __builtin_memcpy(&bits, &state[j], sizeof(bits));
      h = (h ^ bits) * 0x100000001B3ULL;
    }
    uint64_t rng = h;
    // Evaluation in (-1, 1).
    evals[i] = static_cast<float>(
                   static_cast<int64_t>(splitmix64(rng) % 2000001) - 1000000) /
               1000000.0F;
    // Strictly positive priors; the engine normalizes over legal moves.
    float *row = probs + static_cast<size_t>(i) * kNumMoves;
    for (int32_t j = 0; j < kNumMoves; ++j) {
      row[j] = static_cast<float>(splitmix64(rng) % 1000 + 1) / 1000.0F;
    }
  }
}

/// @brief One timed measurement, reported as a single line.
struct Result {
  std::string name;
  double seconds{0.0};
  uint64_t ops{0};

  void report() const {
    const double ns = seconds * 1e9 / static_cast<double>(ops);
    std::printf("  %-34s %10.1f ns/op  %12llu ops  %8.3f s\n", name.c_str(), ns,
                static_cast<unsigned long long>(ops), seconds);
  }
};

}  // namespace bench

#endif
