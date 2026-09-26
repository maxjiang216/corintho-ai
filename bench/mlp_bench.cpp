// Throughput of the in-process CPU network (Mlp) at a given batch size.
//
//   mlp_bench <model.mlp> [rows] [threads] [reps]
//
// Inputs are 0/1 at random, like real game states (the values do not change
// the work: the forward pass is dense, with no data-dependent branches). The
// same batch is evaluated reps times after one warm-up pass.
//
// FLOP counts multiply-adds as two, over every layer including both heads.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "bench_util.h"
#include "mlp.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    std::printf("usage: mlp_bench <model.mlp> [rows] [threads] [reps]\n");
    return 1;
  }
  const Mlp mlp{argv[1]};
  const int32_t rows = argc > 2 ? std::atoi(argv[2]) : 65536;
  const int32_t threads = argc > 3 ? std::atoi(argv[3]) : 1;
  const int32_t reps = argc > 4 ? std::atoi(argv[4]) : 10;

  std::vector<float> states(static_cast<size_t>(rows) * mlp.inputSize());
  uint64_t rng = 12345;
  for (float &x : states)
    x = static_cast<float>(bench::splitmix64(rng) & 1);
  std::vector<float> evals(rows);
  std::vector<float> probs(static_cast<size_t>(rows) * mlp.numMoves());

  mlp.evaluateParallel(states.data(), rows, evals.data(), probs.data(),
                       threads);
  const auto start = bench::Clock::now();
  for (int32_t i = 0; i < reps; ++i)
    mlp.evaluateParallel(states.data(), rows, evals.data(), probs.data(),
                         threads);
  const double seconds = bench::secondsSince(start);
  bench::keep(evals[0]);

  // 70x100 + 11 x 100x100 + 100x1 + 100x96 for model_93
  const double macs_per_row = 70.0 * 100 + 11 * 100.0 * 100 + 100 + 100 * 96;
  const double total_rows = static_cast<double>(rows) * reps;
  const double ns_per_row = seconds * 1e9 / total_rows;
  std::printf("rows %d  threads %d  reps %d\n", rows, threads, reps);
  std::printf("ns per row      %10.1f\n", ns_per_row);
  std::printf("GFLOP/s         %10.1f\n",
              2.0 * macs_per_row * total_rows / seconds / 1e9);
  std::printf("#METRIC ns_per_row %.3f\n", ns_per_row);
  return 0;
}
