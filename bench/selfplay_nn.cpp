// Self-play engine benchmark with a REAL network instead of the stub.
//
//   selfplay_nn <python> <model.tflite> [games] [max_searches]
//               [searches_per_eval] [threads] [seed]
//   selfplay_nn - <model.mlp> [games] ...
//
// Identical to selfplay_bench except for the evaluator. With a python
// interpreter, game states go over a pipe to nn_server.py, which runs the
// tflite model and sends back values and move probabilities. With "-" in its
// place, the in-process Mlp (mlp.h) evaluates a .mlp file written by
// export_mlp.py, on the same number of threads as the search.
//
// The stub's hash-based priors are flat and random; a trained network's are
// peaked, which changes the tree shape (how many children nodes get, how deep
// the search goes). Use this to check that an engine change measured on the
// stub holds with realistic priors.
//
// MLP_CHECK=<model.mlp> in the environment (python mode only) also evaluates
// every batch with Mlp and reports the largest differences from tflite. The
// games are played from the tflite outputs, so they are the same games as
// without the check.
//
// gather_seconds is the serial step between the engine and the network. It
// used to count requests and copy every game's rows into a packed batch; the
// games now write into fixed slots of the trainer's batch (Trainer::requests),
// so it is only the count. Neither engine_seconds nor the network time
// includes it. "requests" counts evaluated rows, including the few unused rows
// in partly filled slots and the slots of finished games.
//
// engine_seconds excludes the network, so engine timings are comparable with
// selfplay_bench's. Wall time is dominated by inference and is not.

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "bench_util.h"
#include "mlp.h"
#include "trainer.h"
#include "util.h"

namespace {

void writeAll(int fd, const void *data, size_t size) {
  const char *p = static_cast<const char *>(data);
  while (size > 0) {
    const ssize_t n = write(fd, p, size);
    if (n <= 0) {
      std::perror("write to nn_server");
      std::exit(1);
    }
    p += n;
    size -= static_cast<size_t>(n);
  }
}

void readAll(int fd, void *data, size_t size) {
  char *p = static_cast<char *>(data);
  while (size > 0) {
    const ssize_t n = read(fd, p, size);
    if (n <= 0) {
      std::fprintf(stderr, "nn_server closed the pipe\n");
      std::exit(1);
    }
    p += n;
    size -= static_cast<size_t>(n);
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::printf(
        "usage: selfplay_nn <python> <model.tflite> [games] [max_searches] "
        "[searches_per_eval] [threads] [seed]\n"
        "       selfplay_nn - <model.mlp> [games] ...\n");
    return 1;
  }
  const std::string python = argv[1];
  const std::string model = argv[2];
  int32_t num_games = argc > 3 ? std::atoi(argv[3]) : 20;
  int32_t max_searches = argc > 4 ? std::atoi(argv[4]) : 1600;
  int32_t searches_per_eval = argc > 5 ? std::atoi(argv[5]) : 16;
  int32_t num_threads = argc > 6 ? std::atoi(argv[6]) : 1;
  int32_t seed = argc > 7 ? std::atoi(argv[7]) : 12345;
  // Same c_puct and epsilon as selfplay_bench
  const float c_puct = 3.0F;
  const float epsilon = 0.25F;

  const bool in_process = python == "-";
  std::unique_ptr<Mlp> mlp;
  std::unique_ptr<Mlp> check;
  if (in_process) {
    mlp = std::make_unique<Mlp>(model);
  } else if (const char *path = std::getenv("MLP_CHECK")) {
    check = std::make_unique<Mlp>(path);
  }

  // Start nn_server.py, next to this binary's source
  int to_child[2] = {-1, -1}, from_child[2] = {-1, -1};
  pid_t pid = -1;
  if (!in_process) {
    if (pipe(to_child) != 0 || pipe(from_child) != 0) {
      std::perror("pipe");
      return 1;
    }
    const std::string server = std::string(BENCH_DIR) + "/nn_server.py";
    pid = fork();
    if (pid == 0) {
      dup2(to_child[0], 0);
      dup2(from_child[1], 1);
      close(to_child[1]);
      close(from_child[0]);
      execl(python.c_str(), python.c_str(), server.c_str(), model.c_str(),
            static_cast<char *>(nullptr));
      std::perror("exec nn_server");
      _exit(1);
    }
    close(to_child[0]);
    close(from_child[1]);
  }

  std::printf("Corintho self-play engine benchmark (real network: %s)\n",
              model.c_str());
  std::printf(
      "games %d  max_searches %d  searches_per_eval %d  threads %d  seed %d\n",
      num_games, max_searches, searches_per_eval, num_threads, seed);

  const size_t batch = static_cast<size_t>(num_games) * searches_per_eval;
  std::vector<float> evals(batch, 0.0F);
  std::vector<float> probs(batch * kNumMoves, 0.0F);
  std::vector<float> check_evals, check_probs;
  if (check) {
    check_evals.resize(batch);
    check_probs.resize(batch * kNumMoves);
  }
  float max_eval_diff = 0.0F;
  float max_prob_diff = 0.0F;
  uint64_t argmax_mismatches = 0;

  Trainer trainer{num_games,         "/tmp", seed,    max_searches,
                  searches_per_eval, c_puct, epsilon, 0 /* num_logged */,
                  num_threads,       false /* testing */};

  double play_time = 0.0;
  double eval_time = 0.0;
  double gather_time = 0.0;
  uint64_t total_requests = 0;
  auto wall_start = bench::Clock::now();
  while (true) {
    auto play_start = bench::Clock::now();
    const bool done = trainer.doIteration(evals.data(), probs.data(), -1);
    play_time += bench::secondsSince(play_start);
    if (done)
      break;
    auto gather_start = bench::Clock::now();
    const int32_t n = trainer.num_requests(-1);
    // The games wrote their rows into their slots of the trainer's batch
    const float *batch = trainer.requests();
    gather_time += bench::secondsSince(gather_start);

    auto eval_start = bench::Clock::now();
    if (in_process) {
      mlp->evaluateParallel(batch, n, evals.data(), probs.data(), num_threads);
    } else {
      writeAll(to_child[1], &n, sizeof(n));
      writeAll(to_child[1], batch,
               static_cast<size_t>(n) * kGameStateSize * sizeof(float));
      readAll(from_child[0], evals.data(),
              static_cast<size_t>(n) * sizeof(float));
      readAll(from_child[0], probs.data(),
              static_cast<size_t>(n) * kNumMoves * sizeof(float));
    }
    eval_time += bench::secondsSince(eval_start);
    if (check) {
      check->evaluateParallel(batch, n, check_evals.data(), check_probs.data(),
                              num_threads);
      for (int32_t i = 0; i < n; ++i) {
        max_eval_diff =
            std::max(max_eval_diff, std::fabs(evals[i] - check_evals[i]));
        const float *a = probs.data() + static_cast<size_t>(i) * kNumMoves;
        const float *b =
            check_probs.data() + static_cast<size_t>(i) * kNumMoves;
        int32_t best_a = 0, best_b = 0;
        for (int32_t m = 0; m < kNumMoves; ++m) {
          max_prob_diff = std::max(max_prob_diff, std::fabs(a[m] - b[m]));
          best_a = a[m] > a[best_a] ? m : best_a;
          best_b = b[m] > b[best_b] ? m : best_b;
        }
        argmax_mismatches += best_a != best_b;
      }
    }
    total_requests += static_cast<uint64_t>(n);
  }
  const double wall = bench::secondsSince(wall_start);
  if (!in_process) {
    close(to_child[1]);
    close(from_child[0]);
    waitpid(pid, nullptr, 0);
  }

  const int32_t turns = trainer.num_samples();
  std::printf("self-play (engine)   %8.3f s\n", play_time);
  std::printf("gather requests      %8.3f s\n", gather_time);
  std::printf("network evaluation   %8.3f s\n", eval_time);
  std::printf("turns per game       %8.2f\n",
              static_cast<double>(turns) / num_games);
  if (check) {
    std::printf("mlp check: max |eval diff| %.3g, max |prob diff| %.3g, "
                "policy argmax differs on %llu of %llu rows\n",
                max_eval_diff, max_prob_diff,
                static_cast<unsigned long long>(argmax_mismatches),
                static_cast<unsigned long long>(total_requests));
    std::printf("#METRIC mlp_max_eval_diff %.6g\n", max_eval_diff);
    std::printf("#METRIC mlp_max_prob_diff %.6g\n", max_prob_diff);
  }
  std::printf("#METRIC engine_seconds %.4f\n", play_time);
  std::printf("#METRIC gather_seconds %.4f\n", gather_time);
  std::printf("#METRIC wall_seconds %.4f\n", wall);
  std::printf("#METRIC turns %d\n", turns);
  std::printf("#METRIC requests %llu\n",
              static_cast<unsigned long long>(total_requests));
  return 0;
}
