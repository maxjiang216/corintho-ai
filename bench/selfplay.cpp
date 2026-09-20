// End-to-end self-play engine benchmark.
//
// Drives Trainer exactly as corintho_ai/python/main.pyx does, but substitutes a
// deterministic stub for the neural network. That isolates the C++ engine --
// the part PLAN.md sections 7.2 and 13 target -- from TensorFlow entirely, so
// engine work needs no Python environment and no GPU.
//
// Reports the same split as the generation logs in generations/*/play_time.txt:
// self-play time vs "prediction" time, plus derived throughput.

#include <cstdint>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bench_util.h"
#include "trainer.h"
#include "util.h"

namespace {

struct Config {
  int32_t num_games{500};
  int32_t max_searches{1600};
  int32_t searches_per_eval{16};
  float c_puct{3.0F};
  float epsilon{0.25F};
  int32_t num_threads{14};
  int32_t seed{12345};
  std::string log_folder{"/tmp"};
};

void usage() {
  std::printf(
      "usage: selfplay_bench [games] [max_searches] [searches_per_eval] "
      "[threads] [seed]\n");
}

}  // namespace

int main(int argc, char **argv) {
  Config cfg;
  if (argc > 1 && std::string(argv[1]) == "-h") {
    usage();
    return 0;
  }
  if (argc > 1)
    cfg.num_games = std::atoi(argv[1]);
  if (argc > 2)
    cfg.max_searches = std::atoi(argv[2]);
  if (argc > 3)
    cfg.searches_per_eval = std::atoi(argv[3]);
  if (argc > 4)
    cfg.num_threads = std::atoi(argv[4]);
  if (argc > 5)
    cfg.seed = std::atoi(argv[5]);

  const size_t batch =
      static_cast<size_t>(cfg.num_games) * cfg.searches_per_eval;

  std::printf("Corintho self-play engine benchmark (stub evaluator)\n");
  std::printf(
      "games %d  max_searches %d  searches_per_eval %d  threads %d  seed %d\n",
      cfg.num_games, cfg.max_searches, cfg.searches_per_eval, cfg.num_threads,
      cfg.seed);
  // The to_eval_ buffer is allocated per game at kGameStateSize * max_searches
  // floats, which is the oversizing described in PLAN.md section 10.2.
  const double to_eval_mb = static_cast<double>(cfg.num_games) *
                            kGameStateSize * cfg.max_searches * sizeof(float) /
                            (1024.0 * 1024.0);
  std::printf("to_eval_ buffers: %.1f MB (only %.1f MB is ever used)\n\n",
              to_eval_mb,
              to_eval_mb * cfg.searches_per_eval / cfg.max_searches);

  std::vector<float> evals(batch, 0.0F);
  std::vector<float> probs(batch * kNumMoves, 0.0F);
  std::vector<float> game_states(batch * kGameStateSize, 0.0F);

  auto setup_start = bench::Clock::now();
  Trainer trainer{cfg.num_games,       cfg.log_folder,    cfg.seed,
                  cfg.max_searches,    cfg.searches_per_eval, cfg.c_puct,
                  cfg.epsilon,         0 /* num_logged */, cfg.num_threads,
                  false /* testing */};
  const double setup_time = bench::secondsSince(setup_start);

  double play_time = 0.0;
  double eval_time = 0.0;
  uint64_t rounds = 0;
  uint64_t total_requests = 0;

  auto wall_start = bench::Clock::now();
  while (true) {
    auto play_start = bench::Clock::now();
    const bool done = trainer.doIteration(evals.data(), probs.data(), -1);
    play_time += bench::secondsSince(play_start);
    if (done)
      break;

    const int32_t num_requests = trainer.num_requests(-1);
    if (num_requests == 0) {
      std::printf("ERROR: no requests during training (see main.pyx)\n");
      return 1;
    }
    trainer.writeRequests(game_states.data(), -1);

    auto eval_start = bench::Clock::now();
    bench::stubEvaluate(game_states.data(), num_requests, evals.data(),
                        probs.data());
    eval_time += bench::secondsSince(eval_start);

    ++rounds;
    total_requests += static_cast<uint64_t>(num_requests);
  }
  const double wall = bench::secondsSince(wall_start);
  const int32_t num_samples = trainer.num_samples();

  std::printf("setup                %8.3f s\n", setup_time);
  std::printf("self-play (engine)   %8.3f s   %5.1f%%\n", play_time,
              100.0 * play_time / wall);
  std::printf("stub evaluation      %8.3f s   %5.1f%%\n", eval_time,
              100.0 * eval_time / wall);
  std::printf("wall                 %8.3f s\n\n", wall);

  std::printf("evaluation rounds    %8llu\n",
              static_cast<unsigned long long>(rounds));
  std::printf("total NN requests    %8llu\n",
              static_cast<unsigned long long>(total_requests));
  std::printf("mean batch size      %8.1f\n",
              static_cast<double>(total_requests) / rounds);
  std::printf("turns (samples)      %8d\n", num_samples);
  std::printf("turns per game       %8.2f\n",
              static_cast<double>(num_samples) / cfg.num_games);
  std::printf("engine s / turn      %8.6f\n", play_time / num_samples);
  std::printf("requests / engine s  %8.0f\n", total_requests / play_time);

  // Consumed by run_suite.sh; see README.md.
  std::printf("#METRIC engine_seconds %.4f\n", play_time);
  std::printf("#METRIC wall_seconds %.4f\n", wall);
  std::printf("#METRIC turns %d\n", num_samples);
  std::printf("#METRIC requests %llu\n",
              static_cast<unsigned long long>(total_requests));
  std::printf("#METRIC requests_per_engine_second %.0f\n",
              total_requests / play_time);
  return 0;
}
