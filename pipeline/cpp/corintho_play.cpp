// Self-play and testing driver for the GPU pipeline (worklog entries 21, 29).
//
//   corintho_play train --model M --games N --out DIR [options]
//   corintho_play test  --new M1 --best M2 --games N --out DIR [options]
//   corintho_play bench --model M [--rows 1000,4000,16000] [--reps 50]
//
// M is a .onnx file (ONNX Runtime, CUDA) or a .mlp file (the CPU network).
//
// train plays N games, at most --in-flight at a time, and writes the samples
// as DIR/states.npy (rows x 70), DIR/values.npy (rows) and DIR/policies.npy
// (rows x 96), with rows = turns x 8 symmetries, plus DIR/selfplay.json.
// Games are played in chunks of --in-flight: all games of a chunk start
// together (with the Trainer's staggered start) and the next chunk starts
// when the last game of this one ends.
//
// test plays N games between two models, alternating colours, and writes
// DIR/test.json with the new model's wins, draws and losses. The promotion
// decision is made in Python (corintho_ai/python/promotion.py).
//
// Options (defaults as in gen_93's metadata):
//   --searches 1600  --spe 16  --c-puct 3.0  --epsilon 0.25
//   --threads 20  --seed 1  --in-flight 1000  --logged 0
//   --groups 1      train: sets of --in-flight games whose searches alternate,
//                   so that with 2 the engine searches one group while the
//                   GPU evaluates the other
//   --stagger 0     train: iterations over which a chunk's games start (0:
//                   the Trainer's original ~16-turn rule; 100 = one turn)
//   --digest        also print the FNV-1a sample digest (as selfplay_nn)
//   --check M2      train only: also evaluate every batch with M2 and report
//                   the largest differences (games follow --model)

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend.h"
#include "npy.h"
#include "trainer.h"
#include "util.h"

namespace {

using Clock = std::chrono::steady_clock;
// Resident memory of this process, in MB (from /proc/self/statm)
double rssMb() {
  std::FILE *f = std::fopen("/proc/self/statm", "r");
  if (f == nullptr)
    return 0;
  long pages = 0, resident = 0;
  if (std::fscanf(f, "%ld %ld", &pages, &resident) != 2)
    resident = 0;
  std::fclose(f);
  return static_cast<double>(resident) * 4096.0 / 1e6;
}

double since(Clock::time_point t) {
  return std::chrono::duration<double>(Clock::now() - t).count();
}

struct Args {
  std::string mode;
  std::map<std::string, std::string> kv;
  bool digest{false};

  std::string str(const std::string &k, const std::string &def = "") const {
    auto it = kv.find(k);
    if (it != kv.end())
      return it->second;
    if (def.empty())
      throw std::runtime_error("missing --" + k);
    return def;
  }
  int32_t i32(const std::string &k, int32_t def) const {
    auto it = kv.find(k);
    return it == kv.end() ? def : std::atoi(it->second.c_str());
  }
  float f32(const std::string &k, float def) const {
    auto it = kv.find(k);
    return it == kv.end() ? def : std::strtof(it->second.c_str(), nullptr);
  }
};

Args parse(int argc, char **argv) {
  if (argc < 2)
    throw std::runtime_error("usage: corintho_play train|test --key value ...");
  Args a;
  a.mode = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string k = argv[i];
    if (k.rfind("--", 0) != 0)
      throw std::runtime_error("unexpected argument " + k);
    k = k.substr(2);
    if (k == "digest") {
      a.digest = true;
      continue;
    }
    if (i + 1 >= argc)
      throw std::runtime_error("missing value for --" + k);
    a.kv[k] = argv[++i];
  }
  return a;
}

struct Fnv {
  uint64_t hash{0xCBF29CE484222325ULL};
  void add(const void *data, size_t bytes) {
    const auto *p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < bytes; ++i)
      hash = (hash ^ p[i]) * 0x100000001B3ULL;
  }
};

// One set of games in flight: its own Trainer and network buffers, and the
// network call it is waiting for, if any
struct Group {
  std::unique_ptr<Trainer> trainer;
  // Each group has its own backend: a backend owns staging buffers, and two
  // groups' network calls can run at the same time
  std::unique_ptr<Backend> backend;
  std::vector<float> values, probs;
  std::future<double> pending;  // seconds the network call took
  int32_t games{0};
};

int runTrain(const Args &a) {
  const std::string model = a.str("model");
  const std::string out = a.str("out");
  const int32_t games = a.i32("games", 25000);
  const int32_t in_flight = std::max(1, a.i32("in-flight", 1000));
  const int32_t num_groups = std::max(1, a.i32("groups", 1));
  const int32_t searches = a.i32("searches", 1600);
  const int32_t spe = a.i32("spe", 16);
  const float c_puct = a.f32("c-puct", 3.0F);
  const float epsilon = a.f32("epsilon", 0.25F);
  const int32_t threads = a.i32("threads", 20);
  const int32_t seed = a.i32("seed", 1);
  const int32_t logged = a.i32("logged", 0);
  // Iterations over which a chunk's games start; 0 keeps the Trainer's
  // original rule (~16 turns). One turn, max_searches / spe iterations,
  // spreads the games over a turn's phases without leaving the batch mostly
  // empty while they ramp up (entry 29).
  const int32_t stagger = a.i32("stagger", 0);

  auto backend = makeBackend(model, threads);
  std::unique_ptr<Backend> check;
  if (a.kv.count("check")) {
    if (num_groups != 1)
      throw std::runtime_error("--check needs --groups 1");
    check = makeBackend(a.str("check"), threads);
  }
  std::vector<float> check_values, check_probs;
  float max_value_diff = 0, max_prob_diff = 0;
  uint64_t argmax_diff = 0;
  NpyWriter states_out{out + "/states.npy", kGameStateSize};
  NpyWriter values_out{out + "/values.npy", 1};
  NpyWriter policies_out{out + "/policies.npy", kNumMoves};

  double engine_s = 0, eval_s = 0, wait_s = 0, write_s = 0;
  uint64_t rows_evaluated = 0, calls = 0;
  int64_t turns_total = 0;
  double score_sum = 0;
  Fnv digest;
  const auto wall_start = Clock::now();
  std::vector<float> st, vs, ps;
  int32_t chunks_started = 0, chunks_done = 0, games_started = 0;

  // Chunk c gets seed + c, and chunks are numbered in the order they start,
  // so a run is deterministic for a given --groups and --in-flight
  auto startChunk = [&](Group &g) {
    if (games_started >= games) {
      g.trainer.reset();
      return;
    }
    g.games = std::min(in_flight, games - games_started);
    // Only the first chunk logs games
    g.trainer = std::make_unique<Trainer>(
        g.games, out, seed + chunks_started, searches, spe, c_puct, epsilon,
        chunks_started == 0 ? logged : 0, threads, false);
    g.trainer->set_stagger_iterations(stagger);
    g.values.assign(static_cast<size_t>(g.games) * spe, 0.0F);
    g.probs.assign(static_cast<size_t>(g.games) * spe * kNumMoves, 0.0F);
    games_started += g.games;
    ++chunks_started;
  };
  auto finishChunk = [&](Group &g) {
    const auto t = Clock::now();
    Trainer &trainer = *g.trainer;
    const int32_t turns = trainer.num_samples();
    const size_t sample_rows = static_cast<size_t>(turns) * kNumSymmetries;
    st.resize(sample_rows * kGameStateSize);
    vs.resize(sample_rows);
    ps.resize(sample_rows * kNumMoves);
    trainer.writeSamples(st.data(), vs.data(), ps.data());
    states_out.append(st.data(), static_cast<int64_t>(sample_rows));
    values_out.append(vs.data(), static_cast<int64_t>(sample_rows));
    policies_out.append(ps.data(), static_cast<int64_t>(sample_rows));
    const float score = trainer.score();
    if (a.digest) {
      // Same fields and order as selfplay_nn's SAMPLE_DIGEST, chunk by chunk
      // in finishing order, so a one-chunk run reproduces its digest exactly
      digest.add(&turns, sizeof(turns));
      digest.add(st.data(), st.size() * sizeof(float));
      digest.add(vs.data(), vs.size() * sizeof(float));
      digest.add(ps.data(), ps.size() * sizeof(float));
      digest.add(&score, sizeof(score));
    }
    write_s += since(t);
    turns_total += turns;
    score_sum += static_cast<double>(score) * g.games;
    ++chunks_done;
    std::fprintf(stderr,
                 "chunk %d done: %d games, %d turns; engine %.1f s, network "
                 "%.1f s (waited %.1f s), wall %.1f s, rss %.0f MB\n",
                 chunks_done - 1, g.games, turns, engine_s, eval_s, wait_s,
                 since(wall_start), rssMb());
  };

  std::vector<Group> groups(static_cast<size_t>(num_groups));
  for (size_t i = 0; i < groups.size(); ++i) {
    groups[i].backend = i == 0 ? std::move(backend) : makeBackend(model, threads);
    startChunk(groups[i]);
  }
  const std::string backend_name = groups[0].backend->describe();
  // Round robin: each group's search runs while the other groups' batches are
  // on the network. With one group this is the plain alternation of search
  // and evaluation. ONNX Runtime allows concurrent Run() on one session.
  bool any = true;
  while (any) {
    any = false;
    for (Group &g : groups) {
      if (!g.trainer)
        continue;
      any = true;
      if (g.pending.valid()) {
        const auto t = Clock::now();
        eval_s += g.pending.get();
        wait_s += since(t);
      }
      auto t = Clock::now();
      const bool finished =
          g.trainer->doIteration(g.values.data(), g.probs.data(), -1);
      engine_s += since(t);
      if (finished) {
        finishChunk(g);
        startChunk(g);
        continue;
      }
      const int32_t rows = g.trainer->num_requests(-1);
      const float *batch = g.trainer->requests();
      rows_evaluated += static_cast<uint64_t>(rows);
      ++calls;
      if (num_groups == 1) {
        t = Clock::now();
        g.backend->evaluate(batch, rows, g.values.data(), g.probs.data());
        const double s = since(t);
        eval_s += s;
        wait_s += s;
      } else {
        Backend *b = g.backend.get();
        float *values = g.values.data();
        float *probs = g.probs.data();
        g.pending = std::async(std::launch::async, [b, batch, rows, values,
                                                    probs] {
          const auto t0 = Clock::now();
          b->evaluate(batch, rows, values, probs);
          return since(t0);
        });
      }
      if (check) {
        check_values.resize(g.values.size());
        check_probs.resize(g.probs.size());
        check->evaluate(batch, rows, check_values.data(), check_probs.data());
        for (int32_t r = 0; r < rows; ++r) {
          max_value_diff = std::max(
              max_value_diff, std::fabs(g.values[r] - check_values[r]));
          const float *p = g.probs.data() + static_cast<size_t>(r) * kNumMoves;
          const float *q =
              check_probs.data() + static_cast<size_t>(r) * kNumMoves;
          int32_t bp = 0, bq = 0;
          for (int32_t m = 0; m < kNumMoves; ++m) {
            max_prob_diff = std::max(max_prob_diff, std::fabs(p[m] - q[m]));
            bp = p[m] > p[bp] ? m : bp;
            bq = q[m] > q[bq] ? m : bq;
          }
          argmax_diff += bp != bq;
        }
      }
    }
  }
  states_out.close();
  values_out.close();
  policies_out.close();
  const double wall = since(wall_start);

  std::ofstream js{out + "/selfplay.json"};
  js << "{\n"
     << "  \"backend\": \"" << backend_name << "\",\n"
     << "  \"games\": " << games << ",\n"
     << "  \"in_flight\": " << in_flight << ",\n"
     << "  \"groups\": " << num_groups << ",\n"
     << "  \"chunks\": " << chunks_done << ",\n"
     << "  \"searches\": " << searches << ",\n"
     << "  \"searches_per_eval\": " << spe << ",\n"
     << "  \"c_puct\": " << c_puct << ",\n"
     << "  \"epsilon\": " << epsilon << ",\n"
     << "  \"threads\": " << threads << ",\n"
     << "  \"seed\": " << seed << ",\n"
     << "  \"turns\": " << turns_total << ",\n"
     << "  \"sample_rows\": " << states_out.rows() << ",\n"
     << "  \"first_player_score\": " << score_sum / games << ",\n"
     << "  \"network_calls\": " << calls << ",\n"
     << "  \"rows_evaluated\": " << rows_evaluated << ",\n"
     << "  \"engine_seconds\": " << engine_s << ",\n"
     << "  \"eval_seconds\": " << eval_s << ",\n"
     << "  \"eval_wait_seconds\": " << wait_s << ",\n"
     << "  \"write_seconds\": " << write_s << ",\n"
     << "  \"wall_seconds\": " << wall << "\n"
     << "}\n";
  std::printf("#METRIC engine_seconds %.4f\n", engine_s);
  std::printf("#METRIC eval_seconds %.4f\n", eval_s);
  std::printf("#METRIC eval_wait_seconds %.4f\n", wait_s);
  std::printf("#METRIC wall_seconds %.4f\n", wall);
  std::printf("#METRIC turns %lld\n", static_cast<long long>(turns_total));
  std::printf("#METRIC rows_evaluated %llu\n",
              static_cast<unsigned long long>(rows_evaluated));
  if (check)
    std::printf("#METRIC check_max_value_diff %.6g\n"
                "#METRIC check_max_prob_diff %.6g\n"
                "#METRIC check_argmax_differs %llu\n",
                max_value_diff, max_prob_diff,
                static_cast<unsigned long long>(argmax_diff));
  if (a.digest)
    std::printf("#METRIC sample_digest %016llx\n",
                static_cast<unsigned long long>(digest.hash));
  return 0;
}

int runTest(const Args &a) {
  const std::string out = a.str("out");
  const int32_t games = a.i32("games", 400);
  const int32_t searches = a.i32("searches", 1600);
  const int32_t spe = a.i32("spe", 16);
  const float c_puct = a.f32("c-puct", 3.0F);
  const float epsilon = a.f32("epsilon", 0.25F);
  const int32_t threads = a.i32("threads", 20);
  const int32_t seed = a.i32("seed", 1);
  const int32_t logged = a.i32("logged", 0);
  // to_play 0 is the new model, 1 the best, as in main.pyx get_predictions
  auto new_model = makeBackend(a.str("new"), threads);
  auto best_model = makeBackend(a.str("best"), threads);

  Trainer trainer{games,  out,     seed,   searches, spe,
                  c_puct, epsilon, logged, threads,  true};
  const size_t cap = static_cast<size_t>(games) * spe;
  std::vector<float> states(cap * kGameStateSize), values(cap),
      probs(cap * kNumMoves);
  double engine_s = 0, eval_s = 0;
  uint64_t rows_evaluated = 0, calls = 0;
  const auto wall_start = Clock::now();
  int32_t to_play = 0;
  // The same loop as main.pyx play_games in testing mode
  while (true) {
    auto t = Clock::now();
    const bool finished =
        trainer.doIteration(values.data(), probs.data(), to_play);
    engine_s += since(t);
    if (finished)
      break;
    const int32_t rows = trainer.num_requests(to_play);
    if (rows == 0) {
      to_play = 1 - to_play;
      continue;
    }
    trainer.writeRequests(states.data(), to_play);
    t = Clock::now();
    (to_play == 0 ? new_model : best_model)
        ->evaluate(states.data(), rows, values.data(), probs.data());
    eval_s += since(t);
    rows_evaluated += static_cast<uint64_t>(rows);
    ++calls;
  }
  const double wall = since(wall_start);
  trainer.writeScores(out + "/score_verbose.txt");
  const int32_t wins = trainer.numWins();
  const int32_t draws = trainer.numDraws();
  std::ofstream js{out + "/test.json"};
  js << "{\n"
     << "  \"new\": \"" << new_model->describe() << "\",\n"
     << "  \"best\": \"" << best_model->describe() << "\",\n"
     << "  \"games\": " << trainer.numGames() << ",\n"
     << "  \"wins\": " << wins << ",\n"
     << "  \"draws\": " << draws << ",\n"
     << "  \"losses\": " << trainer.numGames() - wins - draws << ",\n"
     << "  \"score\": " << trainer.score() << ",\n"
     << "  \"seed\": " << seed << ",\n"
     << "  \"network_calls\": " << calls << ",\n"
     << "  \"rows_evaluated\": " << rows_evaluated << ",\n"
     << "  \"engine_seconds\": " << engine_s << ",\n"
     << "  \"eval_seconds\": " << eval_s << ",\n"
     << "  \"wall_seconds\": " << wall << "\n"
     << "}\n";
  std::printf("#METRIC wins %d\n#METRIC draws %d\n#METRIC games %d\n", wins,
              draws, trainer.numGames());
  std::printf("#METRIC engine_seconds %.4f\n#METRIC eval_seconds %.4f\n"
              "#METRIC wall_seconds %.4f\n",
              engine_s, eval_s, wall);
  return 0;
}

int runBench(const Args &a) {
  auto backend = makeBackend(a.str("model"), a.i32("threads", 20));
  const int32_t reps = a.i32("reps", 50);
  std::string list = a.str("rows", "1000,4000,8000,16000,32000,64000");
  std::vector<int32_t> sizes;
  for (size_t pos = 0; pos < list.size();) {
    const size_t comma = list.find(',', pos);
    sizes.push_back(std::atoi(list.substr(pos, comma - pos).c_str()));
    pos = comma == std::string::npos ? list.size() : comma + 1;
  }
  const int32_t max_rows = *std::max_element(sizes.begin(), sizes.end());
  std::vector<float> states(static_cast<size_t>(max_rows) * kGameStateSize);
  for (size_t i = 0; i < states.size(); ++i)
    states[i] = (i * 2654435761U >> 7) % 3 == 0 ? 1.0F : 0.0F;
  std::vector<float> values(max_rows), probs(static_cast<size_t>(max_rows) * kNumMoves);
  std::printf("%s\n", backend->describe().c_str());
  for (const int32_t rows : sizes) {
    for (int32_t i = 0; i < 3; ++i)  // warm up this shape
      backend->evaluate(states.data(), rows, values.data(), probs.data());
    const auto t = Clock::now();
    for (int32_t i = 0; i < reps; ++i)
      backend->evaluate(states.data(), rows, values.data(), probs.data());
    const double s = since(t) / reps;
    std::printf("rows %6d  %8.3f ms/call  %7.1f ns/row\n", rows, s * 1e3,
                s * 1e9 / rows);
  }
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Args a = parse(argc, argv);
    if (a.mode == "train")
      return runTrain(a);
    if (a.mode == "test")
      return runTest(a);
    if (a.mode == "bench")
      return runBench(a);
    throw std::runtime_error("mode must be train or test");
  } catch (const std::exception &e) {
    std::fprintf(stderr, "corintho_play: %s\n", e.what());
    return 1;
  }
}
