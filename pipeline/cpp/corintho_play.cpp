// Self-play and testing driver for the GPU pipeline (worklog entries 21, 29).
//
//   corintho_play train --model M --games N --out DIR [options]
//   corintho_play test  --new M1 --best M2 --games N --out DIR [options]
//   corintho_play bench --model M [--rows 1000,4000,16000] [--reps 50]
//
// M is a .onnx file (ONNX Runtime, CUDA) or a .mlp file (the CPU network).
// Prefix a .onnx path with trt: or trt16: for the TensorRT provider (fp32
// with TF32, or fp16); engines are cached in trt_cache/ beside the model.
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
//   --solve-p 0     end games by exact solution once the position's horizon
//                   P (2 x reserves + occupied spaces) is at most this; 0:
//                   off (entry 14). --solve-cap 5000000 (positions per
//                   solve; a capped solve is retried once at 20x, then
//                   counts as a draw),
//                   --solve-threads 4, --solve-table 22 (log2 entries)
//   --relabel-p 0   train: play games out as usual, but label every sample
//                   at horizon P <= this with its exact value, and earlier
//                   samples with the first one's, flipped back (entry 20);
//                   uses the --solve-* pool settings. Not with --solve-p
//   --node-p 0      solve search leaves with horizon P <= this exactly
//                   instead of asking the network (entry 15); --node-cap
//                   20000 positions per leaf, then the network
//   --shared-tree 0 train: 1 = both sides of a game search one tree
//                   (entry 17); each move still gets --searches new ones
//   --node-side both  test only: which side uses the node solver (new, best)
//   --stagger 0     train: iterations over which a chunk's games start (0:
//                   the Trainer's original ~16-turn rule; 100 = one turn)
//   --digest        also print the FNV-1a sample digest (as selfplay_nn)
//   --cache L       train only: cache network results for 2^L positions (0,
//                   the default: off), shared by all games and keyed up to
//                   board symmetry (eval_cache.h); ~216 bytes per entry
//   --cache-sym 1   0: key positions exactly, never serving a rotated copy
//   --check M2      train only: also evaluate every batch with M2 and report
//                   the largest differences (games follow --model)

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <utility>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend.h"
#include "eval_cache.h"
#include "node.h"
#include "npy.h"
#include "solver_pool.h"
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
  int64_t i64(const std::string &k, int64_t def) const {
    auto it = kv.find(k);
    return it == kv.end() ? def : std::atoll(it->second.c_str());
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
  // The network call in flight: {seconds on the network, seconds in the cache}
  std::future<std::pair<double, double>> pending;
  int32_t games{0};
  int32_t chunk{0};  // index of the chunk being played
  // With --cache: this call's rows, and the misses sent to the network
  EvalCache::Call cache_call;
  std::vector<float> miss_states, miss_values, miss_probs;
};

// Runs submitted jobs one at a time, in order, on its own thread. The
// network calls of both groups go through it: with --cache every lookup and
// insert then happens on this one thread (no locking), off the engine's
// critical path while the main thread searches the other group.
class SerialWorker {
 public:
  using Result = std::pair<double, double>;
  SerialWorker() : thread_([this] { run(); }) {}
  ~SerialWorker() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    wake_.notify_one();
    thread_.join();
  }
  std::future<Result> submit(std::function<Result()> job) {
    std::packaged_task<Result()> task(std::move(job));
    std::future<Result> result = task.get_future();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      jobs_.push_back(std::move(task));
    }
    wake_.notify_one();
    return result;
  }

 private:
  void run() {
    for (;;) {
      std::packaged_task<Result()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        wake_.wait(lock, [this] { return stop_ || !jobs_.empty(); });
        if (jobs_.empty())
          return;
        task = std::move(jobs_.front());
        jobs_.pop_front();
      }
      task();
    }
  }
  std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<std::packaged_task<Result()>> jobs_;
  bool stop_{false};
  std::thread thread_;
};

// Optional measurement (worklog 2026-09-25-nn-architectures, entry 07): with
// CORINTHO_DUP_DUMP=FILE, every row sent to the network is recorded as
// {game id, call index, 64-bit hash of the 70 inputs}, to count how often a
// position is evaluated more than once, within a game or across games.
struct DupRecord {
  uint32_t game, call;
  uint64_t hash, canonical;  // canonical: least hash over the 8 symmetries
  uint8_t root_p, leaf_p;    // horizon P of the search root and of the row
  uint8_t pad[6];
};
// P = 2 * reserves + occupied spaces, from network inputs
uint8_t horizonOf(const float *row) {
  int32_t p = 0;
  for (int32_t s = 0; s < kBoardSize; ++s)
    p += (row[4 * s] + row[4 * s + 1] + row[4 * s + 2]) > 0.0F;
  for (int32_t i = 0; i < 6; ++i)
    p += 2 * static_cast<int32_t>(row[4 * kBoardSize + i] * 4.0F + 0.5F);
  return static_cast<uint8_t>(p);
}
uint64_t rowHash(const float *row, const int32_t *perm = nullptr) {
  uint64_t h = 1469598103934665603ULL;  // FNV-1a over the inputs x 4
  for (int32_t i = 0; i < kGameStateSize; ++i) {
    const int32_t j =
        perm == nullptr || i >= 4 * kBoardSize ? i
                                               : perm[i / 4] * 4 + i % 4;
    h ^= static_cast<uint64_t>(row[j] * 4.0F + 0.5F);
    h *= 1099511628211ULL;
  }
  h ^= h >> 31;  // final mix; FNV's low bits alone are weak
  h *= 0x9E3779B97F4A7C15ULL;
  return h ^ (h >> 29);
}

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
  // Exact solutions from horizon --solve-p down (entry 14): one pool for
  // the whole run, so its tables stay warm across chunks and groups
  const int32_t solve_p = a.i32("solve-p", 0);
  const int32_t relabel_p = a.i32("relabel-p", 0);
  if (solve_p > 0 && relabel_p > 0) {
    std::fprintf(stderr, "--solve-p and --relabel-p exclude each other\n");
    return 2;
  }
  int64_t relabelled = 0, relabel_changed = 0;
  std::unique_ptr<SolverPool> solver_pool;
  if (solve_p > 0 || relabel_p > 0)
    solver_pool = std::make_unique<SolverPool>(
        a.i32("solve-threads", 4), a.i32("solve-table", 22),
        static_cast<uint64_t>(a.i64("solve-cap", 5000000)));
  // Search leaves solved exactly from horizon --node-p down (entry 15)
  const int32_t node_p = a.i32("node-p", 0);
  const bool shared_tree = a.i32("shared-tree", 0) != 0;
  const uint64_t node_cap = static_cast<uint64_t>(a.i64("node-cap", 20000));
  int64_t adjudicated = 0, solve_unknown = 0;
  double solve_wait_s = 0;  // waiting for solves at chunk ends
  const int32_t cache_log2 = a.i32("cache", 0);
  // CORINTHO_CACHE_VERIFY=1 (with --groups 1): also evaluate every batch in
  // full and compare the rows served from the cache. [0]: rows whose own
  // frame is canonical, [1]: the others; {rows, top move differs, max |dp|,
  // max |dv|}
  const bool verify = std::getenv("CORINTHO_CACHE_VERIFY") != nullptr;
  std::vector<float> verify_values, verify_probs;
  double verify_stats[2][4] = {};
  int32_t verify_printed = 0;
  std::unique_ptr<EvalCache> cache;
  if (cache_log2 > 0)
    cache = std::make_unique<EvalCache>(cache_log2, a.i32("cache-sym", 1) != 0);
  // One network call for group g's batch: with the cache, the lookup fills
  // the hits, only the misses go to the network, and their results are
  // scattered back and stored. Returns {network seconds, cache seconds}.
  auto runCall = [&cache](Group &g, const float *batch,
                          int32_t rows) -> std::pair<double, double> {
    if (!cache) {
      const auto t0 = Clock::now();
      g.backend->evaluate(batch, rows, g.values.data(), g.probs.data());
      return {since(t0), 0.0};
    }
    auto t0 = Clock::now();
    cache->lookup(batch, rows, g.values.data(), g.probs.data(), g.cache_call);
    const auto &misses = g.cache_call.misses;
    const size_t n = misses.size();
    g.miss_states.resize(n * kGameStateSize);
    g.miss_values.resize(n);
    g.miss_probs.resize(n * kNumMoves);
    for (size_t i = 0; i < n; ++i)
      std::copy_n(batch + static_cast<size_t>(misses[i]) * kGameStateSize,
                  kGameStateSize, g.miss_states.data() + i * kGameStateSize);
    double cache_seconds = since(t0);
    t0 = Clock::now();
    if (n > 0)
      g.backend->evaluate(g.miss_states.data(), static_cast<int32_t>(n),
                          g.miss_values.data(), g.miss_probs.data());
    const double network_seconds = since(t0);
    t0 = Clock::now();
    for (size_t i = 0; i < n; ++i) {
      const size_t r = static_cast<size_t>(misses[i]);
      g.values[r] = g.miss_values[i];
      std::copy_n(g.miss_probs.data() + i * kNumMoves, kNumMoves,
                  g.probs.data() + r * kNumMoves);
    }
    cache->insert(g.cache_call, g.values.data(), g.probs.data());
    cache_seconds += since(t0);
    return {network_seconds, cache_seconds};
  };
  SerialWorker worker;

  auto backend = makeBackend(model, threads);
  std::unique_ptr<Backend> check;
  if (a.kv.count("check")) {
    if (num_groups != 1)
      throw std::runtime_error("--check needs --groups 1");
    if (cache_log2 > 0)
      throw std::runtime_error("--check compares whole batches: not with --cache");
    check = makeBackend(a.str("check"), threads);
  }
  std::vector<float> check_values, check_probs;
  float max_value_diff = 0, max_prob_diff = 0;
  uint64_t argmax_diff = 0;
  NpyWriter states_out{out + "/states.npy", kGameStateSize};
  NpyWriter values_out{out + "/values.npy", 1};
  NpyWriter policies_out{out + "/policies.npy", kNumMoves};

  double engine_s = 0, eval_s = 0, wait_s = 0, write_s = 0, cache_s = 0;
  uint64_t rows_evaluated = 0, calls = 0;
  int64_t turns_total = 0;
  double score_sum = 0;
  Fnv digest;
  const auto wall_start = Clock::now();
  std::vector<float> st, vs, ps;
  int32_t chunks_started = 0, chunks_done = 0, games_started = 0;
  FILE *dup_file = nullptr;
  if (const char *dup = std::getenv("CORINTHO_DUP_DUMP"))
    dup_file = std::fopen(dup, "wb");
  std::vector<DupRecord> dup_buf;

  // Chunk c gets seed + c, and chunks are numbered in the order they start,
  // so a run is deterministic for a given --groups and --in-flight
  auto startChunk = [&](Group &g) {
    if (games_started >= games) {
      g.trainer.reset();
      return;
    }
    g.games = std::min(in_flight, games - games_started);
    g.chunk = chunks_started;
    // Only the first chunk logs games
    g.trainer = std::make_unique<Trainer>(
        g.games, out, seed + chunks_started, searches, spe, c_puct, epsilon,
        chunks_started == 0 ? logged : 0, threads, false);
    g.trainer->set_stagger_iterations(stagger);
    if (solve_p > 0)
      g.trainer->setSolver(solver_pool.get(), solve_p);
    if (relabel_p > 0)
      g.trainer->setRelabel(solver_pool.get(), relabel_p);
    if (node_p > 0)
      g.trainer->setNodeSolver(node_p, node_cap);
    if (shared_tree)
      g.trainer->setSharedTree(true);
    g.values.assign(static_cast<size_t>(g.games) * spe, 0.0F);
    g.probs.assign(static_cast<size_t>(g.games) * spe * kNumMoves, 0.0F);
    games_started += g.games;
    ++chunks_started;
  };
  auto finishChunk = [&](Group &g) {
    const auto t = Clock::now();
    Trainer &trainer = *g.trainer;
    const auto t_solves = Clock::now();
    solve_unknown += trainer.finalizeSolves();
    solve_wait_s += since(t_solves);
    adjudicated += trainer.numAdjudicated();
    trainer.relabelCounts(relabelled, relabel_changed);
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
        const auto [network_seconds, cache_seconds] = g.pending.get();
        eval_s += network_seconds;
        cache_s += cache_seconds;
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
      if (dup_file != nullptr) {
        dup_buf.resize(static_cast<size_t>(rows));
        for (int32_t r = 0; r < rows; ++r) {
          const uint32_t game = static_cast<uint32_t>(
              g.chunk * in_flight + g.trainer->gameInSlot(r / spe));
          const float *row = batch + static_cast<size_t>(r) * kGameStateSize;
          uint64_t canonical = rowHash(row);
          for (int32_t k = 1; k < kNumSymmetries; ++k)
            canonical = std::min(canonical, rowHash(row, space_symmetries[k]));
          const int32_t slot_game = g.trainer->gameInSlot(r / spe);
          const Node *root = g.trainer->searchRoot(slot_game);
          float root_state[kGameStateSize];
          uint8_t root_p = 255;
          if (root != nullptr) {
            root->get_game().writeGameState(root_state);
            root_p = horizonOf(root_state);
          }
          dup_buf[static_cast<size_t>(r)] = {
              game,     static_cast<uint32_t>(calls), rowHash(row), canonical,
              root_p,   horizonOf(row),               {}};
        }
        std::fwrite(dup_buf.data(), sizeof(DupRecord), dup_buf.size(),
                    dup_file);
      }
      ++calls;
      if (num_groups == 1) {
        t = Clock::now();
        const auto [network_seconds, cache_seconds] = runCall(g, batch, rows);
        eval_s += network_seconds;
        cache_s += cache_seconds;
        wait_s += since(t);
        if (cache && verify) {
          // Evaluate everything directly and compare the cached rows
          verify_values.resize(static_cast<size_t>(rows));
          verify_probs.resize(static_cast<size_t>(rows) * kNumMoves);
          g.backend->evaluate(batch, rows, verify_values.data(),
                              verify_probs.data());
          std::vector<uint8_t> missed(static_cast<size_t>(rows));
          for (const int32_t r : g.cache_call.misses)
            missed[static_cast<size_t>(r)] = 1;
          for (int32_t r = 0; r < rows; ++r) {
            if (missed[static_cast<size_t>(r)])
              continue;
            const bool same_frame =
                g.cache_call.frame[static_cast<size_t>(r)] ==
                g.cache_call.stored_frame[static_cast<size_t>(r)];
            const float *p = g.probs.data() + static_cast<size_t>(r) * kNumMoves;
            const float *q =
                verify_probs.data() + static_cast<size_t>(r) * kNumMoves;
            int32_t bp = 0, bq = 0;
            float dp = 0;
            for (int32_t m = 0; m < kNumMoves; ++m) {
              dp = std::max(dp, std::fabs(p[m] - q[m]));
              bp = p[m] > p[bp] ? m : bp;
              bq = q[m] > q[bq] ? m : bq;
            }
            if (same_frame && bp != bq && verify_printed < 5) {
              ++verify_printed;
              std::printf("#MISMATCH call %lld row %d frame %d: cached v %.4f "
                          "top %d (%.3f); direct v %.4f top %d (%.3f)\n",
                          static_cast<long long>(calls), r,
                          g.cache_call.frame[static_cast<size_t>(r)],
                          g.values[r], bp, p[bp],
                          verify_values[static_cast<size_t>(r)], bq, q[bq]);
            }
            auto &st = verify_stats[same_frame ? 0 : 1];
            st[0] += 1;
            st[1] += bp != bq;
            st[2] = std::max(st[2], static_cast<double>(dp));
            st[3] = std::max(st[3], static_cast<double>(std::fabs(
                g.values[r] - verify_values[static_cast<size_t>(r)])));
          }
        }
      } else {
        Group *gp = &g;
        g.pending = worker.submit(
            [&runCall, gp, batch, rows] { return runCall(*gp, batch, rows); });
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

  uint64_t node_attempts = 0, node_solved = 0;
  double node_seconds = 0;
  TrainMC::nodeSolveStats(node_attempts, node_solved, node_seconds);
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
     << "  \"cache_log2\": " << cache_log2 << ",\n"
     << "  \"cache_lookups\": " << (cache ? cache->lookups() : 0) << ",\n"
     << "  \"cache_hits\": " << (cache ? cache->hits() : 0) << ",\n"
     << "  \"cache_seconds\": " << cache_s << ",\n"
     << "  \"write_seconds\": " << write_s << ",\n"
     << "  \"solve_horizon\": " << solve_p << ",\n"
     << "  \"adjudicated_games\": " << adjudicated << ",\n"
     << "  \"solve_unknown_games\": " << solve_unknown << ",\n"
     << "  \"node_horizon\": " << node_p << ",\n"
     << "  \"shared_tree\": " << shared_tree << ",\n"
     << "  \"relabel_horizon\": " << relabel_p << ",\n"
     << "  \"relabelled_samples\": " << relabelled << ",\n"
     << "  \"relabel_changed\": " << relabel_changed << ",\n"
     << "  \"node_solve_attempts\": " << node_attempts << ",\n"
     << "  \"node_solved\": " << node_solved << ",\n"
     << "  \"node_solve_seconds\": " << node_seconds << ",\n"
     << "  \"solve_wait_seconds\": " << solve_wait_s << ",\n"
     << "  \"solves\": " << (solver_pool ? solver_pool->solves() : 0) << ",\n"
     << "  \"solves_capped\": " << (solver_pool ? solver_pool->capped() : 0)
     << ",\n"
     << "  \"solve_seconds\": " << (solver_pool ? solver_pool->seconds() : 0.0)
     << ",\n"
     << "  \"solve_max_seconds\": "
     << (solver_pool ? solver_pool->max_seconds() : 0.0) << ",\n"
     << "  \"wall_seconds\": " << wall << "\n"
     << "}\n";
  std::printf("#METRIC engine_seconds %.4f\n", engine_s);
  std::printf("#METRIC eval_seconds %.4f\n", eval_s);
  std::printf("#METRIC eval_wait_seconds %.4f\n", wait_s);
  if (cache)
    std::printf("#METRIC cache_hit_rate %.4f\n#METRIC cache_seconds %.4f\n",
                static_cast<double>(cache->hits()) /
                    static_cast<double>(std::max<uint64_t>(1, cache->lookups())),
                cache_s);
  if (dup_file != nullptr)
    std::fclose(dup_file);
  if (verify)
    for (int32_t i = 0; i < 2; ++i)
      std::printf("#VERIFY %s: %.0f cached rows, top move differs %.0f "
                  "(%.4f%%), max |dp| %.2e, max |dv| %.2e\n",
                  i == 0 ? "stored from the same frame" : "from another frame",
                  verify_stats[i][0], verify_stats[i][1],
                  100 * verify_stats[i][1] / std::max(1.0, verify_stats[i][0]),
                  verify_stats[i][2], verify_stats[i][3]);
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
  // Matches end games by exact solution the same way (entry 14): both
  // players alike, as deployment would. The node solver can be given to one
  // side only (--node-side new|best), to measure what it is worth
  const std::string node_side = a.str("node-side", "both");
  if (node_side != "both" && node_side != "new" && node_side != "best")
    throw std::runtime_error("--node-side: both, new or best");
  if (a.i32("node-p", 0) > 0)
    trainer.setNodeSolver(a.i32("node-p", 0),
                          static_cast<uint64_t>(a.i64("node-cap", 20000)),
                          node_side == "both" ? -1 : node_side == "new" ? 0 : 1);
  const int32_t solve_p = a.i32("solve-p", 0);
  if (solve_p > 0)
    trainer.enableSolver(solve_p,
                         static_cast<uint64_t>(a.i64("solve-cap", 5000000)),
                         a.i32("solve-threads", 4), a.i32("solve-table", 22));
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
  trainer.finalizeSolves();  // exact results of games ended by solution
  trainer.writeScores(out + "/score_verbose.txt");
  // The new agent's score in every game, in game order: the same seed plays
  // the same games, so two matches can be compared game by game
  {
    std::ofstream per_game{out + "/game_scores.txt"};
    for (int32_t i = 0; i < trainer.numGames(); ++i)
      per_game << trainer.newScore(i) << '\n';
  }
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
     << "  \"solve_horizon\": " << solve_p << ",\n"
     << "  \"node_horizon\": " << a.i32("node-p", 0) << ",\n"
     << "  \"node_side\": \"" << node_side << "\",\n"
     << "  \"adjudicated_games\": " << trainer.numAdjudicated() << ",\n"
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
