// Golden digests: behavioural fingerprints of the engine.
//
// A pure performance optimization must not change what the engine computes.
// These digests make that claim checkable rather than assumed: run before a
// change, run after, and require the digest to be byte-identical.
//
//   game    hashes getLegalMoves output over a fixed position corpus.
//           Any move-generation rewrite must preserve this exactly.
//   engine  hashes the full training-sample stream from a fixed-seed run.
//           Catches any change to search semantics.
//
// A change that is *meant* to alter behaviour (root-only Dirichlet noise, for
// example) will move the engine digest. That is expected -- but it must be
// stated deliberately in the commit, never discovered later.

#include <cstdint>

#include <bitset>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "bench_util.h"
#include "game.h"
#include "trainer.h"
#include "util.h"

namespace {

/// @brief FNV-1a over raw bytes.
class Digest {
 public:
  void add(const void *data, size_t bytes) {
    const uint8_t *p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < bytes; ++i) {
      hash_ = (hash_ ^ p[i]) * 0x100000001B3ULL;
    }
  }
  void add(uint64_t value) { add(&value, sizeof(value)); }
  uint64_t value() const { return hash_; }

 private:
  uint64_t hash_{0xCBF29CE484222325ULL};
};

/// @brief Same corpus construction as micro.cpp, kept in step deliberately.
std::vector<Game> buildCorpus(size_t target, uint64_t seed) {
  std::vector<Game> corpus;
  corpus.reserve(target);
  uint64_t rng = seed;
  std::bitset<kNumMoves> legal;
  while (corpus.size() < target) {
    Game game;
    while (corpus.size() < target) {
      corpus.push_back(game);
      game.getLegalMoves(legal);
      const size_t count = legal.count();
      if (count == 0)
        break;
      size_t pick = bench::splitmix64(rng) % count;
      int32_t move_id = 0;
      for (int32_t i = 0; i < kNumMoves; ++i) {
        if (legal[i]) {
          if (pick == 0) {
            move_id = i;
            break;
          }
          --pick;
        }
      }
      game.doMove(move_id);
    }
  }
  return corpus;
}

/// @brief Digest of legal-move generation over the corpus.
uint64_t gameDigest(size_t corpus_size) {
  std::vector<Game> corpus = buildCorpus(corpus_size, 0x1234ABCDULL);
  Digest digest;
  std::bitset<kNumMoves> legal;
  float state[kGameStateSize];
  for (const Game &game : corpus) {
    const bool lines = game.getLegalMoves(legal);
    digest.add(lines ? 1ULL : 0ULL);
    // Hash the bitset in 32-bit chunks; to_string would allocate.
    for (int32_t i = 0; i < kNumMoves; i += 32) {
      uint32_t chunk = 0;
      for (int32_t j = 0; j < 32 && i + j < kNumMoves; ++j) {
        if (legal[i + j])
          chunk |= (1u << j);
      }
      digest.add(&chunk, sizeof(chunk));
    }
    // writeGameState is part of the same contract.
    game.writeGameState(state);
    digest.add(state, sizeof(state));
  }
  return digest.value();
}

/// @brief Digest of a complete fixed-seed self-play run.
/// @details Single-threaded so that the result cannot depend on scheduling.
uint64_t engineDigest(int32_t num_games, int32_t max_searches,
                      int32_t searches_per_eval) {
  const size_t batch = static_cast<size_t>(num_games) * searches_per_eval;
  std::vector<float> evals(batch, 0.0F);
  std::vector<float> probs(batch * kNumMoves, 0.0F);
  std::vector<float> game_states(batch * kGameStateSize, 0.0F);

  Trainer trainer{num_games, std::string("/tmp"), 4242,  max_searches,
                  searches_per_eval, 3.0F, 0.25F, 0, 1, false};

  while (true) {
    if (trainer.doIteration(evals.data(), probs.data(), -1))
      break;
    const int32_t num_requests = trainer.num_requests(-1);
    if (num_requests == 0)
      break;
    trainer.writeRequests(game_states.data(), -1);
    bench::stubEvaluate(game_states.data(), num_requests, evals.data(),
                        probs.data());
  }

  const int32_t num_samples = trainer.num_samples();
  std::vector<float> sample_states(
      static_cast<size_t>(num_samples) * kGameStateSize * kNumSymmetries, 0.0F);
  std::vector<float> sample_evals(
      static_cast<size_t>(num_samples) * kNumSymmetries, 0.0F);
  std::vector<float> sample_probs(
      static_cast<size_t>(num_samples) * kNumMoves * kNumSymmetries, 0.0F);
  trainer.writeSamples(sample_states.data(), sample_evals.data(),
                       sample_probs.data());

  Digest digest;
  digest.add(static_cast<uint64_t>(num_samples));
  digest.add(sample_states.data(), sample_states.size() * sizeof(float));
  digest.add(sample_evals.data(), sample_evals.size() * sizeof(float));
  digest.add(sample_probs.data(), sample_probs.size() * sizeof(float));
  const float score = trainer.score();
  digest.add(&score, sizeof(score));
  return digest.value();
}

}  // namespace

int main(int argc, char **argv) {
  const std::string mode = argc > 1 ? argv[1] : "all";

  if (mode == "game" || mode == "all") {
    const uint64_t d = gameDigest(20000);
    std::printf("game digest   (20000 positions)      %016llx\n",
                static_cast<unsigned long long>(d));
    std::printf("#METRIC digest_game %016llx\n",
                static_cast<unsigned long long>(d));
  }
  if (mode == "engine" || mode == "all") {
    const uint64_t d = engineDigest(20, 200, 16);
    std::printf("engine digest (20 games, 200 searches) %016llx\n",
                static_cast<unsigned long long>(d));
    std::printf("#METRIC digest_engine %016llx\n",
                static_cast<unsigned long long>(d));
  }
  return 0;
}
