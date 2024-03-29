#include "trainer.h"

#include <cstdint>

#include <algorithm>
#include <queue>
#include <string>
#include <vector>

#include <gsl/gsl>
#include <omp.h>

#include "node.h"
#include "selfplayer.h"
#include "trainmc.h"
#include "util.h"

Trainer::Trainer(int32_t num_games, const std::string &log_folder,
                 int32_t seed, int32_t max_searches, int32_t searches_per_eval,
                 float c_puct, float epsilon, int32_t num_logged,
                 int32_t num_threads, bool testing)
    : is_done_{std::vector<bool>(num_games, false)},
      max_searches_{max_searches}, searches_per_eval_{searches_per_eval},
      num_threads_{num_threads}, generator_{gsl::narrow_cast<uint32_t>(seed)} {
  assert(num_games > 0);
  assert(num_logged >= 0);
  assert(num_logged <= num_games);
  assert(max_searches > 0);
  assert(searches_per_eval > 0);
  assert(max_searches >= searches_per_eval);
  assert(c_puct > 0.0);
  assert(epsilon >= 0.0);
  assert(epsilon <= 1.0);
  assert(num_threads > 0);
  initialize(num_games, log_folder, max_searches, searches_per_eval, c_puct,
             epsilon, num_logged, testing);
}

int32_t Trainer::num_requests(int32_t to_play) const noexcept {
  int32_t num_requests = 0;
  for (const auto &game : games_) {
    if (!is_done_[&game - &games_[0]] &&
        ((to_play != 0 && to_play != 1) ||
         game.to_play() == (to_play + game.parity()) % 2)) {
      num_requests += game.num_requests();
    }
  }
  return num_requests;
}

int32_t Trainer::num_samples() const noexcept {
  int32_t num_samples = 0;
  for (const auto &game : games_) {
    num_samples += game.num_samples();
  }
  return num_samples;
}

float Trainer::score() const noexcept {
  float score = 0;
  for (size_t i = 0; i < games_.size(); i += 2) {
    score += games_[i].score();
  }
  for (size_t i = 1; i < games_.size(); i += 2) {
    score += 1.0 - games_[i].score();
  }
  return score / games_.size();
}

float Trainer::avg_mate_length() const noexcept {
  int32_t total_length = 0;
  for (const auto &game : games_) {
    assert(is_done_[&game - &games_[0]]);
    total_length += game.mate_length();
  }
  return static_cast<float>(total_length) / games_.size();
}

void Trainer::writeRequests(float *game_states,
                            int32_t to_play) const noexcept {
  int32_t offset = 0;
  // Testing mode
  // Only count requests from one player
  if (to_play == 0 || to_play == 1) {
    for (size_t i = 0; i < games_.size(); ++i) {
      if (games_[i].to_play() == (to_play + games_[i].parity()) % 2 &&
          !is_done_[i]) {
        games_[i].writeRequests(game_states + offset * kGameStateSize);
        offset += games_[i].num_requests();
      }
    }
    return;
  }
  // Training mode
  for (size_t i = 0; i < games_.size(); ++i) {
    if (!is_done_[i]) {
      games_[i].writeRequests(game_states + offset * kGameStateSize);
      offset += games_[i].num_requests();
    }
  }
}

void Trainer::writeSamples(float *game_states, float *eval_samples,
                           float *prob_samples) const noexcept {
  int32_t offset = 0;
  for (size_t i = 0; i < games_.size(); ++i) {
    games_[i].writeSamples(game_states +
                               offset * kGameStateSize * kNumSymmetries,
                           eval_samples + offset * kNumSymmetries,
                           prob_samples + offset * kNumMoves * kNumSymmetries);
    offset += games_[i].num_samples();
  }
}

void Trainer::writeScores(const std::string &filename) const {
  float scores[games_.size()];
  for (size_t i = 0; i < games_.size(); i += 2) {
    scores[i] = games_[i].score();
  }
  for (size_t i = 1; i < games_.size(); i += 2) {
    scores[i] = 1.0 - games_[i].score();
  }
  // First player score
  int32_t wins = 0;
  int32_t draws = 0;
  for (size_t i = 0; i < games_.size(); i += 2) {
    if (scores[i] == 1.0) {
      ++wins;
    } else if (scores[i] == 0.5) {
      ++draws;
    }
  }
  std::ofstream file = std::ofstream{filename, std::ofstream::out};
  file << "First player wins: " << wins << " / " << games_.size() / 2 << " = "
       << static_cast<float>(wins) / (games_.size() / 2)
       << "\nFirst player draws: " << draws << " / " << games_.size() / 2
       << " = " << static_cast<float>(draws) / (games_.size() / 2)
       << "\nFirst player losses: " << games_.size() / 2 - wins - draws
       << " / " << games_.size() / 2 << " = "
       << static_cast<float>(games_.size() / 2 - wins - draws) /
              (games_.size() / 2)
       << '\n';
  // Second player score
  wins = 0;
  draws = 0;
  for (size_t i = 1; i < games_.size(); i += 2) {
    if (scores[i] == 1.0) {
      ++wins;
    } else if (scores[i] == 0.5) {
      ++draws;
    }
  }
  file << "Second player wins: " << wins << " / " << games_.size() / 2 << " = "
       << static_cast<float>(wins) / (games_.size() / 2)
       << "\nSecond player draws: " << draws << " / " << games_.size() / 2
       << " = " << static_cast<float>(draws) / (games_.size() / 2)
       << "\nSecond player losses: " << games_.size() / 2 - wins - draws
       << " / " << games_.size() / 2 << " = "
       << static_cast<float>(games_.size() / 2 - wins - draws) /
              (games_.size() / 2)
       << '\n';
}

bool Trainer::doIteration(float eval[], float probs[], int32_t to_play) {
  // Training
  if (to_play != 0 && to_play != 1) {
    // Compute offsets for evaluations and probabilities since we use
    // multiprocessing.
    int32_t offset = 0;
    int32_t offsets[games_.size()] = {0};
    for (size_t i = 1; i < games_.size(); ++i) {
      offset += games_[i - 1].num_requests();
      offsets[i] = offset;
    }
    omp_set_num_threads(num_threads_);
#pragma omp parallel for
    for (size_t i = 0; i < games_.size(); ++i) {
      if (!is_done_[i]) {
        // We offset the start of the games to try to get an even distribution
        // of the games across the number of searches in a move. This way, the
        // total number of nodes will be more even. This reduces peak memory
        // usage. Avoid division by 0 in the rare case that games_.size() <
        // max_searches_
        if (i / std::max(games_.size() / max_searches_,
                         static_cast<size_t>(1)) <=
            static_cast<size_t>(searches_done_)) {
          // First search does not depend on pointers being null
          bool done = games_[i].doIteration(eval + offsets[i],
                                            probs + kNumMoves * offsets[i]);
          // Game is done
          if (done) {
            is_done_[i] = true;
          }
        }
      }
    }
    ++searches_done_;
    for (size_t i = 0; i < games_.size(); ++i) {
      if (!is_done_[i]) {
        return false;
      }
    }
    return true;
  }
  // Testing
  int32_t offset = 0;
  int32_t offsets[games_.size()] = {0};
  for (size_t i = 1; i < games_.size(); ++i) {
    // Only count games from one player
    if (games_[i - 1].to_play() == (to_play + games_[i - 1].parity()) % 2 &&
        !is_done_[i - 1]) {
      offset += games_[i - 1].num_requests();
    }
    offsets[i] = offset;
  }
  omp_set_num_threads(num_threads_);
#pragma omp parallel for
  for (size_t i = 0; i < games_.size(); ++i) {
    // No offset in game start (there are not enough games for memory usage to
    // matter).
    if (games_[i].to_play() == (to_play + games_[i].parity()) % 2 &&
        !is_done_[i]) {
      bool done = games_[i].doIteration(eval + offsets[i],
                                        probs + kNumMoves * offsets[i]);
      if (done) {
        is_done_[i] = true;
      }
    }
  }
  for (size_t i = 0; i < games_.size(); ++i) {
    if (!is_done_[i]) {
      return false;
    }
  }
  return true;
}

void Trainer::initialize(int32_t num_games, const std::string &log_folder,
                         int32_t max_searches, int32_t searches_per_eval,
                         float c_puct, float epsilon, int32_t num_logged,
                         bool testing) {
  games_.reserve(num_games);
  for (int32_t i = 0; i < num_logged; ++i) {
    games_.emplace_back(
        generator_(), max_searches, searches_per_eval, c_puct, epsilon,
        std::make_unique<std::ofstream>(log_folder + "/game_" +
                                            std::to_string(i) + ".txt",
                                        std::ofstream::out),
        testing, i % 2);  // Generate parity for test games (changes who plays
                          // first). Does not affect training games
  }
  for (int32_t i = num_logged; i < num_games; ++i) {
    games_.emplace_back(generator_(), max_searches, searches_per_eval, c_puct,
                        epsilon, nullptr, testing, i % 2);
  }
}