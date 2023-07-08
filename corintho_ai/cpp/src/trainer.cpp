#include "trainer.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

#include <omp.h>

#include "node.h"
#include "selfplayer.h"
#include "trainmc.h"
#include "util.h"

Trainer::Trainer(int32_t num_games, const std::string &logging_folder,
                 int32_t seed, int32_t max_searches, int32_t searches_per_eval,
                 float c_puct, float epsilon, int32_t num_logged,
                 int32_t num_threads, bool testing)
    : is_done_{std::vector<bool>(num_games, false)},
      max_searches_{max_searches}, searches_per_eval_{searches_per_eval},
      num_threads_{num_threads}, generator_{seed} {
  assert(num_games > 0);
  assert(num_games % 2 == 0);
  assert(num_logged >= 0);
  assert(num_logged <= num_games);
  assert(max_searches >= 2);
  assert(searches_per_eval > 0);
  assert(max_searches >= searches_per_eval);
  assert(c_puct > 0);
  assert(epsilon >= 0);
  assert(epsilon <= 1);
  assert(num_threads > 0);
  initialize(num_games, logging_folder, max_searches, searches_per_eval, c_puct,
             epsilon, num_logged, testing);
}

int32_t Trainer::numRequests(int32_t to_play) const noexcept {
  int32_t num_requests = 0;
  for (const auto &game : games_) {
    if (!is_done_[&game - &games_[0]] &&
        ((game->to_play() != 0 && game->to_play() != 1) ||
         game->to_play() == (to_play + game->parity()) % 2)) {
      num_requests += game->numRequests();
    }
  }
  return num_requests;
}

int32_t Trainer::numSamples() const noexcept {
  int32_t num_samples = 0;
  for (const auto &game : games_) {
    num_samples += game->numSamples();
  }
  return num_samples;
}

float Trainer::score() const noexcept {
  float score = 0;
  for (const auto &game : games_) {
    score += game->score();
  }
  return score / games_.size();
}

// TODO: Test this thoroughly. Test that the average has a reasonable value
float Trainer::getAvgMateLen() const noexcept {
  int32_t total_length = 0;
  for (const auto &game : games_) {
    total_length += game->mateLength();
  }
  return static_cast<float>(total_length) / games_.size();
}

void Trainer::writeRequests(float *game_states,
                            int32_t to_play) const noexcept {
  int32_t offset = 0;
  // Testing mode
  // Only count requests from one player
  if (to_play == 0 || to_play == 1) {
    for (int32_t i = 0; i < games_.size(); ++i) {
      if (games_[i]->to_play() == (to_play + games_[i]->parity()) % 2 &&
          !is_done_[i]) {
        games_[i]->writeRequests(game_states + offset * kGameStateSize);
        offset += games_[i]->numRequests();
      }
    }
    return;
  }
  // Training mode
  for (int32_t i = 0; i < games_.size(); ++i) {
    if (!is_done_[i]) {
      games_[i]->writeRequests(game_states + offset * kGameStateSize);
      offset += games_[i]->numRequests();
    }
  }
}

void Trainer::writeSamples(float *game_states, float *eval_samples,
                           float *prob_samples) const noexcept {
  int32_t offset = 0;
  for (int32_t i = 0; i < games_.size(); ++i) {
    games_[i]->writeSamples(game_states + offset * kGameStateSize,
                            eval_samples + offset,
                            prob_samples + offset * kNumMoves);
    offset += games_[i]->numSamples();
  }
}

/// TODO: Try multiprocessing
float Trainer::writeScores(const std::string &out_file) const {
  float scores[games_.size()];
  for (uintf i = 0; i < games_.size(); i += 2) {
    scores[i] = games_[i]->score();
  }
  for (uintf i = 1; i < games_.size(); i += 2) {
    scores[i] = 1.0 - games_[i]->score();
  }
  // First player score
  int32_t wins = 0;
  int32_t draws = 0;
  for (int32_t i = 0; i < games_.size(); i += 2) {
    if (scores[i] == 1.0) {
      ++wins;
    } else if (scores[i] == 0.5) {
      ++draws;
    }
  }
  std::ofstream outfile = std::ofstream{out_file, std::ofstream::out};
  outfile << "First player wins: " << wins << " / " << games_.size() / 2
          << " = " << static_cast<float>(wins) / (games_.size() / 2)
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
  for (uintf i = 1; i < games_.size(); i += 2) {
    if (scores[i] == 1.0) {
      ++wins;
    } else if (scores[i] == 0.5) {
      ++draws;
    }
  }
  outfile << "Second player wins: " << wins << " / " << games_.size() / 2
          << " = " << static_cast<float>(wins) / (games_.size() / 2)
          << "\nSecond player draws: " << draws << " / " << games_.size() / 2
          << " = " << static_cast<float>(draws) / (games_.size() / 2)
          << "\nSecond player losses: " << games_.size() / 2 - wins - draws
          << " / " << games_.size() / 2 << " = "
          << static_cast<float>(games_.size() / 2 - wins - draws) /
                 (games_.size() / 2)
          << '\n';
}

bool Trainer::doIteration(float evaluations[], float probabilities[]) {

  uintf offsets[games_.size()];
  offsets[0] = 0;
  omp_set_num_threads(threads);
#pragma omp parallel for
  for (uintf i = 1; i < games_.size(); ++i) {
    offsets[i] = games_[i - 1]->numRequests();
  }
  for (uintf i = 1; i < games_.size(); ++i) {
    offsets[i] += offsets[i - 1];
  }

  omp_set_num_threads(threads);
#pragma omp parallel for
  for (uintf i = 0; i < games_.size(); ++i) {
    if (!is_done_[i]) {
      // Avoid division by 0 in the rare case than num_games < num_iterations /
      // 2, which sometimes occurs when testing small runs
      if (i / std::max((uintf)1, (games_.size() / num_iterations)) <
          searches_done_) {
        bool is_completed = games_[i]->doIteration(
            &evaluations[offsets[i]], &probabilities[kNumMoves * offsets[i]]);
        if (is_completed) {
          is_done_[i] = true;
        }
      } else if (i / std::max((uintf)1, (games_.size() / num_iterations)) ==
                 searches_done_) {
        games_[i]->doIteration(nullptr, nullptr);
      }
    }
  }
  ++searches_done_;
  for (uintf i = 0; i < games_.size(); ++i) {
    if (!is_done_[i]) {
      return false;
    }
  }
  return true;
}

bool Trainer::doIteration(float evaluations[], float probabilities[],
                          uintf to_play) {
  if (searches_done_ == 0) {
    omp_set_num_threads(threads);
#pragma omp parallel for
    for (uintf i = 0; i < games_.size(); ++i) {
      games_[i]->doIteration();
    }
    searches_done_ = 1;
    return false;
  }

  uintf offsets[games_.size()];
  offsets[0] = 0;
  omp_set_num_threads(threads);
#pragma omp parallel for
  for (uintf i = 1; i < games_.size(); ++i) {
    if (games_[i - 1]->to_play() == (to_play + games_[i - 1]->parity()) % 2 &&
        !is_done_[i - 1]) {
      offsets[i] = games_[i - 1]->numRequests();
    } else {
      offsets[i] = 0;
    }
  }
  for (uintf i = 1; i < games_.size(); ++i) {
    offsets[i] += offsets[i - 1];
  }

  omp_set_num_threads(threads);
#pragma omp parallel for
  for (uintf i = 0; i < games_.size(); ++i) {
    if (games_[i]->to_play() == (to_play + games_[i]->parity()) % 2 &&
        !is_done_[i]) {
      bool is_completed = games_[i]->doIteration(
          &evaluations[offsets[i]], &probabilities[kNumMoves * offsets[i]]);
      if (is_completed) {
        is_done_[i] = true;
      }
    }
  }
  for (uintf i = 0; i < games_.size(); ++i) {
    if (!is_done_[i]) {
      return false;
    }
  }
  return true;
}

void Trainer::initialize(bool testing, uintf num_games, uintf num_logged,
                         float c_puct, float epsilon, uintf searches_per_eval_,
                         const string &logging_folder) {

  // Remember to pass down hyperparameters to selfplayer and trainmc

  games_.reserve(num_games);

  if (num_games < num_logged)
    num_logged = num_games;

  if (testing) {
    for (uintf i = 0; i < num_logged; ++i) {
      games_.emplace_back(new SelfPlayer{
          generator_(), num_iterations, searches_per_eval_, c_puct, epsilon,
          std::move(std::make_unique<std::ofstream>(
              logging_folder + "/game_" + std::to_string(i) + ".txt",
              std::ofstream::out)),
          i % 2});
    }
    for (uintf i = num_logged; i < num_games; ++i) {
      games_.emplace_back(new SelfPlayer{generator_(), num_iterations,
                                         searches_per_eval_, c_puct, epsilon,
                                         nullptr, i % 2});
    }
  } else {
    for (uintf i = 0; i < num_logged; ++i) {
      games_.emplace_back(new SelfPlayer{
          generator_(), num_iterations, searches_per_eval_, c_puct, epsilon,
          std::move(std::make_unique<std::ofstream>(
              logging_folder + "/game_" + std::to_string(i) + ".txt",
              std::ofstream::out))});
    }
    for (uintf i = num_logged; i < num_games; ++i) {
      games_.emplace_back(new SelfPlayer{generator_(), num_iterations,
                                         searches_per_eval_, c_puct, epsilon,
                                         nullptr});
    }
  }
}