#include "trainer.h"
#include "node.h"
#include "trainmc.h"
#include "util.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

using std::cerr;
using std::string;
using std::vector;

Trainer::Trainer(uintf num_games, uintf num_logged, uintf num_iterations,
                 float c_puct, float epsilon, const string &logging_folder,
                 uintf random_seed)
    : num_iterations{std::max(num_iterations, (uintf)2)}, iterations_done{0},
      games_done{0}, is_done{vector<bool>(num_games, false)}, generator{
                                                                  random_seed} {
  initialize(false, num_games, num_logged, c_puct, epsilon, logging_folder);
}

Trainer::Trainer(uintf num_games, uintf num_logged, uintf num_iterations,
                 float c_puct, float epsilon, const string &logging_folder,
                 uintf random_seed, bool)
    : num_iterations{std::max(num_iterations, (uintf)2)}, iterations_done{0},
      games_done{0}, is_done{vector<bool>(num_games, false)}, generator{
                                                                  random_seed} {
  initialize(true, num_games, num_logged, c_puct, epsilon, logging_folder);
}

Trainer::~Trainer() {
  for (uintf i = 0; i < games.size(); ++i) {
    delete games[i];
  }
}

bool Trainer::do_iteration(float evaluations[], float probabilities[],
                           float game_states[]) {

  for (uintf i = 0; i < games.size(); ++i) {
    if (!is_done[i]) {
      // Avoid division by 0 in the rare case than num_games < num_iterations /
      // 2, which sometimes occurs when testing small runs
      if (i / std::max((uintf)1, (num_games / (num_iterations / 2))) <
          iterations_done) {
        bool is_completed = games[i]->do_iteration(
            evaluations[i], &probabilities[i * NUM_MOVES],
            &game_states[i * GAME_STATE_SIZE]);
        if (is_completed) {
          is_done[i] = true;
          ++games_done;
          if (games_done == num_games) {
            return true;
          }
        }
      } else if (i / std::max((uintf)1, (num_games / (num_iterations / 2))) ==
                 iterations_done) {
        games[i]->do_first_iteration(&game_states[i * GAME_STATE_SIZE]);
      }
    }
  }
  ++iterations_done;

  return false;
}

bool Trainer::do_iteration(const float evaluations_1[],
                           const float probabilities_1[],
                           const float evaluations_2[],
                           const float probabilities_2[], float game_states[]) {
  for (uintf i = 0; i < num_games; ++i) {
    if (!is_done[i]) {
      // Avoid division by 0 in the rare case than num_games < num_iterations /
      // 2
      if (i / std::max((uintf)1, (num_games / (num_iterations / 2))) <
          iterations_done) {
        bool is_completed = games[i]->do_iteration(
            evaluations_1[i], &probabilities_1[i * NUM_MOVES], evaluations_2[i],
            &probabilities_2[i * NUM_MOVES], &game_states[i * GAME_STATE_SIZE]);
        if (is_completed) {
          is_done[i] = true;
          ++games_done;
          if (games_done == num_games) {
            return true;
          }
        }
      } else if (i / std::max((uintf)1, (num_games / (num_iterations / 2))) ==
                 iterations_done) {
        games[i]->do_first_iteration(&game_states[i * GAME_STATE_SIZE]);
      }
    }
  }
  ++iterations_done;

  return false;
}

void Trainer::initialize(bool testing, uintf num_games, uintf num_logged,
                         float c_puct, float epsilon,
                         const string &logging_folder) {

  games.reserve(num_games);

  if (num_games < num_logged)
    num_logged = num_games;

  if (testing) {
    for (uintf i = 0; i < num_logged; ++i) {
      games.emplace_back(
          new SelfPlayer{i % 2, new std::mt19937(generator()),
                         new std::ofstream{logging_folder + "/game_" +
                                               std::to_string(i) + ".txt",
                                           std::ofstream::out}});
    }
    for (uintf i = num_logged; i < num_games; ++i) {
      games.emplace_back(new SelfPlayer{i % 2, new std::mt19937(generator())});
    }
  } else {
    for (uintf i = 0; i < num_logged; ++i) {
      games.emplace_back(
          new SelfPlayer{new std::mt19937(generator()),
                         new std::ofstream{logging_folder + "/game_" +
                                               std::to_string(i) + ".txt",
                                           std::ofstream::out}});
    }
    for (uintf i = num_logged; i < num_games; ++i) {
      games.emplace_back(new SelfPlayer{new std::mt19937(generator())});
    }
  }

  // Set TrainMC static variables
  TrainMC::set_statics(num_iterations, c_puct, epsilon);
}

uintf Trainer::count_samples() const {
  uintf counter = 0;
  for (uintf i = 0; i < num_games; ++i) {
    counter += games[i]->count_samples();
  }
  return counter;
}

void Trainer::write_samples(float *game_states, float *evaluation_samples,
                            float *probability_samples) const {
  uintf offset = 0;
  for (uintf i = 0; i < num_games; ++i) {
    uintf num_samples = games[i]->write_samples(
        game_states + offset * GAME_STATE_SIZE, evaluation_samples + offset,
        probability_samples + offset * NUM_MOVES);
    offset += num_samples;
  }
}

float Trainer::get_score() const {
  float score = 0;
  for (uintf i = 0; i < num_games; i += 2) {
    score += games[i]->get_score();
  }
  for (uintf i = 1; i < num_games; i += 2) {
    score += 1.0 - games[i]->get_score();
  }
  return score / (float)num_games;
}

bool Trainer::is_all_done() const { return games_done == num_games; }