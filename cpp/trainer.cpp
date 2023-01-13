#include "trainer.h"
#include "node.h"
#include "trainmc.h"
#include "util.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <queue>
#include <string>
#include <vector>

using std::cerr;
using std::string;
using std::vector;

Trainer::Trainer(uintf num_games, uintf num_logged, uintf num_iterations,
                 float c_puct, float epsilon, uintf threads,
                 uintf searches_per_eval, const string &logging_folder,
                 uintf random_seed)
    : num_iterations{std::max(num_iterations, (uintf)2)}, threads{threads},
      searches_per_eval{searches_per_eval}, iterations_done{0},
      is_done{vector<bool>(num_games, false)}, generator{random_seed} {
  initialize(false, num_games, num_logged, c_puct, epsilon, searches_per_eval,
             logging_folder);
}

Trainer::Trainer(uintf num_games, uintf num_logged, uintf num_iterations,
                 float c_puct, float epsilon, uintf threads,
                 uintf searches_per_eval, const string &logging_folder,
                 uintf random_seed, bool)
    : num_iterations{std::max(num_iterations, (uintf)2)}, threads{threads},
      searches_per_eval{searches_per_eval}, iterations_done{0},
      is_done{vector<bool>(num_games, false)}, generator{random_seed} {
  initialize(true, num_games, num_logged, c_puct, epsilon, searches_per_eval,
             logging_folder);
}

Trainer::~Trainer() {
  for (uintf i = 0; i < games.size(); ++i) {
    delete games[i];
  }
}

bool Trainer::do_iteration(float evaluations[], float probabilities[]) {

  uintf offsets[games.size()];
  offsets[0] = 0;
  omp_set_num_threads(threads);
#pragma omp parallel for
  for (uintf i = 1; i < games.size(); ++i) {
    offsets[i] = games[i - 1]->count_requests();
  }
  for (uintf i = 1; i < games.size(); ++i) {
    offsets[i] += offsets[i - 1];
  }

  omp_set_num_threads(threads);
#pragma omp parallel for
  for (uintf i = 0; i < games.size(); ++i) {
    if (!is_done[i]) {
      // Avoid division by 0 in the rare case than num_games < num_iterations /
      // 2, which sometimes occurs when testing small runs
      if (i / std::max((uintf)1, (games.size() / num_iterations)) <
          iterations_done) {
        bool is_completed = games[i]->do_iteration(
            &evaluations[offsets[i]], &probabilities[NUM_MOVES * offsets[i]]);
        if (is_completed) {
          is_done[i] = true;
        }
      } else if (i / std::max((uintf)1, (games.size() / num_iterations)) ==
                 iterations_done) {
        games[i]->do_first_iteration();
      }
    }
  }
  ++iterations_done;
  for (uintf i = 0; i < games.size(); ++i) {
    if (!is_done[i]) {
      return false;
    }
  }
  return true;
}

bool Trainer::do_iteration(float evaluations[], float probabilities[],
                           uintf to_play) {
  if (iterations_done == 0) {
    omp_set_num_threads(threads);
#pragma omp parallel for
    for (uintf i = 0; i < games.size(); ++i) {
      games[i]->do_first_iteration();
    }
    iterations_done = 1;
    return false;
  }

  uintf offsets[games.size()];
  offsets[0] = 0;
  omp_set_num_threads(threads);
#pragma omp parallel for
  for (uintf i = 1; i < games.size(); ++i) {
    if (games[i - 1]->to_play == (to_play + games[i - 1]->seed) % 2 &&
        !is_done[i - 1]) {
      offsets[i] = games[i - 1]->count_requests();
    } else {
      offsets[i] = 0;
    }
  }
  for (uintf i = 1; i < games.size(); ++i) {
    offsets[i] += offsets[i - 1];
  }

  omp_set_num_threads(threads);
#pragma omp parallel for
  for (uintf i = 0; i < games.size(); ++i) {
    if (games[i]->to_play == (to_play + games[i]->seed) % 2 && !is_done[i]) {
      bool is_completed = games[i]->do_iteration(
          &evaluations[offsets[i]], &probabilities[NUM_MOVES * offsets[i]]);
      if (is_completed) {
        is_done[i] = true;
      }
    }
  }
  for (uintf i = 0; i < games.size(); ++i) {
    if (!is_done[i]) {
      return false;
    }
  }
  return true;
}

void Trainer::initialize(bool testing, uintf num_games, uintf num_logged,
                         float c_puct, float epsilon, uintf searches_per_eval,
                         const string &logging_folder) {

  // Set TrainMC static variables
  TrainMC::set_statics(num_iterations, c_puct, epsilon, searches_per_eval);

  games.reserve(num_games);

  if (num_games < num_logged)
    num_logged = num_games;

  if (testing) {
    for (uintf i = 0; i < num_logged; ++i) {
      games.emplace_back(new SelfPlayer{
          searches_per_eval, i % 2, new std::mt19937(generator()),
          new std::ofstream{logging_folder + "/game_" + std::to_string(i) +
                                ".txt",
                            std::ofstream::out}});
    }
    for (uintf i = num_logged; i < num_games; ++i) {
      games.emplace_back(new SelfPlayer{searches_per_eval, i % 2,
                                        new std::mt19937(generator())});
    }
  } else {
    for (uintf i = 0; i < num_logged; ++i) {
      games.emplace_back(
          new SelfPlayer{searches_per_eval, new std::mt19937(generator()),
                         new std::ofstream{logging_folder + "/game_" +
                                               std::to_string(i) + ".txt",
                                           std::ofstream::out}});
    }
    for (uintf i = num_logged; i < num_games; ++i) {
      games.emplace_back(
          new SelfPlayer{searches_per_eval, new std::mt19937(generator())});
    }
  }
}

uintf Trainer::count_nodes() const {
  uintf counts[games.size()];
#pragma omp parallel for
  for (uintf i = 0; i < games.size(); ++i) {
    counts[i] = games[i]->count_nodes();
  }
  uintf counter = 0;
  for (uintf i = 0; i < games.size(); ++i) {
    counter += counts[i];
  }
  return counter;
}

uintf Trainer::write_requests(float *game_states) const {
  uintf offsets[games.size()];
  offsets[0] = 0;
  omp_set_num_threads(threads);
#pragma omp parallel for
  for (uintf i = 1; i < games.size(); ++i) {
    offsets[i] = games[i - 1]->count_requests();
  }
  for (uintf i = 1; i < games.size(); ++i) {
    offsets[i] += offsets[i - 1];
  }
  omp_set_num_threads(threads);
#pragma omp parallel for
  for (uintf i = 0; i < games.size(); ++i) {
    games[i]->write_requests(game_states + offsets[i] * GAME_STATE_SIZE);
  }
  return offsets[games.size() - 1] + games[games.size() - 1]->count_requests();
}

uintf Trainer::write_requests(float *game_states, uintf to_play) const {
  uintf offsets[games.size()];
  offsets[0] = 0;
  omp_set_num_threads(threads);
#pragma omp parallel for
  for (uintf i = 1; i < games.size(); ++i) {
    if (games[i - 1]->to_play == (to_play + games[i - 1]->seed) % 2 &&
        !is_done[i - 1]) {
      offsets[i] = games[i - 1]->count_requests();
    } else {
      offsets[i] = 0;
    }
  }
  for (uintf i = 1; i < games.size(); ++i) {
    offsets[i] += offsets[i - 1];
  }
  omp_set_num_threads(threads);
#pragma omp parallel for
  for (uintf i = 0; i < games.size(); ++i) {
    if (games[i]->to_play == (to_play + games[i]->seed) % 2 && !is_done[i]) {
      games[i]->write_requests(game_states + offsets[i] * GAME_STATE_SIZE);
    }
  }
  uintf num_requests = offsets[games.size() - 1];
  if (games[games.size() - 1]->to_play ==
          (to_play + games[games.size() - 1]->seed) % 2 &&
      !is_done[games.size() - 1]) {
    num_requests += games[games.size() - 1]->count_requests();
  }
  return num_requests;
}

uintf Trainer::count_samples() const {
  uintf counts[games.size()];
  omp_set_num_threads(threads);
#pragma omp parallel for
  for (uintf i = 0; i < games.size(); ++i) {
    counts[i] = games[i]->count_samples();
  }
  uintf counter = 0;
  for (uintf i = 0; i < games.size(); ++i) {
    counter += counts[i];
  }
  return counter;
}

void Trainer::write_samples(float *game_states, float *evaluation_samples,
                            float *probability_samples) const {
  uintf offsets[games.size()];
  offsets[0] = 0;
  omp_set_num_threads(threads);
#pragma omp parallel for
  for (uintf i = 1; i < games.size(); ++i) {
    offsets[i] = games[i - 1]->count_samples();
  }
  for (uintf i = 1; i < games.size(); ++i) {
    offsets[i] += offsets[i - 1];
  }
  for (uintf i = 0; i < games.size(); ++i) {
    games[i]->write_samples(
        game_states + offsets[i] * GAME_STATE_SIZE * SYMMETRY_NUM,
        evaluation_samples + offsets[i] * SYMMETRY_NUM,
        probability_samples + offsets[i] * NUM_MOVES * SYMMETRY_NUM);
  }
}

float Trainer::get_score(const std::string &out_file) const {
  float scores[games.size()];
  omp_set_num_threads(threads);
#pragma omp parallel for
  for (uintf i = 0; i < games.size(); i += 2) {
    scores[i] = games[i]->get_score();
  }
#pragma omp parallel for
  for (uintf i = 1; i < games.size(); i += 2) {
    scores[i] = 1.0 - games[i]->get_score();
  }
  float score = 0;
  uintf wins = 0, draws = 0;
  auto outfile = new std::ofstream{out_file, std::ofstream::out};
  for (uintf i = 0; i < games.size(); i += 2) {
    score += scores[i];
    if (scores[i] == 1.0) {
      ++wins;
    } else if (scores[i] == 0.5) {
      ++draws;
    }
  }
  *outfile << "First player wins: " << wins << " / " << games.size() / 2
           << " = " << (float)wins / (games.size() / 2)
           << "\nFirst player draws: " << draws << " / " << games.size() / 2
           << " = " << (float)draws / (games.size() / 2)
           << "\nFirst player losses: " << games.size() / 2 - wins - draws
           << " / " << games.size() / 2 << " = "
           << (float)(games.size() / 2 - wins - draws) / (games.size() / 2)
           << '\n';
  wins = 0;
  draws = 0;
  for (uintf i = 1; i < games.size(); i += 2) {
    score += scores[i];
    if (scores[i] == 1.0) {
      ++wins;
    } else if (scores[i] == 0.5) {
      ++draws;
    }
  }
  *outfile << "Second player wins: " << wins << " / " << games.size() / 2
           << " = " << (float)wins / (games.size() / 2)
           << "\nSecond player draws: " << draws << " / " << games.size() / 2
           << " = " << (float)draws / (games.size() / 2)
           << "\nSecond player losses: " << games.size() / 2 - wins - draws
           << " / " << games.size() / 2 << " = "
           << (float)(games.size() / 2 - wins - draws) / (games.size() / 2)
           << '\n';
  delete outfile;
  return score / (float)games.size();
}

bool Trainer::is_all_done() const {
  for (uintf i = 0; i < games.size(); ++i) {
    if (!is_done[i]) {
      std::cerr << i << ' ' << games[i]->players[games[i]->to_play].root->game << '\n';
      return false;
    }
  }
  return true;
}