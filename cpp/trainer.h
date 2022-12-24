#ifndef TRAINER_H
#define TRAINER_H

#include "selfplayer.h"
#include "util.h"
#include <queue>
#include <random>
#include <string>
#include <vector>

// This is the class that should interact with Cython

class Trainer {

  // SelfPlayer objects
  std::vector<SelfPlayer *> games;

  // Number of iterations per move (used to compute offsets)
  uintf num_iterations, threads, searches_per_eval;
  // Counter used to keep track of number of iterations done for offsets
  uintf iterations_done;

  // Track which games are done
  std::vector<bool> is_done;

  // Random generator for all operations
  std::mt19937 generator;

  // Initialize SelfPlayers (factored out of different version of constructor)
  void initialize(bool testing, uintf num_games, uintf num_logged, float c_puct,
                  float epsilon, uintf searches_per_eval,
                  const std::string &logging_folder);

public:
  // Training
  Trainer() = default;
  Trainer(uintf num_games, uintf num_logged, uintf num_iterations, float c_puct,
          float epsilon, uintf threads, uintf searches_per_eval,
          const std::string &logging_folder, uintf random_seed);
  // Testing
  Trainer(uintf num_games, uintf num_logged, uintf num_iterations, float c_puct,
          float epsilon, uintf threads, uintf searches_per_eval,
          const std::string &logging_folder, uintf random_seed, bool);
  ~Trainer();

  // Main function that will be called by Cython
  // Training version
  bool do_iteration(float evaluations[], float probabilities[],
                    float game_states[]);
  // Testing version
  bool do_iteration(float evaluations[], float probabilities[],
                    float game_states[], uintf to_play);

  // Counts the number of samples in all the games
  uintf count_samples() const;

  void write_samples(float *game_states, float *evaluation_samples,
                     float *probability_samples) const;

  float get_score() const;

  uintf count_nodes() const;
};

#endif