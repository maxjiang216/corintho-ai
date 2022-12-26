#ifndef SELFPLAYER_H
#define SELFPLAYER_H

#include "trainmc.h"
#include "util.h"
#include <array>
#include <fstream>
#include <random>
#include <string>
#include <utility>
#include <vector>

struct Sample {
  // We do not store the evaluation as it is only computable once the game is
  // complete
  std::array<float, GAME_STATE_SIZE> game_state;
  std::array<float, NUM_MOVES> probabilities;

  Sample(std::array<float, GAME_STATE_SIZE> game_state,
         std::array<float, NUM_MOVES> probabilities)
      : game_state{game_state}, probabilities{probabilities} {}
};

class SelfPlayer {

  // Monte Carlo tree for the players
  TrainMC players[2];

  uintf to_play;

  // Random generator for all operations
  std::mt19937 *generator;

  // Training samples
  std::vector<Sample> samples;

  // Game result for first player
  uint8s result;

  // seed is only used in testing
  uintf seed;

  // This way we don't allocate memory if there is no logging file
  std::ofstream *logging_file, *verbose_file;

  bool do_iteration(float game_state[GAME_STATE_SIZE]);

public:
  // Training mode
  SelfPlayer(std::mt19937 *generator);
  SelfPlayer(std::mt19937 *generator, std::ofstream *logging_file,
             std::ofstream *verbose_file);
  // Testing mode
  SelfPlayer(uintf seed, std::mt19937 *generator);
  SelfPlayer(uintf seed, std::mt19937 *generator, std::ofstream *logging_file,
             std::ofstream *verbose_file);
  ~SelfPlayer();

  void do_first_iteration(float game_state[GAME_STATE_SIZE]);
  bool do_iteration(float evaluation[], float probabilities[],
                    float game_state[]);

  uintf count_samples() const;

  uintf write_samples(float *game_states, float *evaluation_samples,
                      float *probability_samples) const;

  float get_score() const;
  uintf count_nodes() const;

  friend class Trainer;
};

#endif