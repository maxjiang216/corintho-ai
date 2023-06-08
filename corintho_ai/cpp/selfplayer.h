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

  // Positions to evaluate
  float *to_eval;

  // Training samples
  std::vector<Sample> samples;

  // Game result for first player
  uint8s result;

  // seed is only used in testing
  uintf seed;

  // This way we don't allocate memory if there is no logging file
  std::ofstream *logging_file;

  // Turn at which game is solved
  uintf mate_turn;

  bool do_iteration();

public:
  // Training mode
  SelfPlayer(uintf searches_per_eval, std::mt19937 *generator);
  SelfPlayer(uintf searches_per_eval, std::mt19937 *generator,
             std::ofstream *logging_file);
  // Testing mode
  SelfPlayer(uintf searches_per_eval, uintf seed, std::mt19937 *generator);
  SelfPlayer(uintf searches_per_eval, uintf seed, std::mt19937 *generator,
             std::ofstream *logging_file);
  ~SelfPlayer();

  void do_first_iteration();
  bool do_iteration(float evaluation[], float probabilities[]);

  uintf count_requests() const;
  void write_requests(float *game_states) const;

  uintf count_samples() const;

  void write_samples(float *game_states, float *evaluation_samples,
                     float *probability_samples) const;

  float get_score() const;
  uintf count_nodes() const;

  uintf get_mate_length() const;

  friend class Trainer;
};

#endif