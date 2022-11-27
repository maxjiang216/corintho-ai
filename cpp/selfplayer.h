#ifndef SELFPLAYER_H
#define SELFPLAYER_H

#include "trainmc.h"
#include "util.h"
#include <array>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

class Trainer;

struct Sample {
  // We do not store the evaluation as it is only computable once the game is
  // complete
  std::array<float, GAME_STATE_SIZE> game_state;
  std::array<float, NUM_TOTAL_MOVES> probabilities;
};

class SelfPlayer {

  TrainMC players[2];
  // This could be bool, but int is probably faster
  // Also makes it easier for Trainer to assign seeds
  uintf to_play;

  // Game result for first player
  Result result;

  bool logging;

  // seed is only used in testing
  uintf seed;

  std::vector<Sample> samples;

  // This way we don't allocate memory if there is no logging file
  std::ofstream *logging_file;

  Trainer *trainer;

  bool do_iteration(float game_state[GAME_STATE_SIZE]);

public:
  // Training mode
  SelfPlayer(Trainer *trainer);
  SelfPlayer(Trainer *trainer, uintf id, const std::string &logging_folder);
  // Testing mode
  SelfPlayer(uintf seed, Trainer *trainer);
  // This is slightly inefficient, but more general, and only happens at most
  // once per testing run
  SelfPlayer(uintf seed, Trainer *trainer, uintf id,
             const std::string &logging_folder);
  ~SelfPlayer();

  void do_first_iteration(float game_state[GAME_STATE_SIZE]);
  // Training
  bool do_iteration(float evaluation, float probabilities[NUM_TOTAL_MOVES],
                    float dirichlet_noise[NUM_MOVES],
                    float game_state[GAME_STATE_SIZE]);
  // Testing
  bool do_iteration(float evaluation_1,
                    const float probabilities_1[NUM_TOTAL_MOVES],
                    float evaluation_2,
                    const float probabilities_2[NUM_TOTAL_MOVES],
                    const float dirichlet_noise[NUM_MOVES],
                    float game_state[GAME_STATE_SIZE]);

  uintf get_root(uintf player_num) const;

  uintf count_samples() const;

  uintf write_samples(float *game_states, float *evaluation_samples,
                      float *probability_samples) const;

  float get_score() const;
};

#endif