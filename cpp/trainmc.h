#ifndef TRAINMC_H
#define TRAINMC_H

#include "node.h"
#include "util.h"
#include <array>
#include <bitset>
#include <random>

using std::bitset;

class Game;

class TrainMC {

  // I think we have to initialize these
  inline static uintf max_iterations = 1600;
  inline static float c_puct = 1.0, epsilon = 0.25;
  // Number of moves to use weighted random
  const uintf NUM_OPENING_MOVES = 6;

  Node *root, *cur;

  // Used to keep track of when to choose a move
  uintf iterations_done;

  bool testing, logging;

  std::mt19937 *generator;

  void receive_evaluation(float evaluation,
                          const float probabilities[NUM_MOVES]);
  bool search(float game_state[GAME_STATE_SIZE]);

public:
  // Training
  TrainMC(std::mt19937 *generator);
  // Testing
  TrainMC(std::mt19937 *generator, bool);

  TrainMC(TrainMC &&) = default;
  ~TrainMC() = default;

  // First iterations are guaranteed not to end a turn
  // First iteration on starting position
  void do_first_iteration(float game_state[GAME_STATE_SIZE]);

  bool do_iteration(float evaluation_result, float probabilities[NUM_MOVES],
                    float game_state[GAME_STATE_SIZE]);

  // Choose the next child to visit
  uintf choose_move(float game_state[GAME_STATE_SIZE],
                    float probability_sample[NUM_MOVES]);

  bool receive_opp_move(uintf move_choice, float game_state[GAME_STATE_SIZE],
                        const Game &game, uintf depth);

  // Accessors
  const Game &get_game() const;
  bool is_uninitialized() const;

  static void set_statics(uintf new_max_iterations, float new_c_puct,
                          float new_epsilon);

  friend class SelfPlayer;
};

#endif
