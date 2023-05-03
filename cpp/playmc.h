#ifndef PLAYMC_H
#define PLAYMC_H

#include "node.h"
#include "util.h"
#include <array>
#include <bitset>
#include <random>
#include <vector>

using std::bitset;

class Game;

class PlayMC {

  uintf max_iterations, searches_per_eval;
  float c_puct, epsilon;

  Node *root, *cur;

  // Which index to write to for the searched node
  uintf eval_index;
  // Nodes we have searches this cycle
  std::vector<Node *> searched;
  // We want to evaluate these vectors
  float *to_eval;

  // Used to keep track of when to choose a move
  uintf iterations_done;

  bool logging;

  std::mt19937 *generator;

  void receive_evaluation(float evaluation[], float probabilities[]);
  bool search();

  void move_down(Node *prev_node);

public:
  PlayMC() = default;
  PlayMC(uintf max_iterations, uintf searches_per_eval, float c_puct,
         float epsilon, bool logging, uintf seed);
  // PlayMC for web app
  PlayMC(bool *board, uintf to_play, uintf *pieces, uintf max_iterations,
  uintf searches_per_eval, uintf seed);
  ~PlayMC();

  void do_first_iteration(uintf move_choice);

  bool do_iteration(float evaluation[], float probabilities[]);

  // Choose the next child to visit
  uintf choose_move();

  // Get legal moves for the current state
  void get_legal_moves(bool *legal_moves);
  // Get winning moves for the current state
  void get_winning_moves(bool *winning_moves);

  void receive_opp_move(uintf move_choice);

  uintf write_requests(float game_states[]) const;

  bool is_done() const;
  bool has_won() const;
  bool has_drawn() const;
  void print_game() const;
};

#endif
