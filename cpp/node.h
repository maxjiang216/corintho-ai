#ifndef NODE_H
#define NODE_H

#include "game.h"
#include "util.h"
#include <bitset>
#include <iostream>

// Node in Monte Carlo Tree
class alignas(64) Node {

  static constexpr float MAX_PROBABILITY = 511.0;

  struct Edge {
    uint16s move_id : 7, probability : 9;
  };

  // Visited this evaluation cycle
  bool all_visited;
  uint8s result, depth, num_legal_moves, child_num;
  uint16s visits;
  float evaluation, denominator;
  Edge *edges;
  Node *parent, *next_sibling, *first_child;

  // Game state
  Game game;

  void initialize_edges();

public:
  // Used to create root nodes
  Node();
  // Create a new tree root from arbitrary state
  Node(const Game &game, uint8s depth);
  // Common create new node function
  // Will copy a game, then apply the move
  Node(const Game &game, uint8s depth, Node *parent, Node *next_sibling,
       uint8s move_choice);
  ~Node();

  // Returns whether there are lines
  bool get_legal_moves(std::bitset<NUM_MOVES> &legal_moves) const;

  // Accessors
  bool is_terminal() const;
  float get_probability(uintf edge_index) const;

  void write_game_state(float game_state[GAME_STATE_SIZE]) const;

  uintf count_nodes() const;

  void print_main_line(std::ostream *logging_file) const;

  friend class TrainMC;
  friend class SelfPlayer;
};

#endif