#ifndef NODE_H
#define NODE_H

#include "game.h"
#include "util.h"
#include <bitset>

// Node in Monte Carlo Tree
class Node {

  const float MAX_PROBABILITY = 511.0;

  struct Edge {
    uint8s move_id:7, probability:9;
  };

  // Game state
  Game game;
  
  float evaluation;
  uint16s visits;
  uint8s depth;
  Result result;

  uint8s num_legal_moves;
  Edge* edges;
  float denominator;

  Node *parent, *first_child, *next_sibling;

public:
  // Used to create root nodes
  Node();
  // Create a new tree root from arbitrary state
  Node(const Game &game, uint8s depth);
  // Common create new node function
  // Will copy a game, then apply the move
  Node(const Game &game, uint8s depth, Node *parent, Node *next_sibling, uintf move_choice);
  ~Node() = default;

  // Returns whether there are lines
  bool get_legal_moves(std::bitset<NUM_MOVES> &legal_moves) const;

  // Accessors
  bool is_terminal() const;
  float get_probability(uintf move_choice) const;

  void write_game_state(float game_state[GAME_STATE_SIZE]) const;

  friend class TrainMC
};

#endif