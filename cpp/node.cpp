#include "node.h"
#include "game.h"
#include "util.h"
#include <bitset>
#include <iostream>
using std::bitset;
using std::cerr;

// Starting position is never a terminal node
Node::Node()
    : game{Game()}, visits{0}, depth{0}, result{RESULT_NONE}, edges{nullptr},
      parent{nullptr} {
  initialize_edges();
}

// We use this when opponent makes a move we did not anticipate
// We only receive this move if it is not a terminal position
// Or else the game would have ended
Node::Node(const Game &game, uint8s depth)
    : game{game}, visits{0}, depth{depth}, result{RESULT_NONE}, edges{nullptr},
      parent{nullptr} {
  initialize_edges();
}

Node::Node(const Game &other_game, uint8s depth, Node *parent,
           Node *next_sibling, uint8s move_choice)
    : game{other_game}, visits{0}, depth{depth}, parent{parent},
      next_sibling{next_sibling}, child_num{move_choice} {
  game.do_move(move_choice);
  initialize_edges();
}

Node::~Node() {
  delete[] edges;
  delete next_sibling;
  delete first_child;
}

bool Node::get_legal_moves(std::bitset<NUM_MOVES> &legal_moves) const {
  return game.get_legal_moves(legal_moves);
}

bool Node::is_terminal() const { return result != RESULT_NONE; }

float Node::get_probability(uintf edge_index) const {
  return (float)edges[edge_index].probability * denominator;
}

void Node::write_game_state(float game_state[GAME_STATE_SIZE]) const {
  game.write_game_state(game_state);
}

void Node::initialize_edges() {
  bitset<NUM_MOVES> legal_moves;
  bool is_lines = game.get_legal_moves(legal_moves);
  uintf num_legal_moves = legal_moves.count();
  // Terminal node
  if (num_legal_moves == 0) {
    // Decisive game
    if (is_lines) {
      // Second player win
      if (game.to_play == 0) {
        result = RESULT_LOSS;
      } else {
        result = RESULT_WIN;
      }
    } else {
      result = RESULT_DRAW;
    }
  }
  // Otherwise, allocate some edges
  else {
    Edge *edges = (Edge *)(new char[num_legal_moves * sizeof(Edge)]);
    for (uintf i = 0; i < NUM_MOVES; ++i) {
      if (legal_moves[i]) {
        new (edges + i) Edge(i, 0);
      }
    }
  }
}