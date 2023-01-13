#include "node.h"
#include "game.h"
#include "move.h"
#include "util.h"
#include <bitset>
#include <iostream>
using std::bitset;
using std::cerr;

// Starting position is never a terminal node
Node::Node()
    : all_visited{true}, result{RESULT_NONE}, depth{0},
      num_legal_moves{0}, child_num{0}, visits{1}, evaluation{0.0},
      denominator{0.0}, edges{nullptr}, parent{nullptr}, next_sibling{nullptr},
      first_child{nullptr}, game{Game()} {
  initialize_edges();
}

// We use this when opponent makes a move we did not anticipate
// We only receive this move if it is not a terminal position
// Or else the game would have ended
Node::Node(const Game &game, uint8s depth)
    : all_visited{true}, result{RESULT_NONE}, depth{depth},
      num_legal_moves{0}, child_num{0}, visits{1}, evaluation{0.0},
      denominator{0.0}, edges{nullptr}, parent{nullptr}, next_sibling{nullptr},
      first_child{nullptr}, game{game} {
  initialize_edges();
}

Node::Node(const Game &other_game, uint8s depth, Node *parent,
           Node *next_sibling, uint8s move_choice)
    : all_visited{true}, result{RESULT_NONE}, depth{depth},
      num_legal_moves{0}, child_num{move_choice}, visits{1}, evaluation{0.0},
      denominator{0.0}, edges{nullptr}, parent{parent},
      next_sibling{next_sibling}, first_child{nullptr}, game{other_game} {
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

bool Node::is_terminal() const {
  return result == RESULT_LOSS || result == RESULT_WIN || result == RESULT_DRAW;
}

bool Node::is_known() const { return result != RESULT_NONE; }

float Node::get_probability(uintf edge_index) const {
  return (float)edges[edge_index].probability * denominator;
}

void Node::write_game_state(float game_state[GAME_STATE_SIZE]) const {
  game.write_game_state(game_state);
}

void Node::initialize_edges() {
  bitset<NUM_MOVES> legal_moves;
  bool is_lines = game.get_legal_moves(legal_moves);
  num_legal_moves = legal_moves.count();
  // Terminal node
  if (num_legal_moves == 0) {
    // Otherwise we overcount by 1
    visits = 0;
    // Decisive game
    if (is_lines) {
      // Current player loss
      result = RESULT_LOSS;
    } else {
      result = RESULT_DRAW;
    }
  }
  // Otherwise, allocate some edges
  else {
    edges = new Edge[num_legal_moves];
    uintf edge_index = 0;
    for (uintf i = 0; i < NUM_MOVES; ++i) {
      if (legal_moves[i]) {
        edges[edge_index] = Edge(i, 0);
        ++edge_index;
      }
    }
  }
}

uintf Node::count_nodes() const {
  uintf counter = 1;
  Node *cur_child = first_child;
  while (cur_child != nullptr) {
    counter += cur_child->count_nodes();
    cur_child = cur_child->next_sibling;
  }
  return counter;
}

void Node::print_main_line(std::ostream *logging_file) const {
  Node *cur_child = first_child, *best_child = nullptr;
  uintf max_visits = 0, edge_index = 0;
  float max_eval = 0.0, prob = 0.0;
  while (cur_child != nullptr) {
    if (edges[edge_index].move_id == cur_child->child_num) {
      if (cur_child->visits > max_visits ||
          (cur_child->visits == max_visits &&
           cur_child->evaluation > max_eval)) {
        best_child = cur_child;
        max_visits = cur_child->visits;
        max_eval = cur_child->evaluation;
        prob = get_probability(edge_index);
      }
      cur_child = cur_child->next_sibling;
    }
    ++edge_index;
  }
  if (best_child != nullptr) {
    *logging_file << (uintf)best_child->depth << ". "
                  << Move{best_child->child_num} << " v: " << max_visits
                  << " e: " << max_eval << " p: " << prob << '\t';
    best_child->print_main_line(logging_file);
  }
}