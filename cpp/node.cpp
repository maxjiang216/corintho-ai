#include "node.h"
#include "game.h"
#include "util.h"
#include <bitset>
#include <iostream>
using std::bitset;
using std::cerr;

Node::Node(uintf seed)
    : game{Game()}, visits{1}, depth{0}, parent{0}, seed{seed},
      visited{bitset<NUM_MOVES>()} {}

Node::Node(uintf seed, const Game &other_game, uintf depth)
    : game{other_game}, visits{1}, depth{depth}, parent{0}, seed{seed},
      visited{bitset<NUM_MOVES>()} {}

Node::Node(uintf seed, const Game &other_game, uintf depth, uintf parent,
           uintf move_choice)
    : game{other_game}, visits{1}, depth{depth + 1}, parent{parent}, seed{seed},
      visited{bitset<NUM_MOVES>()} {
  game.do_move(move_choice);
}

void Node::overwrite(uintf new_seed) {
  game = Game();
  visits = 1;
  depth = 0;
  seed = new_seed;
  visited.reset();
}

void Node::overwrite(uintf new_seed, const Game &new_game, uintf new_depth) {
  game = new_game;
  visits = 1;
  depth = new_depth;
  seed = new_seed;
  visited.reset();
}

void Node::overwrite(uintf new_seed, const Game &new_game, uintf new_depth,
                     uintf new_parent, uintf move_choice) {
  game = new_game;
  visits = 1;
  depth = new_depth + 1;
  parent = new_parent;
  seed = new_seed;
  game.do_move(move_choice);
  visited.reset();
}

const Game &Node::get_game() const { return game; }

uintf Node::get_to_play() const { return game.get_to_play(); }

bool Node::is_legal(uintf id) const { return game.is_legal(id); }

Result Node::get_result() const { return game.get_result(); }

bool Node::is_terminal() const { return game.is_terminal(); }

uintf Node::get_visits() const { return visits; }

uintf Node::get_depth() const { return depth; }

uintf Node::get_parent() const { return parent; }

uintf Node::get_seed() const { return seed; }

float Node::get_evaluation() const { return evaluation; }

bool Node::has_visited(uintf move_choice) const { return visited[move_choice]; }

float Node::get_probability(uintf move_choice) const {
  return (float)probabilities[move_choice] * (1.0 / 127.0);
}

void Node::increment_visits() { ++visits; }

void Node::null_parent() { parent = 0; }

void Node::add_evaluation(float new_evaluation) {
  evaluation += new_evaluation;
}

void Node::set_probability(uintf move_choice, unsigned char probability) {
  probabilities[move_choice] = probability;
}

void Node::set_visit(uintf move_choice) { visited.set(move_choice); }

void Node::write_game_state(float game_state[GAME_STATE_SIZE]) const {
  game.write_game_state(game_state);
}

void Node::write_game_state(
    std::array<float, GAME_STATE_SIZE> &game_state) const {
  game.write_game_state(game_state);
}
