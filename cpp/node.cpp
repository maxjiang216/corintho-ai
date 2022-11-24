#include "node.h"
#include "game.h"
#include "util.h"
#include <bitset>

using std::bitset;

Node::Node(): game{Game()}, visits{1}, depth{0}, parent{0}, visited{bitset<NUM_MOVES>()} {}

Node::Node(const Game &other_game, uintf depth): game{other_game}, visits{1}, depth{depth},
                                                 parent{0}, visited{bitset<NUM_MOVES>()} {}

Node::Node(const Game &other_game, uintf depth, uintf parent, uintf move_choice, uintf pos):
           game{other_game}, visits{1}, depth{depth+1}, parent{parent}, visited{bitset<NUM_MOVES>()} {
    game.do_move(move_choice);
}

void Node::overwrite() {
    game = Game();
    visits = 1;
    depth = 0;
    parent = 0;
    visited.reset();
}

void Node::overwrite(const Game &new_game, uintf new_depth) {
    game = new_game;
    visits = 1;
    depth = new_depth;
    parent = 0;
    visited.reset();
}

void Node::overwrite(const Game &new_game, uintf new_depth, uintf new_parent,
                     uintf move_choice, uintf pos, bool is_stale) {
    game = new_game;
    visits = 1;
    depth = new_depth + 1;
    parent = new_parent;
    game.do_move(move_choice);
    visited.reset();
}

const Game& Node::get_game() const {
    return game;
}

uintf Node::get_to_play() const {
    return game.get_to_play();
}

bool Node::is_legal(uintf id) const {
    return game.is_legal(id);
}

Result Node::get_result() const {
    return game.get_result();
}

bool Node::is_terminal() const {
    return game.is_terminal();
}

uintf Node::get_visits() const {
    return visits;
}

uintf Node::get_depth() const {
    return depth;
}

uintf Node::get_parent() const {
    return parent;
}

float Node::get_evaluation() const {
    return evaluation;
}

bool Node::has_visited(uintf move_choice) const {
    return visited[move_choice];
}

float Node::get_probability(uintf move_choice) const {
    return probabilities[move_choice];
}

void Node::increment_visits() {
    ++visits;
}

void Node::null_parent() {
    parent = 0;
}

void Node::add_evaluation(float new_evaluation) {
    evaluation += new_evaluation;
}

void Node::set_probability(uintf move_choice, float probability) {
    probabilities[move_choice] = probability;
}

void Node::adjust_probability(uintf move_choice, float scalar, float noise) {
    probabilities[move_choice] = probabilities[move_choice] * scalar + noise;
}

void Node::set_visit(uintf move_choice) {
    visited.set(move_choice);
}

void Node::write_game_state(float game_state[GAME_STATE_SIZE]) const {
    game.write_game_state(game_state);
}

