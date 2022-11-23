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

// Check for terminal state
bool Node::is_terminal() {
    return game.is_terminal();
}

// Returns depth of node
uintf Node::get_depth() {
    return depth;
}

// return a const ref to game
const Game& Node::get_game() {
    return game;
}

Result Node::get_result() {
    return game.get_result();
}

// returns whether child is visited
bool Node::has_visited(uintf move_choice) {
    return visited[move_choice];
}

uintf Node::get_visits() {
    return visits;
}

void Node::add_evaluation(float new_evaluation) {
    evaluation += new_evaluation;
}

float Node::get_evaluation() {
    return evaluation;
}

uintf Node::get_parent() {
    return parent;
}

void Node::null_parent() {
    parent = 0;
}

void Node::increment_visits() {
    ++visits;
}

uintf Node::get_to_play() {
    return game.get_to_play();
}

void Node::set_probability(uintf id, float probability) {
    probabilities[id] = probability;
}

float Node::get_probability(uintf id) {
    return probabilities[id];
}

bool Node::is_legal(uintf id) {
    return legal_moves[id];
}

void Node::adjust_probability(uintf id, float scalar, float noise) {
    probabilities[id] = probabilities[id] * scalar + noise;
}

void Node::write_game_state(float game_state[GAME_STATE_SIZE]) {
    game.write_game_state(game_state);
}

void Node::set_visit(uintf move_choice) {
    visited.set(move_choice);
}