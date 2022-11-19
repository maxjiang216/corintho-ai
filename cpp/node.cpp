#include "node.h"
#include "util.h"
#include "game.h"

#include <bitset>

using std::bitset;

// Only used to create the root node (starting position)
Node::Node(): game{Game()}, visits{1}, depth{0} {

    // Get legal moves in starting position. Cannot be terminal node
    game.get_legal_moves(legal_moves);
    visited.reset();

}

// Occasionally need to create root nodes from arbitrary game states
Node::Node(const Game &other_game, uint depth): game{other_game}, visits{1}, depth{depth} {
    game.get_legal_moves(legal_moves);
    visited.reset();
}

// Pass game by reference, then copy it
// This should be more efficient as only a pointer is passed as an argument
// Which is smaller
Node::Node(const Game &other_game, uint depth, uint parent, uint move_choice): game{other_game}, visits{1},
depth{depth+1}, parent{parent} {
    game.do_move(move_choice);
    game.get_legal_moves(legal_moves);
    visited.reset();
}

void Node::overwrite(const Game &new_game, uint new_depth, uint new_parent, uint move_choice) {
    game = game;
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

void write_game_state(float game_state[GAME_STATE_SIZE]) {
    game.write_game_state(game_state);
}