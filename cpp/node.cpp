#include "node.h"
#include "util.h"

#include <bitset>

using std::bitset;

// Only used to create the root node (starting position)
Node::Node(): game{Game()}, visits{1}, depth{0}, evaluation{0} {

    // Get legal moves in starting position. Cannot be terminal node
    game.get_legal_moves(*legal_moves);
    visited.reset();

}

// Pass game by reference, then copy it
// This should be more efficient as only a pointer is passed as an argument
// Which is smallers
Node::Node(const Game &game, uint8 depth, uint32 parent, uint8 move_choice): game{game}, visits{1},
depth{depth+1}, parent{parent}, visited{bitset<NUM_MOVES>().reset()} {
    game.do_move(move_choice);
    visited.reset();
}

Node::overwrite(const Game &new_game, uint8 new_depth, uint32 new_parent, uint8 move_choice) {
    game = game;
    visits = 1;
    depth = new_depth + 1;
    parent = new_parent;
    game.do_move(move_choice);
    visited.reset();
}