#include "node.h"
#include "util.h"

#include <bitset>

using std::bitset;

// Only used to create the root node (starting position)
// Depending on how get_legal_moves works, we might not need to set legal_moves to all 0s
Node::Node(): game{Game()}, visits{0}, depth{0}, evaluation{0},
legal_moves{bitset<NUM_MOVES>().set()} {

    // Get legal moves in starting position. Cannot be terminal node
    game.get_legal_moves(*legal_moves);

}

Node::Node(Game game, uint8 depth, uint32 parent, uint8 move_choice): game{game}, visits{1}, depth{depth+1},
parent{parent} {
    game.do_move(move_choice);
}
