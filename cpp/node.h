#ifndef NODE_H
#define NODE_H

#include "game.h"
#include "util.h"
#include <bitset>
#include <array>

using std::bitset;

// Node in Monte Carlo Tree
  class Node {

    // Game state
    Game game;
    // visits is number of times this node has been searched
    uint16 visits;
    // depth starts at 0
    uint8 depth;
    // Index of parent node
    uint32 parent;
    float evaluation;
    std::array<float, NUM_MOVES> probabilities;
    // legal_moves denotes which moves are legal, stores in node instead of game since Node is the one using it (?)
    // visited denotes which children have been visited, useful for many computations
    bitset<NUM_MOVES> legal_moves, visited;

  public:
    
    // Used to create the root node
    Node();
    Node(const Game &game);
    // Used when writing into a new node
    // Will copy a game, then apply the move
    Node(const Game &game, uint8 depth, uint32 parent, uint8 move_choice);
    ~Node() = default;

    // overwrite relevant parts of node
    // other things will be overwritten lazily
    void overwrite(const Game &new_game, uint8 new_depth, uint32 new_parent, uint8 move_choice);

    bool is_terminal();

    void write_game_state(float game_state[GAME_STATE_SIZE]);

  };

#endif