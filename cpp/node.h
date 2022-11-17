#ifndef NODE_H
#define NODE_H

#include "game.h"
#include "util.h"
#include <bitset>

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
    float evaluation, probabilities[NUM_MOVES];
    // legal_moves denotes which moves are legal
    // visited denotes which children have been visited, useful for many computations
    bitset<NUM_MOVES> legal_moves, visited;

    // Tracks if we still need to keep this node, lazy deletions
    // Could we move this (or other attributes) into trainer
    // For bools, using bitset would save some space (assuming ~50% filled)
    // also, this variable is only used to manage the tree
    // It is toggled when the tree moves down, though
    // But trainer could handle this, just a BFS excluding one of the children
    // bool could be faster than bitset, we can do empirical testing afterwards
    bool is_stale;

  public:
    
    // Used to create the root node
    Node();
    // Used when writing into a new node
    // Will copy a game, then apply the move
    Node(Game game, uint8 depth, uint32 parent, uint8 move_choice);
    ~Node() = default;

  };

#endif