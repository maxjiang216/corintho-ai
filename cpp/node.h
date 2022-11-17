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
    // We should have something to construct a node on allocated memory (placement new)
    // We should also have something to overwrite an existing node with a new node (probably just = operator)
    // Although, we only need to overwrite the game and perhaps reset some flags, so maybe a separate method would be better
    // Used when writing into a new node
    // Will copy a game, then apply the move
    Node(Game game, uint8 depth, uint32 parent, uint8 move_choice);
    ~Node() = default;

  };

#endif