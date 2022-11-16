#ifndef TRAINMC_H
#define TRAINMC_H

#include "game.h"
#include <bitset>
#include <memory>

using std::bitset;
using std::unique_ptr;
using std::shared_ptr;

// Node in Monte Carlo Tree
  class Node {
      Game game;
      // Visits is number of times this node has been searched
      // Depth starts at 0
      unsigned int visits, depth, parent;
      float evaluation, probabilities[NUM_LEGAL_MOVES];
      bitset<NUM_LEGAL_MOVES> legal_moves, visited;
      // Tracks if we still need to keep this node, lazy deletions
      // Could we move this (or other attributes) into trainer
      // For bools, using bitset would save some space (assuming ~50% filled)
      // also, this variable is only used to manage the tree
      // It is toggled when the tree moves down, though
      // But trainer could handle this, just a BFS excluding one of the children
      // bool could be faster than bitset, we can do empirical testing afterwards
      bool is_stale;

    public:
      Node();
      Node(Game);
      ~Node() = default;
  };

class TrainMC {

  public:

    unsigned int root, cur_node, iterations_done;
    bool testing, logging;

    static int max_iterations = 1600;
    static float c_puct = 1, epsilon = 0.25;

    void first_search();
    int choose_next();
    void search(float, float &noisy_probabilities[])

  public:

    TrainMC(bool testing = false);
    ~TrainMC() = default;
    
    void receive_opp_move(int);
    int choose_move()

};

#endif
