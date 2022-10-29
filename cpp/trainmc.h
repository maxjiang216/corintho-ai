#ifndef TRAINMC_H
#define TRAINMC_H

#include "game.h"
#include <bitset>
#include <memory>

using std::bitset;
using std::unique_ptr;
using std::shared_ptr;

class TrainMC {

    // Node in Monte Carlo Tree
    class Node {
        Game game;
	      // Visits is number of times this node has been searched
	      int visits, depth;
	      float evaluation, probabilities[96];
	      unique_ptr<bitset<96>> legal_moves;
        vector<shared_ptr<Node>> children;
	      Node *parent;
      public:
	      Node();
	      Node(Game);
	      ~Node() = default;
    };

    shared_ptr<Node> root;
    int iterations_done;
    Node *cur_node;

    static int max_iterations = 1600;
    static float c_puct = 1, epsilon = 0.25;

    int choose_next(Node &node);

  public:

    TrainMC();
    ~TrainMC() = default;

    void first_search();
    void search();
    void receive_opp_move(int);

};

#endif
