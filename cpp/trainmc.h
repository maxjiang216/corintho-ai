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
	int visits, depth;
	float evaluation, noisy_probabilities[96];
	bitset<96> legal_moves;
        vector<shared_ptr<Node>> children;
	Node *parent;
      public:
	Node();
	~Node() = default;
    };

    shared_ptr<Node> root;
    int iterations_done;
    Node *cur_node;

    static int iterations;
    static float c_puct, epsilon;

  public:

    TrainMC();
    ~TrainMC() = default;


};
#endif
