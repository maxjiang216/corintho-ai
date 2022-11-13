#ifndef TRAINMC_H
#define TRAINMC_H

#include "game.h"
#include <bitset>
#include <memory>

using std::bitset;
using std::unique_ptr;
using std::shared_ptr;

class TrainMC {

  public:

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
    bool testing;

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
