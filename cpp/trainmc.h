#ifndef TRAINMC_H
#define TRAINMC_H

#include "game.h"
#include "util.h"
#include <bitset>
#include <memory>

using std::bitset;
using std::unique_ptr;
using std::shared_ptr;

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
