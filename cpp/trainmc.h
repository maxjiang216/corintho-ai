#ifndef TRAINMC_H
#define TRAINMC_H

#include "game.h"
#include "trainer.h"
#include "util.h"
#include <bitset>
#include <memory>

using std::bitset;
using std::unique_ptr;
using std::shared_ptr;

class TrainMC {

    Trainer *trainer;
    uint32 root, cur_node;
    // Used to keep track of when to choose a move
    uint16 iterations_done;
    bool testing, logging;

    static uint16 max_iterations = 1600;
    static float c_puct = 1.0, epsilon = 0.25;

    int choose_next();
    void search(float, float &noisy_probabilities[])

  public:

    // Training
    TrainMC(Trainer *trainer);
    TrainMC(Trainer *trainer, bool);
    // Testing
    TrainMC(Trainer *trainer, bool logging, bool);
    ~TrainMC() = default;
    
    void receive_opp_move(int);
    int do_iterations()
    void do_first_iteration();

};

#endif
