#ifndef SELFPLAYER_H
#define SELFPLAYER_H

#include "trainer.h"
#include "trainmc.h"
#include "util.h"
#include <vector>
#include <array>

using std::vector;
using std::array;

class SelfPlayer {

    bool testing, logging;
    // Seed is only used in testing
    uint8 seed;
    TrainMC players[2];

  public:
    
    // Training mode
    SelfPlayer(Trainer *trainer);
    SelfPlayer(bool, Trainer *trainer);
    // Testing mode
    SelfPlayer(bool logging, bool seed, Trainer *trainer);
    ~SelfPlayer() = default;

    void do_first_iteration();
    void do_iteration(float evaluation_result, float probability_result[NUM_TOTAL_MOVES],
    float dirichlet_noise[NUM_MOVES]);

};

#endif