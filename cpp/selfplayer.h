#ifndef SELFPLAYER_H
#define SELFPLAYER_H

#include "trainer.h"
#include "trainmc.h"
#include <vector>
#include <array>

using std::vector;
using std::array;

class SelfPlayer {

    // Seed is only used in testing
    // Optimize for space. Speed of testing is not important
    bool testing, logging, seed;
    TrainMC players[2];

  public:
    
    // Training mode
    SelfPlayer();
    SelfPlayer(bool logging);
    // Testing mode
    SelfPlayer(bool logging, bool seed);
    ~SelfPlayer() = default;

    void do_first_iteration();
    void do_iteration(float evaluation_result, float probability_result[NUM_LEGAL_MOVES], float dirichlet_noise[NUM_LEGAL_MOVES]);

};

#endif