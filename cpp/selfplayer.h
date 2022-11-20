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
    uint seed;
    TrainMC players[2];
    // This could be bool, but int is probably faster
    // Also makes it easier for Trainer to assign seeds
    uint to_play;
    Trainer *trainer;

  public:
    
    // Training mode
    SelfPlayer(Trainer *trainer);
    SelfPlayer(bool, Trainer *trainer);
    // Testing mode
    SelfPlayer(uint seed, Trainer *trainer);
    SelfPlayer(bool, uint seed, Trainer *trainer);
    ~SelfPlayer() = default;

    void do_first_iteration(float game_state[GAME_STATE_SIZE]);
    void do_iteration(float evaluation_result, float probability_result[NUM_TOTAL_MOVES],
    float dirichlet_noise[NUM_MOVES], float game_state[GAME_STATE_SIE]);

};

#endif