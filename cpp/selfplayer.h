#ifndef SELFPLAYER_H
#define SELFPLAYER_H

#include "trainmc.h"
#include "util.h"
#include <vector>
#include <string>
#include <memory>
#include <fstream>

class Trainer;

class SelfPlayer {

    TrainMC players[2];
    // This could be bool, but int is probably faster
    // Also makes it easier for Trainer to assign seeds
    uintf to_play;

    bool logging;

    // seed is only used in testing
    uintf seed;

    // This way we don't allocate memory if there is no logging file
    std::unique_ptr<std::ofstream> logging_file;
    
    Trainer *trainer;

    bool do_iteration(float game_state[GAME_STATE_SIZE]);

  public:
    
    // Training mode
    SelfPlayer(Trainer *trainer);
    SelfPlayer(Trainer *trainer, uintf id, const std::string &logging_folder);
    // Testing mode
    SelfPlayer(uintf seed, Trainer *trainer);
    // This is slightly inefficient, but more general, and only happens at most once per testing run
    SelfPlayer(uintf seed, Trainer *trainer, uintf id, const std::string &logging_folder);
    SelfPlayer(SelfPlayer&&) = default;
    ~SelfPlayer() = default;

    void do_first_iteration(float game_state[GAME_STATE_SIZE]);
    // Training
    bool do_iteration(float evaluation, float probabilities[NUM_TOTAL_MOVES],
                      float dirichlet_noise[NUM_MOVES], float game_state[GAME_STATE_SIZE]);
    // Testing
    bool do_iteration(float evaluation_1, float probabilities_1[NUM_TOTAL_MOVES],
                      float evaluation_2, float probabilities_2[NUM_TOTAL_MOVES],
                      float dirichlet_noise[NUM_MOVES], float game_state[GAME_STATE_SIZE]);
    
    uintf get_root(uintf player_num) const;

};

#endif