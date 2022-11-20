#ifndef TRAINMC_H
#define TRAINMC_H

#include "game.h"
#include "trainer.h"
#include "node.h"
#include "util.h"
#include <bitset>
#include <memory>

using std::bitset;
using std::unique_ptr;
using std::shared_ptr;

class TrainMC {

    Trainer *trainer;
    // cur is the index of the current node
    uintf root, cur;
    // We often access as pointer
    // Should we store root_node? Check where we use it, if we use it multiple times before it updates
    Node *cur_node;
    // Used to keep track of when to choose a move
    uintf iterations_done;
    bool testing, logging;

    static uintf max_iterations;
    static float c_puct, epsilon;

    uintf choose_next();
    // I want to keep these as normal arrays instead of std::array for now
    // For predictability
    bool search(float game_state[GAME_STATE_SIZE]);
    void receive_evaluation(float evaluation, float probabilities[NUM_TOTAL_MOVES],
    float dirichlet_nosie[NUM_MOVES]);

  public:

    // Training
    TrainMC(Trainer *trainer);
    TrainMC(Trainer *trainer, bool);
    // Testing
    TrainMC(Trainer *trainer, bool logging, bool);
    ~TrainMC() = default;
    
    bool receive_opp_move(uintf move_choice, float game_state[GAME_STATE_SIZE]);
    bool do_iteration(float game_state[GAME_STATE_SIZE]);
    bool do_iteration(float evaluation_result, float probabilities[NUM_TOTAL_MOVES],
    float dirichlet_noise[NUM_MOVES], float game_state[GAME_STATE_SIZE]);
    void do_first_iteration(float game_state[GAME_STATE_SIZE]);
    void do_first_iteration(const Game &game, float game_state[GAME_STATE_SIZE]);

    uintf choose_move();

};

#endif
