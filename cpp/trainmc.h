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
    uint32 root, cur;
    // We often access as pointer
    // Should we store root_node? Check where we use it, if we use it multiple times before it updates
    Node *cur_node;
    // Used to keep track of when to choose a move
    uint16 iterations_done;
    bool testing, logging;

    static uint16 max_iterations = 1600;
    static float c_puct = 1.0, epsilon = 0.25;

    uint8 choose_next();
    // I want to keep these as normal arrays instead of std::array for now
    // For predictability
    bool search(float evaluation, float probabilities[NUM_TOTAL_MOVES], float dirichlet_nosie[NUM_MOVES]);

  public:

    // Training
    TrainMC(Trainer *trainer);
    TrainMC(Trainer *trainer, bool);
    // Testing
    TrainMC(Trainer *trainer, bool logging, bool);
    ~TrainMC() = default;
    
    void receive_opp_move(uint8 move_choice);
    uint8 do_iterations()
    void do_first_iteration(float game_state[GAME_STATE_SIZE]);
    void do_first_iteration(const Game &game, float game_state[GAME_STATE_SIZE]);

};

#endif
