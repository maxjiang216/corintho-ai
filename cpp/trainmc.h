#ifndef TRAINMC_H
#define TRAINMC_H

#include "node.h"
#include "util.h"
#include <bitset>
#include <memory>

using std::bitset;
using std::unique_ptr;
using std::shared_ptr;

class Trainer;
class Game;

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

    inline static uintf max_iterations = 1600;
    inline static float c_puct = 1.0, epsilon = 0.25;

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

    uintf get_root();
    bool is_uninitialized();
    const Game& get_game();

    static void set_statics(uintf new_max_iterations, float new_c_puct, float new_epsilon);

};

#endif
