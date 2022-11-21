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

    // I think we have to initialize these
    inline static uintf max_iterations = 1600;
    inline static float c_puct = 1.0, epsilon = 0.25;
    // Number of moves to use weighted random
    const uintf NUM_OPENING_MOVES = 4;

    // root and current nodes
    uintf root, cur;
    // Should we store root_node?
    Node *cur_node;

    // Used to keep track of when to choose a move
    uintf iterations_done;

    bool testing, logging;

    Trainer *trainer;

    void receive_evaluation(float evaluation, float probabilities[NUM_TOTAL_MOVES],
                            float dirichlet_nosie[NUM_MOVES]);
    bool search(float game_state[GAME_STATE_SIZE]);
    uintf choose_next();

  public:

    // Training
    TrainMC(Trainer *trainer);
    TrainMC(Trainer *trainer, bool);
    // Testing
    TrainMC(bool logging, Trainer *trainer, bool);
    ~TrainMC() = default;

    // First iterations are guaranteed not to end a turn
    // First iteration on starting position
    void do_first_iteration(float game_state[GAME_STATE_SIZE]);
    // First iteration on arbitrary position
    void do_first_iteration(const Game &game, float game_state[GAME_STATE_SIZE]);
    // First iteration of move
    bool do_iteration(float game_state[GAME_STATE_SIZE]);
    bool do_iteration(float evaluation_result, float probabilities[NUM_TOTAL_MOVES],
                      float dirichlet_noise[NUM_MOVES], float game_state[GAME_STATE_SIZE]);
    
    // Choose the next child to visit
    uintf choose_move();

    bool receive_opp_move(uintf move_choice, float game_state[GAME_STATE_SIZE],
                          const Game &game, uintf depth);

    // Accessors
    uintf get_root() const;
    const Game& get_game() const;    
    uintf get_depth() const;
    bool is_uninitialized() const;

    static void set_statics(uintf new_max_iterations, float new_c_puct, float new_epsilon);

};

#endif
