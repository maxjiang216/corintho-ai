#ifndef TRAINER_H
#define TRAINER_H

#include "trainmc.h"
#include "selfplayer.h"
#include <vector>
#include <memory>

using std::vector;
using std::unique_ptr;

// This is the class that should interact with Cython

// Number of Nodes to allocate together
const int BLOCK_SIZE = 100;
// Size of game state sample
const int GAME_STATE_SIZE = 96;
// Number of possible legal moves
const int NUM_LEGAL_MOVES = 96;

class Trainer {

    // We use a hash table to store the nodes
    // In this case, unique_ptr does not add too much space overhead
    // We could consider using normal pointers to get better memory locality
    // But we usually have random access so it might not be significant
    vector<unique_ptr<TrainMC::Node>> hash_table;
    vector<SelfPlayer> games;
    // Number of games to play
    int num_games;
    // Number of games to log
    int num_logged;
    
    // Number of iterations per move (used to do offsets and to pass onto SelfPlayer objects)
    int num_iterations;
    // Counter used to keep track of game offsets
    int iterations_done;

    // Location to write game states to evaluate
    float states_to_evaluate[][GAME_STATE_SIZE];

    void rehash();

  public:

    Trainer(int num_games, int num_logged, int num_iterations, float states_to_evaluate[][GAME_STATE_SIZE], float c_puct, float epsilon);
    ~Trainer() = default;

    void do_iteration(float evluation_results[], float probability_results[][NUM_LEGAL_MOVES]);

    // Place node in hash table
    void place();

};

#endif