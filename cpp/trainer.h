#ifndef TRAINER_H
#define TRAINER_H

#include "trainmc.h"
#include "selfplayer.h"
#include "util.h"
#include <vector>
#include <array>
#include <memory>
#include <utility>
#include <random>

using std::vector;
using std::array;
using std::unique_ptr;
using std::pair;

// This is the class that should interact with Cython

class Trainer {

    // We use a hash table to store the nodes
    vector<Node*> hash_table;

    // Blocks used to allocate in chunk, saves on allocation cost
    Node *cur_block;
    // Index of first available node in current block
    int cur_ind;

    vector<SelfPlayer> games;
    // Number of games to play
    uint16 num_games;
    
    // Number of iterations per move (used to do offsets and to pass onto SelfPlayer objects)
    uint16 num_iterations;
    // Counter used to keep track of game offsets
    uint16 iterations_done;

    // Location to write game states to evaluate
    float states_to_evaluate[][GAME_STATE_SIZE];

    // Random generator for all operation
    std::mt19937 generator;

    void allocate(uint32 pos);
    void rehash();

  public:

    Trainer(int num_games, int num_logged, int num_iterations, float states_to_evaluate[][GAME_STATE_SIZE],
    float c_puct, float epsilon);
    ~Trainer();

    // We need a testing version of do_iteration, where we pass 2 sets of results, one for each neural net (dirichlet noise can be shared)
    void do_iteration(float evaluation_results[], float probability_results[][NUM_TOTAL_MOVES],
    float dirichlet[][NUM_MOVES]);

    // Place root in hash table (different hash function)
    uint32 place_root();
    // Place node in hash table
    uint32 find_node(uint32 parent_num, uint32 move_choice);
    uint32 place_node(uint32 parent_num, uint32 move_choice);

};

#endif