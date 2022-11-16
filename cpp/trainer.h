#ifndef TRAINER_H
#define TRAINER_H

#include "trainmc.h"
#include "selfplayer.h"
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

// Number of Nodes to allocate together
const int BLOCK_SIZE = 128;
// Size of game state sample
const int GAME_STATE_SIZE = 96;
// Number of possible legal moves
const int NUM_LEGAL_MOVES = 96;

class Trainer {

    // We use a hash table to store the nodes
    // In this case, unique_ptr does not add too much space overhead
    // We could consider using normal pointers to get better memory locality
    // But we usually have random access so it might not be significant
    vector<Node*> hash_table;

    // Blocks used to allocate in chunk, saves on allocation cost
    TrainMC::Node *cur_block;
    // Index of first available node in current block
    int cur_ind;

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

    // Random generator for all operation
    std::mt19937 generator;

    void allocate(unsigned int pos);
    void rehash();

  public:

    Trainer(int num_games, int num_logged, int num_iterations, float states_to_evaluate[][GAME_STATE_SIZE], float c_puct, float epsilon);
    ~Trainer();

    void do_iteration(float evaluation_results[], float probability_results[][NUM_LEGAL_MOVES]);

    // Place root in hash table (different hash function)
    unsigned int place_root(unsigned int game_num);
    // Place node in hash table
    unsigned int find_node(unsigned int parent_num, unsigned int move_choice);
    unsigned int place_node(unsigned int parent_num, unsigned int move_choice);

};

#endif