#ifndef TRAINER_H
#define TRAINER_H

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
    // Keep track of which nodes are stale
    // We need a move_tree method that takes a tree root
    // and the move choice to avoid marking that branch for deletion
    vector<bool> is_stale;
    vector<Node*> blocks;

    // Blocks used to allocate in chunk, saves on allocation cost
    Node *cur_block;
    // Index of first available node in current block
    uintf cur_ind;

    vector<SelfPlayer> games;
    // Number of games to play
    uintf num_games;
    
    // Number of iterations per move (used to do offsets and to pass onto SelfPlayer objects)
    uintf num_iterations;
    // Counter used to keep track of game offsets
    uintf iterations_done;

    // Random generator for all operation
    std::mt19937 generator;

    // Place root
    void place(uintf pos);
    void place (uintf pos, const Game &game, uintf depth);
    // Place node
    void place(uintf pos, const Game &game, uintf depth, uintf parent, uintf move_choice);
    // Rehash
    void rehash();

  public:

    Trainer(bool testing, uintf num_games, uintf num_logged, uintf num_iterations,
    float c_puct, float epsilon);
    ~Trainer();

    // We need a testing version of do_iteration, where we pass 2 sets of results, one for each neural net (dirichlet noise can be shared)
    void do_iteration(float evaluations[], float probabilities[][NUM_TOTAL_MOVES],
    float dirichlet[][NUM_MOVES], float game_states[][GAME_STATE_SIZE]);

    // Place root in hash table (random hash)
    uintf place_root();
    uintf place_root(const Game &game, uintf depth);
    // Place node in hash table
    uintf place_next(const Game &game, uintf depth, uintf parent, uintf move_choice);
    // Find the child node
    uintf find_next(uintf parent, uintf move_choice);

    Node* get_node(uintf id);

    void move_down(uintf root, uintf move_choice);
    void delete_tree(uintf root);

    // used by other classes to generate random numbers
    uintf generate();

};

#endif