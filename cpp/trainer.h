#ifndef TRAINER_H
#define TRAINER_H

#include "selfplayer.h"
#include "util.h"
#include <vector>
#include <random>
#include <queue>
#include <string>

// This is the class that should interact with Cython

class Trainer {

    // Number of Nodes to allocate together
    // Determine best number for this empirically
    // This will be deprecated once we use Boost pools
    static const uintf BLOCK_SIZE = 4096;

    // Determine best starting size empirically (per game)
    static const uintf HASH_TABLE_SIZE = 5003;

    // Prime multiplicative factor used in hashing child from parent node
    // We can tinker with this constant to minimize collisions
    static const uintf MULT_FACTOR = 10009;
    // Additive factor to space children of the same parent node
    // We don't want children to be close to avoid clustering, but they do not need to be too far
    // Average capacity <LOAD_FACTOR should probabilistically prevent long clusters
    // Largest prime less than HASH_TABLE_SIZE / NUM_MOVES, so that we never wrap around
    // Guarantees no collision amongst siblings as long as probe length is sufficiently short
    static const uintf ADD_FACTOR = 23;

    // Load factor for hash table rehashing
    static constexpr float LOAD_FACTOR = 0.75;

    // Number of games to play
    uintf num_games;
    // Number of iterations per move (used to compute offsets)
    uintf num_iterations;
    // Counter used to keep track of number of iterations done for offsets
    uintf iterations_done;
    // Keep track of how many games are done
    uintf games_done;

    // SelfPlayer objects
    std::vector<SelfPlayer*> games;
    // Track which games are done
    std::vector<bool> is_done;

    // We use a hash table to store the nodes
    std::vector<Node*> hash_table;
    // Keep track of which nodes are stale
    std::vector<bool> is_stale;

    // Blocks used to allocate in chunk, saves on allocation cost
    // Eventually this should be replaced with Boost pools
    Node *cur_block;
    // Index of first available node in current block
    uintf cur_ind;
    // List of blocks that have been allocated so we can delete them appropriately
    std::vector<Node*> blocks;

    // Add some std::fstreams for logging files

    // Random generator for all operations
    std::mt19937 generator;

    // Initialize SelfPlayers (factored out of different version of constructor)
    void initialize(bool testing, uintf num_games, uintf num_logged,
                    float c_puct, float epsilon, const std::string &logging_folder);

    // Hash child nodes
    static uintf hash(uintf seed, uintf move_choice);

    // Place root
    void place(uintf pos, uintf seed);
    void place (uintf pos, uintf seed, const Game &game, uintf depth);
    // Place node
    void place(uintf pos, uintf seed, const Game &game, uintf depth, uintf parent, uintf move_choice);

    // Stale all nodes in the queue and their descendents
    void stale_all(std::queue<uintf> &nodes);

    // Rehash
    void rehash();

  public:

    // Training
    Trainer() = default;
    Trainer(uintf num_games, uintf num_logged, uintf num_iterations,
            float c_puct, float epsilon, const std::string &logging_folder, uintf random_seed);
    // Testing
    Trainer(uintf num_games, uintf num_logged, uintf num_iterations,
            float c_puct, float epsilon, const std::string &logging_folder, uintf random_seed, bool);
    //Trainer(const Trainer&) = default;
    ~Trainer();

    // Main function that will be called by Cython
    // Training version
    bool do_iteration(float evaluations[], float probabilities[], float dirichlet[],
                      float game_states[]);
    // Testing version
    bool do_iteration(float evaluations_1[], float probabilities_1[],
                      float evaluations_2[], float probabilities_2[],
                      float dirichlet_noise[], float game_states[]);

    // Place root in hash table (random hash)
    // Place starting position
    uintf place_root();
    // Place a given position
    // This is used when receiving an opponent move the player did not search (rare)
    uintf place_root(const Game &game, uintf depth);

    // Place child node in hash table
    uintf place_next(uintf seed, const Game &game, uintf depth, uintf parent, uintf move_choice);
    // Find the child node
    uintf find_next(uintf seed, uintf move_choice) const;

    // Get the pointer to a Node given the index
    Node* get_node(uintf id) const;

    // Stale nodes
    // move_down used normally when a move has been chosen
    void move_down(uintf root, uintf move_choice);
    // delete_tree is used when the game ends
    // or when moving to a move not searched
    void delete_tree(uintf root);

    // Call occasionally by Cython to rehash if too filled
    void rehash_if_full();

    // used by other classes to generate random numbers
    uintf generate();

};

#endif