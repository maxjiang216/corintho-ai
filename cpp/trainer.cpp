#include "trainer.h"
#include <chrono>

using std::make_pair;

// Determine best starting size empirically (per game)
// Should this be prime?
const int HASH_TABLE_SIZE = 2003;
const int MAXIMUM_PROBES = 10;
// Prime multiplicative factor used in hashing child from parent node
// We can tinker with this constant to minimize collisions
// We can use a prime close to the sqrt of the number of entries (or some loose upper bound of that value)
// 2003 nodes per game, 25000 games, times 2 for possible rehash
const unsigned int MULT_FACTOR = 10009;
// Additive factor to space children of the same parent node
// We don't want children to be close to avoid clustering, but they do not need to be too far
// Average capacity close to 0.5 should probabilistically prevent long clusters
// Prime number?
const unsigned int ADD_FACTOR = 1009;

Trainer::Trainer(int num_games, int num_logged, int num_iterations, float states_to_evaluate[][GAME_STATE_SIZE], float c_puct, float epsilon):
                 cur_block{new Node[BLOCK_SIZE]}, cur_ind{0}, num_games{num_games}, num_logged{num_logged}, num_iterations{num_iterations}, states_to_evaluate{states_to_evaluate}, generator{std::mt19937{std::chrono::system_clock::now().time_since_epoch().count()}} {
    int table_size = num_games * HASH_TABLE_SIZE;
    hash_table.reserve(table_size);
    for (int i = 0; i < table_size; ++i) {
        hash_table.push_back(unique_ptr<Node>(nullptr));
    }
    games.reserve(num_games);
    for (int i = 0; i < num_games; ++i) {
        games.push_back(SelfPlayer());
    }
}

void do_iteration(float evaluation_results[], float probability_results[][NUM_LEGAL_MOVES]) {
    for (int i = 0; i < num_games; ++i) {
        // Pass neural net results
        if (i / num_iterations < iterations_done) {
            games[i].do_iteration(evaluation_results[i], probability_results[i]);
        }
        // First iteration
        else if (i / num_iterations == iterations_done) {
            games[i].do_first_iteration();
        }
    }
    ++iterations_done;
}

int find_first() {
    // Use random hash value
    return find(generator());
}

int find_node(int parent_num, int move_choice) {
    // The starting position for probing needs to be deterministic
    // In order for get_next to work
    find(parent_num * MULT_FACTOR + move_choice * ADD_FACTOR);

}

int get_node(int parent_num, int move_choice) {
    unsigned int pos = (parent_num * MULT_FACTOR + move_choice * ADD_FACTOR) % hash_table.size();
    // We assume matching parent is sufficient
    // We do not want to store more than we need to
    // Parent must be stored to propagate evaluations up
    // If we use this as a "sufficient" check for equality
    // We do not need to store an additional hash value
    // And we do not need to check that the game states are equal
    // Since children of a parent node are quite spaced out
    // The chance of a collision between children is effectively 0
    while (hash_table[pos] && !hash_table[pos]->is_stale && !hash_table[pos]->parent == parent_num) {
        ++pos;
    }
    // The node should always be found
    return pos;
}

// Should give TrainMC the index of the unique ptr of the Node to use
// There are no guarantees on what is already stored in the Node
// It may be an old Node
int find(int hash_val) {
    unsigned int pos = hash_val % hash_table.size();
    // We need to continue probing
    // We keep the capacity of the table below a certain constant (0.6?)
    // Long clusters are probabilistically almost impossible
    while (hash_table[pos] && !hash_table[pos]->is_stale) {
        ++pos;
    }
    // Unused space, allocate new node
    if (!hash_table[pos]) {
        // Need a new block
        if (cur_ind == BLOCK_SIZE) {
            cur_block = new Node[BLOCK_SIZE];
            cur_ind = 0;
        }
        hash_table[pos] = unique_ptr<Node>(cur_block + cur_ind);
        ++cur_ind;
    }
    // Otherwise, we are at a stale node, and we can simply return that location
    return pos;
}