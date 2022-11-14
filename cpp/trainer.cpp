#include "trainer.h"
#include <chrono>
#include <queue>
#include <algorithm>

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
    // We should first check if rehash is needed
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

unsigned int find_first() {
    // Use random hash value
    unsigned int pos = generator() % hash_table.size();
    // We need to continue probing
    // We keep the capacity of the table below a certain constant (0.6?)
    // Long clusters are probabilistically almost impossible
    while (hash_table[pos] && !hash_table[pos]->is_stale) {
        pos = (pos + 1) % hash_table.size();
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

unsigned int find_node(unsigned int parent_num, int move_choice) {
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
        pos = (pos + 1) % hash_table.size();
    }
    // Node found
    if (hash_table[pos]->parent == parent_num) {
        return pos;
    }
    // Not found, create new node
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

void rehash() {
    // Double the size of the new table
    // Rehashing should be very seldom (no reason for more than once per run)
    // Time efficiency is not too important
    // Pointers in the table are small compared to Nodes
    // So space efficiency is also not too important
    vector<unique_ptr<Node>> new_table;
    unsigned int new_size = 2 * hash_table.size();
    new_table.reserve(new_size);
    for (unsigned int i = 0; i < new_size; ++i) {
        new_table.emplace_back(nullptr);
    }

    // Queue for processing nodes
    std::queue<int> nodes;
    // Process roots first. We can reseed random values
    for (size_t i = 0; i < games.size(); ++i) {
        for (size_t j = 0; j < 2; ++j) {
            unsigned int pos = generator() % new_size, root = games[i].players[j].root;
            // There are no stale nodes
            // We only check for spaces already taken, although a collision at this point is very unlikely
            while (new_table[pos]) {
                pos = (pos + 1) % new_size;
            }
            new_table[pos] = hash_table[root];
            for (unsigned int k = 0; k < NUM_LEGAL_MOVES; ++k) {
                nodes.push((root * MULT_FACTOR + k * ADD_FACTOR) % hash_table.size());
            }
        }
    }
    // Process internal nodes
    // Breadth first search should help higher nodes have less probing in the new table
    // And these are more likely to be processed more in the future (?)
    // We could use a priority queue based on number of visits
    // But that could be more hassle than is worth it
    while (!nodes.empty()) {
        unsigned int pos = queue.front();
        while (!hash_table[pos])
    }
    hash_table = std::move(new_table);
}