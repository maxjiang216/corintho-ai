#include "trainer.h"
#include "trainmc.h"
#include "node.h"
#include "util.h"
#include <chrono>
#include <queue>
#include <algorithm>
#include <iostream>
using std::cout;

// Number of Nodes to allocate together
// Determine best number for this empirically
const uintf BLOCK_SIZE = 128;

// Determine best starting size empirically (per game)
// Should this be prime?
const uintf HASH_TABLE_SIZE = 2003;

// Prime multiplicative factor used in hashing child from parent node
// We can tinker with this constant to minimize collisions
// We can use a prime close to the sqrt of the number of entries (or some loose upper bound of that value)
// 2003 nodes per game, 25000 games, times 2 for possible rehash
// int type should be the same as the table size
const uintf MULT_FACTOR = 10009;
// Additive factor to space children of the same parent node
// We don't want children to be close to avoid clustering, but they do not need to be too far
// Average capacity close to 0.5 should probabilistically prevent long clusters
// Prime number?
const uintf ADD_FACTOR = 1009;

Trainer::Trainer(bool testing, uintf num_games, uintf num_logged, uintf num_iterations, float c_puct, float epsilon):
                 hash_table{vector<Node*>(num_games * HASH_TABLE_SIZE, nullptr)},
                 is_stale{vector<bool>(num_games * HASH_TABLE_SIZE, false)},
                 cur_block{(Node*)(new char[BLOCK_SIZE*sizeof(Node)])}, cur_ind{0},
                 num_games{num_games}, num_iterations{num_iterations}, iterations_done{0},
                 generator{std::mt19937{std::chrono::system_clock::now().time_since_epoch().count()}} {
    games.reserve(num_games);
    if (testing) {
        // this saves taking a max
        // speed in this part of the code is whatever though, maybe we combine these for brevity
        if (num_logged <= num_games) {
            for (uintf i = 0; i < num_logged; ++i) {
                games.emplace_back(true, i % 2, this);
            }
            for (uintf i = num_logged; i < num_games; ++i) {
                games.emplace_back(i % 2, this);
            }
        }
        else {
            for (uintf i = 0; i < num_games; ++i) {
                games.emplace_back(true, i % 2, this);
            }
        }
    }
    else {
        if (num_logged <= num_games) {
            // Is there a way to fill many objects after the fact?
            // We could construct an object and memcopy a bunch, in theory
            // Actually memcopy with a loop probably works and is close to optimal
            // Memcopy might be bad for object, though, only try this to optimize
            for (uintf i = 0; i < num_logged; ++i) {
                games.emplace_back(true, this);
            }
            for (uintf i = num_logged; i < num_games; ++i) {
                games.emplace_back(this);
            }
        }
        else {
            for (uintf i = 0; i < num_games; ++i) {
                games.emplace_back(true, this);
            }
        }
    }
    blocks.emplace_back(cur_block);
    // Set TrainMC static variables
    TrainMC::set_statics(num_iterations, c_puct, epsilon);
}

// How to prevent memory leak?
// Boost pool might make this easier
// Otherwise we need to keep all the char arrays we make?
Trainer::~Trainer() {
    for (size_t i = 0; i < blocks.size(); ++i) {
        delete[] blocks[i];
    }
}

// Should we use pointers instead?
void Trainer::do_iteration(float evaluations[], float probabilities[][NUM_TOTAL_MOVES],
float dirichlet_noise[][NUM_MOVES], float game_states[][GAME_STATE_SIZE]) {
    // We should first check if rehash is needed
    for (uintf i = 0; i < num_games; ++i) {
        // Pass neural net results
        if (i / num_iterations < iterations_done) {
            games[i].do_iteration(evaluations[i], probabilities[i], dirichlet_noise[i], game_states[i]);
        }
        // First iteration
        else if (i / num_iterations == iterations_done) {
            games[i].do_first_iteration(game_states[i]);
        }
    }
    uintf n_nodes = 0;
    for (uintf i = 0; i < hash_table.size(); ++i) {
        if (hash_table[i] != nullptr && !is_stale[i]) {
            ++n_nodes;
        }
    }
    cout << "There are " << n_nodes << " nodes in the hash_table and the hash table size is " << hash_table.size() << '\n';
    ++iterations_done;
}

uintf Trainer::place_root() {
    // Use random hash value
    uintf pos = generator() % hash_table.size();
    while (hash_table[pos] && !is_stale[pos]) {
        cout << pos << '\n';
        pos = (pos + 1) % hash_table.size();
    }
    // Unused space, place new node
    if (!hash_table[pos]) {
        place(pos);
    }
    // Overwrite stale node with starting position
    else {
        hash_table[pos]->overwrite();
    }
    return pos;
}

uintf Trainer::place_root(const Game &game, uintf depth) {
    // Use random hash value
    uintf pos = generator() % hash_table.size();
    while (hash_table[pos] && !is_stale[pos]) {
        pos = (pos + 1) % hash_table.size();
    }
    // Unused space, place new node
    if (!hash_table[pos]) {
        place(pos, game, depth);
    }
    else {
        hash_table[pos]->overwrite(game, depth);
    }
    // Otherwise, we are at a stale node, and we can simply return that location
    // The TrainMC will overwrite it
    // it does not need to tell the difference between a new node and a stale node
    return pos;
}

// move_choice is uintf so that we can copy a smaller number as a parameter
// 1 conversion either way
// This is very minor though
// We can't use depth in the hash
// Since find_next does not have access to this
uintf Trainer::place_next(const Game &game, uintf depth, uintf parent, uintf move_choice) {
    uintf pos = (parent * MULT_FACTOR + move_choice * ADD_FACTOR) % hash_table.size();
    // When placing a node, we can replace a stale node
    while (hash_table[pos] && !is_stale[pos] && hash_table[pos]->get_parent() != parent) {
        pos = (pos + 1) % hash_table.size();
    }

    // Unused space, allocate new node
    if (!hash_table[pos]) {
        place(pos, game, depth, parent, move_choice);
    }
    // Otherwise, we are at a stale node, and we can simply return that location
    // Overwrite it
    else {
        hash_table[pos]->overwrite(game, depth, parent, move_choice);
    }
    // Return the position, same in both cases
    return pos;
}

uintf Trainer::find_next(uintf parent, uintf move_choice) {
    uintf pos = (parent * MULT_FACTOR + move_choice * ADD_FACTOR) % hash_table.size();
    // We assume matching parent is sufficient
    // We do not want to store more than we need to
    // Parent must be stored to propagate evaluations up
    // If we use this as a "sufficient" check for equality
    // We do not need to store an additional hash value
    // And we do not need to check that the game states are equal
    // Since children of a parent node are quite spaced out
    // The chance of a collision between children is effectively 0
    // When finding a node, we stop at a match
    // The caller must make sure the node exists
    // Otherwise errors will probably happen
    while (hash_table[pos]->get_parent() != parent) {
        pos = (pos + 1) % hash_table.size();
    }
    // Node should always be found
    return pos;
}

inline Node* Trainer::get_node(uintf id) {
    return hash_table[id];
}

void Trainer::rehash() {
    // Double the size of the new table
    // Rehashing should be very seldom (no reason for more than once per run)
    // Time efficiency is not too important
    // Pointers in the table are small compared to Nodes
    // So space efficiency is also not too important
    uintf new_size = 2 * hash_table.size();
    vector<Node*> new_table(new_size, nullptr);

    // Queue for processing nodes
    // Store the index of the node we wish to rehash and the new hash
    // which is computed before hand since we need the parent
    std::queue<std::pair<uintf, uintf>> nodes;
    // Process roots first. We can reseed random values
    for (size_t i = 0; i < games.size(); ++i) {
        for (uintf j = 0; j < 2; ++j) {
            uintf pos = generator() % new_size, root = games[i].get_root(j);
            // There are no stale nodes
            // We only check for spaces already taken, although a collision at this point is very unlikely
            while (new_table[pos]) {
                pos = (pos + 1) % new_size;
            }
            // Use memcopy?
            new_table[pos] = hash_table[root];
            for (uintf k = 0; k < NUM_MOVES; ++k) {
                // Only search child nodes that have been visited
                if (hash_table[root]->has_visited(k)) {
                    nodes.push(std::make_pair(find_next(root, k), (root * MULT_FACTOR + k * ADD_FACTOR) % new_size));
                }
            }
        }
    }
    // Process internal nodes
    // Breadth first search should help higher nodes have less probing in the new table
    // And these are more likely to be processed more in the future (?)
    // We could use a priority queue based on number of visits
    // But that could be more hassle than is worth it
    while (!nodes.empty()) {
        uintf cur = nodes.front().first, pos = nodes.front().second;
        Node *cur_node = get_node(cur);
        while (new_table[pos]) {
            pos = (pos + 1) % new_size;
        }
        new_table[pos] = cur_node;
        for (uintf i = 0; i < NUM_MOVES; ++i) {
            // Only search child nodes that have been visited
            if (cur_node->has_visited(i)) {
                nodes.push(std::make_pair(find_next(cur, i), (cur * MULT_FACTOR + i * ADD_FACTOR) % new_size));
            }
        }
        nodes.pop();
    }
    hash_table = std::move(new_table);
    // None in the new table are stale
    is_stale = vector<bool>(new_size, false);
}

void Trainer::place(uintf pos) {
    // Need a new block
    if (cur_ind == BLOCK_SIZE) {
        cur_block = (Node*)(new char[BLOCK_SIZE*sizeof(Node)]);
        cur_ind = 0;
    }
    // Placement new for root node
    hash_table[pos] = new (cur_block + cur_ind) Node();

    ++cur_ind;
}

void Trainer::place(uintf pos, const Game &game, uintf depth) {
    // Need a new block
    if (cur_ind == BLOCK_SIZE) {
        cur_block = (Node*)(new char[BLOCK_SIZE*sizeof(Node)]);
        blocks.emplace_back(cur_block);
        cur_ind = 0;
    }
    // Placement new for root node
    hash_table[pos] = new (cur_block + cur_ind) Node(game, depth);

    ++cur_ind;
}

void Trainer::place(uintf pos, const Game &game, uintf depth, uintf parent, uintf move_choice) {
    // Need a new block
    if (cur_ind == BLOCK_SIZE) {
        cur_block = (Node*)(new char[BLOCK_SIZE*sizeof(Node)]);
        blocks.emplace_back(cur_block);
        cur_ind = 0;
    }
    // Placement new for internal node
    hash_table[pos] = new (cur_block + cur_ind) Node(game, depth, parent, move_choice);

    ++cur_ind;
}

void Trainer::move_down(uintf root, uintf move_choice) {
    // stale the root
    is_stale[root] = true;
    std::queue<uint> nodes;
    // Push children except for chosen one
    Node *root_node = get_node(root);
    for (uintf i = 0; i < NUM_MOVES; ++i) {
        if (root_node->has_visited(i) && i != move_choice) nodes.push(find_next(root, i));
    }
    while (!nodes.empty()) {
        uintf cur = nodes.front();
        Node *cur_node = get_node(cur);
        is_stale[cur] = true;
        for (uintf i = 0; i < NUM_MOVES; ++i) {
            if (cur_node->has_visited(i)) nodes.push(find_next(cur, i));
        }
        nodes.pop();
    }
}

// Deletes the whole tree
// Used when a game is done
// Or rarely if a tree moves down to a new node
void Trainer::delete_tree(uintf root) {
    std::queue<uint> nodes;
    nodes.push(root);
    while (!nodes.empty()) {
        uintf cur = nodes.front();
        Node *cur_node = get_node(cur);
        is_stale[cur] = true;
        for (uintf i = 0; i < NUM_MOVES; ++i) {
            if (cur_node->has_visited(i)) nodes.push(find_next(cur, i));
        }
        nodes.pop();
    }
}

uintf Trainer::generate() {
    return generator();
}