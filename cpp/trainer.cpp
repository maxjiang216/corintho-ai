#include "trainer.h"
#include "util.h"
#include <chrono>
#include <queue>
#include <algorithm>

// Number of Nodes to allocate together
// Determine best number for this empirically
const uint BLOCK_SIZE = 128;

// Determine best starting size empirically (per game)
// Should this be prime?
const uint HASH_TABLE_SIZE = 2003;

// Prime multiplicative factor used in hashing child from parent node
// We can tinker with this constant to minimize collisions
// We can use a prime close to the sqrt of the number of entries (or some loose upper bound of that value)
// 2003 nodes per game, 25000 games, times 2 for possible rehash
// int type should be the same as the table size
const uint MULT_FACTOR = 10009;
// Additive factor to space children of the same parent node
// We don't want children to be close to avoid clustering, but they do not need to be too far
// Average capacity close to 0.5 should probabilistically prevent long clusters
// Prime number?
const uint ADD_FACTOR = 1009;

Trainer::Trainer(bool testing, uint num_games, uint num_logged, uint num_iterations,
float states_to_evaluate[][GAME_STATE_SIZE], float c_puct, float epsilon):
                 hash_table{vector<Node*>(nullptr, num_games * HASH_TABLE_SIZE)}, cur_block{new Node[BLOCK_SIZE]},
                 cur_ind{0}, num_games{num_games}, num_iterations{num_iterations},
                 states_to_evaluate{states_to_evaluate},
                 generator{std::mt19937{std::chrono::system_clock::now().time_since_epoch().count()}} {
    games.reserve(num_games);
    if (testing) {
        // this saves taking a max
        // speed in this part of the code is whatever though, maybe we combine these for brevity
        if (num_logged <= num_games) {
            for (uint i = 0; i < num_logged; ++i) {
                games.emplace_back(true, i % 2, this);
            }
            for (uint i = num_logged; i < num_games; ++i) {
                games.emplace_back(i % 2, this);
            }
        }
        else {
            for (uint i = 0; i < num_games; ++i) {
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
            for (uint i = 0; i < num_logged; ++i) {
                games.emplace_back(true, this);
            }
            for (uint i = num_logged; i < num_games; ++i) {
                games.emplace_back(this);
            }
        }
        else {
            for (uint i = 0; i < num_games; ++i) {
                games.emplace_back(true, this);
            }
        }
    }
    // Set TrainMC static variables
    TrainMC::max_iterations = num_iterations;
    TrainMC::c_puct = c_puct;
    TrainMC::epsilon = epsilon;
}

// How to prevent memory leak?
// Boost pool might make this easier
// Otherwise we need to keep all the char arrays we make?
Trainer::~Trainer() {
    for (size_t i = 0; i < hash_table.size(); ++i) {
        ~(*(hash_table[i]));
    }
}

// Should we use pointers instead?
void Trainer::do_iteration(float evaluation_results[], float probability_results[][NUM_TOTAL_MOVES],
float dirichlet_noise[][NUM_MOVES]) {
    // We should first check if rehash is needed
    for (uint i = 0; i < num_games; ++i) {
        // Pass neural net results
        if (i / num_iterations < iterations_done) {
            games[i].do_iteration(evaluation_results[i], probability_results[i], dirichlet_noise[i], game_states[i]);
        }
        // First iteration
        else if (i / num_iterations == iterations_done) {
            games[i].do_first_iteration(game_states[i]);
        }
    }
    ++iterations_done;
}

uint Trainer::place_root() {
    // Use random hash value
    uint pos = generator() % hash_table.size();
    while (hash_table[pos] && !hash_table[pos]->is_stale) {
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

uint Trainer::place_root(const Game &game, uint depth) {
    // Use random hash value
    uint pos = generator() % hash_table.size();
    while (hash_table[pos] && !hash_table[pos]->is_stale) {
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

// move_choice is uint so that we can copy a smaller number as a parameter
// 1 conversion either way
// This is very minor though
// We can't use depth in the hash
// Since find_next does not have access to this
uint Trainer::place_next(const Game &game, uint depth, uint parent, uint move_choice) {
    uint pos = (parent * MULT_FACTOR + move_choice * ADD_FACTOR) % hash_table.size();
    // When placing a node, we can replace a stale node
    while (hash_table[pos] && !hash_table[pos]->is_stale && hash_table[pos]->parent != parent) {
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

uint Trainer::find_next(uint parent, uint move_choice) {
    uint pos = (parent * MULT_FACTOR + move_choice * ADD_FACTOR) % hash_table.size();
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
    while (hash_table[pos]->parent != parent_num) {
        pos = (pos + 1) % hash_table.size();
    }
    // Node should always be found
    return pos;
}

Node* Trainer::get_node(uint id) {
    return hash_table[id];
}

void Trainer::rehash() {
    // Double the size of the new table
    // Rehashing should be very seldom (no reason for more than once per run)
    // Time efficiency is not too important
    // Pointers in the table are small compared to Nodes
    // So space efficiency is also not too important
    uint new_size = 2 * hash_table.size();
    vector<Node*> new_table(nullptr, new_size);

    // Queue for processing nodes
    std::queue<uint> nodes;
    // Process roots first. We can reseed random values
    for (size_t i = 0; i < games.size(); ++i) {
        for (uint j = 0; j < 2; ++j) {
            uint pos = generator() % new_size, root = games[i].players[j].root;
            // There are no stale nodes
            // We only check for spaces already taken, although a collision at this point is very unlikely
            while (new_table[pos]) {
                pos = (pos + 1) % new_size;
            }
            // Use memcopy?
            new_table[pos] = hash_table[root];
            for (uint k = 0; k < NUM_MOVES; ++k) {
                // Only search child nodes that have been visited
                if (hash_table[root]->visited.get(k)) {
                    nodes.push((root * MULT_FACTOR + k * ADD_FACTOR) % hash_table.size());
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
        uint pos = queue.front();
        while (hash_table[pos]->parent != parent_num) {
            pos = (pos + 1) % hash_table.size();
        }
        while (new_table[pos]) {
            pos = (pos + 1) % new_size;
        }
        new_table[pos] = hash_table[queue.front()];
        for (uint i = 0; i < NUM_MOVES; ++i) {
            // Only search child nodes that have been visited
            if (hash_table[queue.front()]->visited.get(i)) {
                nodes.push((queue.front() * MULT_FACTOR + i * ADD_FACTOR) % hash_table.size());
            }
        }
        queue.pop();
    }
    hash_table = std::move(new_table);
}

void Trainer::place(uint pos) {
    // Need a new block
    if (cur_ind == BLOCK_SIZE) {
        cur_block = new char[BLOCK_SIZE*sizeof(Node)];
        cur_ind = 0;
    }
    // Placement new for root node
    hash_table[pos] = new (cur_block + cur_ind*sizeof(Node)) Node();

    ++cur_ind;
}

void Trainer::place(uint pos, const Game &game) {
    // Need a new block
    if (cur_ind == BLOCK_SIZE) {
        cur_block = new char[BLOCK_SIZE*sizeof(Node)];
        cur_ind = 0;
    }
    // Placement new for root node
    hash_table[pos] = new (cur_block + cur_ind*sizeof(Node)) Node(game);

    ++cur_ind;
}

void Trainer::place(uint pos, const Game &game, uint depth) {
    // Need a new block
    if (cur_ind == BLOCK_SIZE) {
        cur_block = new char[BLOCK_SIZE*sizeof(Node)];
        cur_ind = 0;
    }
    // Placement new for root node
    hash_table[pos] = new (cur_block + cur_ind*sizeof(Node)) Node(game, depth);

    ++cur_ind;
}

void Trainer::place(uint pos, const Game &game, uint depth, uint parent, uint move_choice) {
    // Need a new block
    if (cur_ind == BLOCK_SIZE) {
        cur_block = new char[BLOCK_SIZE*sizeof(Node)];
        cur_ind = 0;
    }
    // Placement new for internal node
    hash_table[pos] = new (cur_block + cur_ind*sizeof(Node)) Node(game, depth, parent, move_choice);

    ++cur_ind;
}

void Trainer::move_down(uint root, uint move_choice) {
    // stale the root
    is_stale[root] = true;
    std::queue<uint> nodes;
    // Push children except for chosen one
    Node *root_node = get_node(root);
    for (uint i = 0; i < NUM_MOVES; ++i) {
        if (root_node->visited.get(i) && i != move_choice) nodes.push(find_next(root, i));
    }
    while (!nodes.empty()) {
        uint cur = nodes.front();
        Node *cur_node = get_node(cur);
        is_stale[i] = true;
        for (uint i = 0; i < NUM_MOVES; ++i) {
            if (cur_node->visited.get(i)) nodes.push(find_next(cur, i));
        }
    }
}

// Deletes the whole tree
// Used when a game is done
// Or rarely if a tree moves down to a new node
void Trainer::delete_tree(uint root) {
    std::queue<uint> nodes;
    nodes.push(root);
    while (!nodes.empty()) {
        uint cur = nodes.front();
        Node *cur_node = get_node(cur);
        is_stale[i] = true;
        for (uint i = 0; i < NUM_MOVES; ++i) {
            if (cur_node->visited.get(i)) nodes.push(find_next(cur, i));
        }
    }
}