#include "trainer.h"
#include "trainmc.h"
#include "node.h"
#include "util.h"
#include <chrono>
#include <queue>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
using std::cout;

using std::vector;

// PUBLIC METHODS

// Check if using Node* instead of char* constructs the object prematurely
// We can print from the Node constructor to see
Trainer::Trainer(uintf num_games, uintf num_logged, uintf num_iterations, float c_puct, float epsilon):
                 num_games{std::max(num_games, (uintf)1)}, num_iterations{std::max(num_iterations, (uintf)2)},
                 iterations_done{0}, games_done{0}, is_done{vector<bool>(num_games, false)},
                 hash_table{vector<Node*>(num_games * HASH_TABLE_SIZE + 1, nullptr)},
                 is_stale{vector<bool>(num_games * HASH_TABLE_SIZE + 1, false)},
                 cur_block{(Node*)(new char[BLOCK_SIZE*sizeof(Node)])}, cur_ind{0},
                 generator{std::mt19937{(uintf)std::chrono::system_clock::now().time_since_epoch().count()}} {
    initialize(false, num_games, num_logged, c_puct, epsilon);
}

Trainer::Trainer(uintf num_games, uintf num_logged, uintf num_iterations, float c_puct, float epsilon, bool):
                 num_games{std::max(num_games, (uintf)1)}, num_iterations{std::max(num_iterations, (uintf)2)},
                 iterations_done{0}, games_done{0}, is_done{vector<bool>(num_games, false)},
                 hash_table{vector<Node*>(num_games * HASH_TABLE_SIZE + 1, nullptr)},
                 is_stale{vector<bool>(num_games * HASH_TABLE_SIZE + 1, false)},
                 cur_block{(Node*)(new char[BLOCK_SIZE*sizeof(Node)])}, cur_ind{0},
                 generator{std::mt19937{(uintf)std::chrono::system_clock::now().time_since_epoch().count()}} {
    initialize(true, num_games, num_logged, c_puct, epsilon);
}

Trainer::~Trainer() {
    for (size_t i = 0; i < blocks.size(); ++i) {
        delete[] blocks[i];
    }
    delete[] cur_block;
}

bool Trainer::do_iteration(float evaluations[], float probabilities[][NUM_TOTAL_MOVES],
                           float dirichlet_noise[][NUM_MOVES], float game_states[][GAME_STATE_SIZE]) {

    for (uintf i = 0; i < num_games; ++i) {
        if (!is_done[i]) {
            // Avoid division by 0 in the rare case than num_games < num_iterations / 2
            if (i / std::max((uintf)1, (num_games / (num_iterations / 2))) < iterations_done) {
                bool is_completed = games[i].do_iteration(evaluations[i], probabilities[i],
                                                          dirichlet_noise[i], game_states[i]);
                if (is_completed) {
                    cout << "Game " << i << " is complete!\n";
                    is_done[i] = true;
                    ++games_done;
                    if (games_done == num_games) {
                        return true;
                    }
                }
            }
            else if (i / std::max((uintf)1, (num_games / (num_iterations / 2))) == iterations_done) {
                games[i].do_first_iteration(game_states[i]);
            }
        }
    }
    ++iterations_done;

    return false;

}

bool Trainer::do_iteration(float evaluations_1[], float probabilities_1[][NUM_TOTAL_MOVES],
                           float evaluations_2[], float probabilities_2[][NUM_TOTAL_MOVES],
                           float dirichlet_noise[][NUM_MOVES], float game_states[][GAME_STATE_SIZE]) {
    for (uintf i = 0; i < num_games; ++i) {
        if (!is_done[i]) {
            // Avoid division by 0 in the rare case than num_games < num_iterations / 2
            if (i / std::max((uintf)1, (num_games / (num_iterations / 2))) < iterations_done) {
                bool is_completed = games[i].do_iteration(evaluations_1[i], probabilities_1[i],
                                                          evaluations_2[i], probabilities_2[i],
                                                          dirichlet_noise[i], game_states[i]);
                if (is_completed) {
                    is_done[i] = true;
                    ++games_done;
                    if (games_done == num_games) {
                        return true;
                    }
                }
            }
            else if (i / std::max((uintf)1, (num_games / (num_iterations / 2))) == iterations_done) {
                games[i].do_first_iteration(game_states[i]);
            }
        }
    }
    ++iterations_done;

    return false;

}

uintf Trainer::place_root() {

    // Use random hash value
    uintf pos = generator() % hash_table.size();

    // Probe for usable space
    while (hash_table[pos] != nullptr && !is_stale[pos]) {
        pos = (pos + 1) % hash_table.size();
    }

    // Unused space, place new node
    if (hash_table[pos] == nullptr) {
        place(pos);
    }
    // Overwrite stale node with starting position
    else {
        hash_table[pos]->overwrite();
        is_stale[pos] = false;
    }

    return pos;

}

uintf Trainer::place_root(const Game &game, uintf depth) {

    // Use random hash value
    uintf pos = generator() % (hash_table.size() - 1) + 1;

    // Probe for usable space
    while (hash_table[pos] != nullptr && !is_stale[pos]) {
        pos = pos % (hash_table.size() - 1) + 1;
    }

    // Unused space, place new node
    if (hash_table[pos] == nullptr) {
        place(pos, game, depth);
    }
    // Overwrite stale node
    else {
        hash_table[pos]->overwrite(game, depth);
        is_stale[pos] = false;
    }

    return pos;

}

uintf Trainer::place_next(const Game &game, uintf depth, uintf parent, uintf move_choice) {

    uintf pos = hash(parent, move_choice, hash_table.size());

    // Probe for usable space
    while (hash_table[pos] != nullptr && !is_stale[pos]) {
        pos = pos % (hash_table.size() - 1) + 1;
    }

    // Unused space, allocate new node
    if (hash_table[pos] == nullptr) {
        place(pos, game, depth, parent, move_choice);
    }
    // Overwrite stale node
    else {
        hash_table[pos]->overwrite(game, depth, parent, move_choice, pos, is_stale[pos]);
        is_stale[pos] = false;
    }

    hash_table[parent]->children[move_choice] = pos;
    
    return pos;

}

uintf Trainer::find_next(uintf parent, uintf move_choice) const {

    // Match hash function from place_next
    uintf pos = hash(parent, move_choice, hash_table.size());

    // We assume matching parent is sufficient
    // and we assume no nullptrs are encountered
    // Be wary of matching parents due to overwrite nodes, though
    uintf counter = 0;
    while (hash_table[pos]->get_parent() != parent || is_stale[pos]) {
        //std::cerr << pos << ' ' << hash_table[pos] << ' ' << hash_table[pos]->get_parent() << ' ' << parent << ' ' << hash_table[parent]->children[move_choice] << '\n';
        pos = pos % (hash_table.size() - 1) + 1;
        ++counter;
    }

    if (counter > 1) std::cerr << "probed " << counter << " times!\n";

    // We assume the node is always found
    return pos;

}

Node* Trainer::get_node(uintf id) const {
    return hash_table[id];
}

void Trainer::move_down(uintf root, uintf move_choice) {

    // Stale the root
    //std::fstream fs ("over.txt", std::fstream::app);
    //fs << "stale move_down " << root << '\n';
    is_stale[root] = true;

    std::queue<uintf> nodes;

    // Push children except for chosen one
    Node *root_node = get_node(root);
    for (uintf i = 0; i < NUM_MOVES; ++i) {
        if (root_node->has_visited(i) && i != move_choice) nodes.push(find_next(root, i));
    }

    // Stale all descendents
    stale_all(nodes);

}

void Trainer::delete_tree(uintf root) {

    std::queue<uintf> nodes;
    nodes.push(root);

    // Stale root and all descendents
    stale_all(nodes);

}

void Trainer::rehash_if_full() {

    // Measure how full the hash table is
    uintf active_nodes = 0;
    for (uintf i = 0; i < hash_table.size(); ++i) {
        if (hash_table[i] != nullptr && !is_stale[i]) {
            ++active_nodes;
        }
    }

    std::cerr << "active nodes " << active_nodes << " table size " << hash_table.size() << '\n';

    // Rehash if too full
    if ((float)active_nodes / hash_table.size() > LOAD_FACTOR) {
        rehash();
    }

}

// This method is not const because the state of the generator changes
// when we generate a random number
uintf Trainer::generate() {
    return generator();
}

void Trainer::initialize(bool testing, uintf num_games, uintf num_logged, float c_puct, float epsilon) {

    games.reserve(num_games);

    if (num_games < num_logged) num_logged = num_games;

    if (testing) {
        for (uintf i = 0; i < num_logged; ++i) {
            games.emplace_back(i % 2, this, true);
        }
        for (uintf i = num_logged; i < num_games; ++i) {
            games.emplace_back(i % 2, this);
        }
    }
    else {
        for (uintf i = 0; i < num_logged; ++i) {
            games.emplace_back(this, true);
        }
        for (uintf i = num_logged; i < num_games; ++i) {
            games.emplace_back(this);
        }
    }

    // Set TrainMC static variables
    TrainMC::set_statics(num_iterations, c_puct, epsilon);
    
}

uintf Trainer::hash(uintf parent, uintf move_choice, uintf table_size) {
    return (parent * MULT_FACTOR + move_choice * ADD_FACTOR + 1) % (table_size - 1) + 1;
}

void Trainer::place(uintf pos) {
    
    // Need a new block
    if (cur_ind == BLOCK_SIZE) {
        // Record the old block so we can delete it later
        blocks.emplace_back(cur_block);
        cur_block = (Node*)(new char[BLOCK_SIZE*sizeof(Node)]);
        cur_ind = 0;
    }

    // Placement new
    hash_table[pos] = new (cur_block + cur_ind) Node();
    ++cur_ind;

}

void Trainer::place(uintf pos, const Game &game, uintf depth) {

    // Need a new block
    if (cur_ind == BLOCK_SIZE) {
        // Record the old block so we can delete it later
        blocks.emplace_back(cur_block);
        cur_block = (Node*)(new char[BLOCK_SIZE*sizeof(Node)]);
        cur_ind = 0;
    }

    // Placement new
    hash_table[pos] = new (cur_block + cur_ind) Node(game, depth);
    ++cur_ind;

}

void Trainer::place(uintf pos, const Game &game, uintf depth, uintf parent, uintf move_choice) {

    // Need a new block
    if (cur_ind == BLOCK_SIZE) {
        // Record the old block so we can delete it later
        blocks.emplace_back(cur_block);
        cur_block = (Node*)(new char[BLOCK_SIZE*sizeof(Node)]);
        cur_ind = 0;
    }

    // Placement new
    hash_table[pos] = new (cur_block + cur_ind) Node(game, depth, parent, move_choice, pos);
    ++cur_ind;

}

void Trainer::stale_all(std::queue<uintf> &nodes) {
    //std::cerr << "stale_all " << nodes.front() << '\n';
    while (!nodes.empty()) {
        uintf cur = nodes.front();
        Node *cur_node = get_node(cur);
        std::fstream fs ("over.txt", std::fstream::app);
        fs << "stale stale_all " << cur << '\n';
        is_stale[cur] = true;
        for (uintf i = 0; i < NUM_MOVES; ++i) {
            if (cur_node->has_visited(i)) nodes.push(find_next(cur, i));
        }
        nodes.pop();
    }
}

void Trainer::rehash() {

    // Double the size
    // This constant should be good enough
    // And it is easy to work with
    uintf new_size = hash_table.size() * 2;
    vector<Node*> new_table(new_size, nullptr);

    // We need to store the old index to reference it later
    // As well as the new hash
    std::queue<std::pair<uintf, uintf>> nodes;

    // Process roots first
    // We reseed random values
    for (uintf i = 0; i < games.size(); ++i) {
        for (uintf j = 0; j < 2; ++j) {
            uintf pos = generator() % new_size, root = games[i].get_root(j);
            // There are no stale nodes in the new table yet
            // A collision at this point is very unlikely
            while (new_table[pos] != nullptr) {
                pos = (pos + 1) % new_size;
            }
            // Use memcopy to copy?
            new_table[pos] = hash_table[root];
            for (uintf k = 0; k < NUM_MOVES; ++k) {
                // Only search child nodes that have been visited
                if (hash_table[root]->has_visited(k)) {
                    nodes.push(std::make_pair(find_next(root, k),
                               hash(root, k, new_size)));
                }
            }
        }
    }

    while (!nodes.empty()) {
        uintf cur = nodes.front().first, pos = nodes.front().second;
        Node *cur_node = get_node(cur);
        while (new_table[pos]) {
            pos = (pos + 1) % new_size;
        }
        new_table[pos] = cur_node;
        for (uintf i = 0; i < NUM_MOVES; ++i) {
            if (cur_node->has_visited(i)) nodes.push(std::make_pair(find_next(cur, i),
                                                     hash(cur, i, new_size)));
        }
        nodes.pop();
    }

    hash_table = std::move(new_table);

    // No nodes in the new table are stale
    // We need to use the old is_stale in the function so replace at the end
    is_stale = vector<bool>(new_size, false);

}