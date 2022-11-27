#include "trainer.h"
#include "trainmc.h"
#include "node.h"
#include "util.h"
#include <chrono>
#include <queue>
#include <algorithm>
#include <vector>
#include <string>
#include <iostream>

using std::vector;
using std::string;
using std::cerr;

Trainer::Trainer(uintf num_games, uintf num_logged, uintf num_iterations,
                 float c_puct, float epsilon, const string &logging_folder, uintf random_seed):
                 num_games{std::max(num_games, (uintf)1)}, num_iterations{std::max(num_iterations, (uintf)2)},
                 iterations_done{0}, games_done{0}, is_done{vector<bool>(num_games, false)},
                 hash_table{vector<Node*>(num_games * HASH_TABLE_SIZE, nullptr)},
                 is_stale{vector<bool>(num_games * HASH_TABLE_SIZE, false)},
                 cur_block{(Node*)(new char[BLOCK_SIZE*sizeof(Node)])}, cur_ind{0},
                 logging_folder{logging_folder}, generator{random_seed} {
    initialize(false, num_games, num_logged, c_puct, epsilon, logging_folder);
}

Trainer::Trainer(uintf num_games, uintf num_logged, uintf num_iterations,
                 float c_puct, float epsilon, const string &logging_folder, uintf random_seed, bool):
                 num_games{std::max(num_games, (uintf)1)}, num_iterations{std::max(num_iterations, (uintf)2)},
                 iterations_done{0}, games_done{0}, is_done{vector<bool>(num_games, false)},
                 hash_table{vector<Node*>(num_games * HASH_TABLE_SIZE, nullptr)},
                 is_stale{vector<bool>(num_games * HASH_TABLE_SIZE, false)},
                 cur_block{(Node*)(new char[BLOCK_SIZE*sizeof(Node)])}, cur_ind{0},
                 logging_folder{logging_folder}, generator{random_seed} {
    initialize(true, num_games, num_logged, c_puct, epsilon, logging_folder);
}

Trainer::~Trainer() {
    for (size_t i = 0; i < blocks.size(); ++i) {
        delete[] blocks[i];
    }
    delete[] cur_block;
}

bool Trainer::do_iteration(float evaluations[], float probabilities[],
                           float dirichlet_noise[], float game_states[]) {

    for (uintf i = 0; i < num_games; ++i) {
        if (!is_done[i]) {
            // Avoid division by 0 in the rare case than num_games < num_iterations / 2
            if (i / std::max((uintf)1, (num_games / (num_iterations / 2))) < iterations_done) {
                bool is_completed = games[i]->do_iteration(evaluations[i], &probabilities[i*NUM_TOTAL_MOVES],
                                                           &dirichlet_noise[i*NUM_MOVES], &game_states[i*GAME_STATE_SIZE]);
                if (is_completed) {
                    is_done[i] = true;
                    ++games_done;
                    if (games_done == num_games) {
                        return true;
                    }
                }
            }
            else if (i / std::max((uintf)1, (num_games / (num_iterations / 2))) == iterations_done) {
                games[i]->do_first_iteration(&game_states[i*GAME_STATE_SIZE]);
            }
        }
    }
    ++iterations_done;

    return false;

}

bool Trainer::do_iteration(float evaluations_1[], float probabilities_1[],
                           float evaluations_2[], float probabilities_2[],
                           float dirichlet_noise[], float game_states[]) {
    for (uintf i = 0; i < num_games; ++i) {
        if (!is_done[i]) {
            // Avoid division by 0 in the rare case than num_games < num_iterations / 2
            if (i / std::max((uintf)1, (num_games / (num_iterations / 2))) < iterations_done) {
                bool is_completed = games[i]->do_iteration(evaluations_1[i], &probabilities_1[i*NUM_TOTAL_MOVES],
                                                           evaluations_2[i], &probabilities_2[i*NUM_TOTAL_MOVES],
                                                           &dirichlet_noise[i*NUM_MOVES], &game_states[i*GAME_STATE_SIZE]);
                if (is_completed) {
                    is_done[i] = true;
                    ++games_done;
                    if (games_done == num_games) {
                        return true;
                    }
                }
            }
            else if (i / std::max((uintf)1, (num_games / (num_iterations / 2))) == iterations_done) {
                games[i]->do_first_iteration(&game_states[i*GAME_STATE_SIZE]);
            }
        }
    }
    ++iterations_done;

    return false;

}

uintf Trainer::place_root() {

    // Use random hash value
    uintf new_hash = generator();

    uintf pos = new_hash % hash_table.size();

    // Probe for usable space
    while (hash_table[pos] != nullptr && !is_stale[pos]) {
        pos = (pos + 1) % hash_table.size();
    }

    // Unused space, place new node
    if (hash_table[pos] == nullptr) {
        place(pos, new_hash);
    }
    // Overwrite stale node with starting position
    else {
        hash_table[pos]->overwrite(new_hash);
        is_stale[pos] = false;
    }

    return pos;

}

uintf Trainer::place_root(const Game &game, uintf depth) {

    // Use random hash value
    uintf new_hash = generator();
    uintf pos = new_hash % hash_table.size();

    // Probe for usable space
    while (hash_table[pos] != nullptr && !is_stale[pos]) {
        pos = (pos + 1) % hash_table.size();
    }

    // Unused space, place new node
    if (hash_table[pos] == nullptr) {
        place(pos, new_hash, game, depth);
    }
    // Overwrite stale node
    else {
        hash_table[pos]->overwrite(new_hash, game, depth);
        is_stale[pos] = false;
    }

    return pos;

}

uintf Trainer::place_next(uintf seed, const Game &game, uintf depth, uintf parent, uintf move_choice) {

    uintf new_hash = hash(seed, move_choice);
    uintf pos = (new_hash) % hash_table.size();

    // Probe for usable space
    while (hash_table[pos] != nullptr && !is_stale[pos]) {
        pos = (pos + 1) % hash_table.size();
    }

    // Unused space, allocate new node
    if (hash_table[pos] == nullptr) {
        place(pos, new_hash, game, depth, parent, move_choice);
    }
    // Overwrite stale node
    else {
        hash_table[pos]->overwrite(new_hash, game, depth, parent, move_choice);
        is_stale[pos] = false;
    }
    
    return pos;

}

uintf Trainer::find_next(uintf seed, uintf move_choice) const {

    // Match hash function from place_next
    uintf new_hash = hash(seed, move_choice);
    uintf pos = new_hash % hash_table.size();

    // We assume matching parent is sufficient
    // and we assume no nullptrs are encountered
    // Be wary of matching parents due to overwrite nodes, though
    uintf counter = 0;
    while (is_stale[pos] || hash_table[pos]->get_seed() != new_hash) {
        pos = (pos + 1) % hash_table.size();
        ++counter;
    }

    // We assume the node is always found
    return pos;

}

Node* Trainer::get_node(uintf id) const {
    return hash_table[id];
}

void Trainer::move_down(uintf root, uintf move_choice) {

    // Stale the root
    is_stale[root] = true;

    std::queue<uintf> nodes;

    // Push children except for chosen one
    Node *root_node = get_node(root);
    for (uintf i = 0; i < NUM_MOVES; ++i) {
        if (root_node->has_visited(i) && i != move_choice) nodes.push(find_next(root_node->get_seed(), i));
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

void Trainer::initialize(bool testing, uintf num_games, uintf num_logged,
                         float c_puct, float epsilon, const string &logging_folder) {

    games.reserve(num_games);

    if (num_games < num_logged) num_logged = num_games;

    if (testing) {
        for (uintf i = 0; i < num_logged; ++i) {
            games.emplace_back(new SelfPlayer{i % 2, this, i, logging_folder});
        }
        for (uintf i = num_logged; i < num_games; ++i) {
            games.emplace_back(new SelfPlayer{i % 2, this});
        }
    }
    else {
        for (uintf i = 0; i < num_logged; ++i) {
            games.emplace_back(new SelfPlayer{this, i, logging_folder});
        }
        for (uintf i = num_logged; i < num_games; ++i) {
            games.emplace_back(new SelfPlayer{this});
        }
    }

    // Set TrainMC static variables
    TrainMC::set_statics(num_iterations, c_puct, epsilon);
    
}

uintf Trainer::count_samples() const {
    uintf counter = 0;
    for (uintf i = 0; i < num_games; ++i) {
        counter += games[i]->count_samples();
    }
    return counter;
}

void Trainer::write_samples(float *game_states, float *evaluation_samples, float *probability_samples) const {
    uintf offset = 0;
    for (uintf i = 0; i < num_games; ++i) {
        uintf num_samples = games[i]->write_samples(game_states + offset * GAME_STATE_SIZE,
                                                    evaluation_samples + offset,
                                                    probability_samples + offset * NUM_TOTAL_MOVES);
        offset += num_samples;
    }
}

uintf Trainer::hash(uintf seed, uintf move_choice) {
    return seed * MULT_FACTOR + move_choice * ADD_FACTOR + 1;
}

void Trainer::place(uintf pos, uintf seed) {
    
    // Need a new block
    if (cur_ind == BLOCK_SIZE) {
        // Record the old block so we can delete it later
        blocks.emplace_back(cur_block);
        cur_block = (Node*)(new char[BLOCK_SIZE*sizeof(Node)]);
        cur_ind = 0;
    }

    // Placement new
    hash_table[pos] = new (cur_block + cur_ind) Node(seed);
    ++cur_ind;

}

void Trainer::place(uintf pos, uintf seed, const Game &game, uintf depth) {

    // Need a new block
    if (cur_ind == BLOCK_SIZE) {
        // Record the old block so we can delete it later
        blocks.emplace_back(cur_block);
        cur_block = (Node*)(new char[BLOCK_SIZE*sizeof(Node)]);
        cur_ind = 0;
    }

    // Placement new
    hash_table[pos] = new (cur_block + cur_ind) Node(seed, game, depth);
    ++cur_ind;

}

void Trainer::place(uintf pos, uintf seed, const Game &game, uintf depth, uintf parent, uintf move_choice) {

    // Need a new block
    if (cur_ind == BLOCK_SIZE) {
        // Record the old block so we can delete it later
        blocks.emplace_back(cur_block);
        cur_block = (Node*)(new char[BLOCK_SIZE*sizeof(Node)]);
        cur_ind = 0;
    }

    // Placement new
    hash_table[pos] = new (cur_block + cur_ind) Node(seed, game, depth, parent, move_choice);
    ++cur_ind;

}

void Trainer::stale_all(std::queue<uintf> &nodes) {
    while (!nodes.empty()) {
        uintf cur = nodes.front();
        Node *cur_node = get_node(cur);
        is_stale[cur] = true;
        for (uintf i = 0; i < NUM_MOVES; ++i) {
            if (cur_node->has_visited(i)) {
                nodes.push(find_next(cur_node->get_seed(), i));
            }
        }
        nodes.pop();
    }
}

void Trainer::rehash() {

    // Double the size
    // This constant should be good enough
    // And it is easy to work with
    uintf new_size = hash_table.size() * 2 + 1;
    vector<Node*> new_table(new_size, nullptr);

    std::cerr << "Rehashing! Old size: " << hash_table.size() << " New size: " << new_size << '\n';

    // We need to store the old index to reference it later
    // As well as the new hash
    std::queue<std::pair<uintf, uintf>> nodes;

    // Process roots first
    // We reseed random values
    for (uintf i = 0; i < games.size(); ++i) {
        for (uintf j = 0; j < 2; ++j) {
            uintf root = games[i]->get_root(j), root_seed = get_node(root)->get_seed();
            uintf pos = root_seed % new_size;
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
                    nodes.push(std::make_pair(find_next(root_seed, k),
                               hash(root_seed, k)));
                }
            }
        }
    }

    while (!nodes.empty()) {
        uintf cur = nodes.front().first, seed = nodes.front().second, pos = seed % new_size;
        Node *cur_node = get_node(cur);
        while (new_table[pos]) {
            pos = (pos + 1) % new_size;
        }
        new_table[pos] = cur_node;
        for (uintf i = 0; i < NUM_MOVES; ++i) {
            if (cur_node->has_visited(i)) nodes.push(std::make_pair(find_next(seed, i),
                                                     hash(seed, i)));
        }
        nodes.pop();
    }

    hash_table = std::move(new_table);

    // No nodes in the new table are stale
    // We need to use the old is_stale in the function so replace at the end
    is_stale = vector<bool>(new_size, false);

}

float Trainer::get_score() const {
    float score = 0;
    for (uintf i = 0; i < num_games; ++i) {
        score += games[i]->get_score();
    }
    return score / (float)num_games;
}

bool Trainer::is_all_done() const {
    return games_done == num_games;
}