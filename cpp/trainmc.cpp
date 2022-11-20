#include "trainmc.h"
#include "trainer.h"
#include <memory>
#include <bitset>
#include <math.h>
#include <random>
#include <iostream>
using std::cout;

using std::unique_ptr;
using std::bitset;

// Used to initialize tree
// Training
TrainMC::TrainMC(Trainer *trainer): trainer{trainer}, cur_node{nullptr}, iterations_done{1},
testing{false}, logging{false} {
}

TrainMC::TrainMC(Trainer *trainer, bool): trainer{trainer}, cur_node{nullptr}, iterations_done{1}, testing{false}, logging{true} {
}
// Testing
TrainMC::TrainMC(Trainer *trainer, bool, bool logging): trainer{trainer}, cur_node{nullptr}, iterations_done{1}, testing{true},
logging{logging} {
    root = trainer->place_root();
    cur = root;
    cur_node = trainer->get_node(root);
}

// First search on root node
// Guaranteed to not yield a move output
// We need to pass the memory buffers into here
void TrainMC::do_first_iteration(float game_state[GAME_STATE_SIZE]) {
    cout << "TrainMC::do_first_iteration\n";
    // The second TrainMC does not need the starting position root node
    // Also, doing this here lets us detect the first iteration on the second tree
    // By testing nullptr for cur_node
    root = trainer->place_root();
    cout << "TrainMC::do_first_iteration 38\n";
    cur = root;
    cur_node = trainer->get_node(root);
    cout << "TrainMC::do_first_iteration 41\n";

    cur_node->write_game_state(game_state);

    cout << "TrainMC::do_first_iteration 45\n";

}

void TrainMC::do_first_iteration(const Game &game, float game_state[GAME_STATE_SIZE]) {

    // The second TrainMC does not need the starting position root node
    // Also, doing this here lets us detect the first iteration on the second tree
    // By testing nullptr for cur_node
    root = trainer->place_root(game, 1);
    cur = root;
    cur_node = trainer->get_node(root);

    cur_node->write_game_state(game_state);

}

// Choose the next child to visit
uintf TrainMC::choose_next() {

    float max_value = -2.0;
    // Initialize this variable
    // Gets rid of a warning, safer, speed is negligible
    uintf move_choice = 0;

    for (uintf i = 0; i < NUM_MOVES; ++i) {
        float u;
        // Check if it is a legal move
        if (cur_node->is_legal(i)) {
            // if not visited, set action value to 0
            if (!(cur_node->has_visited(i))) {
                // double check the -1 here
                u = cur_node->get_probability(i) * pow(cur_node->get_visits() - 1, 0.5);
            }
            else {
                Node *next = trainer->get_node(trainer->find_next(cur, i));
                u = -1.0 * next->get_evaluation() / (float)next->get_visits() +
                    cur_node->get_probability(i) * pow((float)cur_node->get_visits() - 1, 0.5) / ((float)next->get_visits() + 1);
            }
            // We assume there are no ties
            if (u > max_value) {
                max_value = u;
                move_choice = i;
            }
        }
    }

    return move_choice;

}

void TrainMC::receive_evaluation(float evaluation, float probabilities[NUM_TOTAL_MOVES],
float dirichlet_noise[NUM_MOVES]) {
    // We need to to figure out how to map the probabilities
    
    // Now, apply the legal move filter
    // Legal moves should be found at node creation
    // There is no point in doing it lazily (?)
    // meanwhile, keep track of the total sum so we can normalize afterwards;
    float sum = 0;
    for (uintf i = 0; i < NUM_MOVES; ++i) {
        if (!(cur_node->is_legal(i))) {
            cur_node->set_probability(i, 0.0);
        }
        else {
            sum += cur_node->get_probability(i);
        }
    }

    // Multiplying by this is more efficient
    float scalar = 1.0 / sum;

    // Normalize probabilities and apply dirichlet noise
    for (uintf i = 0; i < NUM_MOVES; ++i) {
        // Only application of dirichlet noise needs the if
        // Could we separate these? The number of branches is the same though
        if (cur_node->get_probability(i) > 0) {
            // is it faster to do them together? Then there is no in-place operations
            cur_node->adjust_probability(i, scalar, dirichlet_noise[i]);
        }
    }

    // Propagate evaluation
    float cur_evaluation = evaluation * pow(-1.0, (float)cur_node->get_to_play());
    while (cur_node != nullptr) {
        cur_node->add_evaluation(cur_evaluation);
        cur_evaluation *= -1;
        cur = cur_node->get_parent();
        cur_node = trainer->get_node(cur);
    }
}

bool TrainMC::do_iteration(float game_state[GAME_STATE_SIZE]) {
    cout << "TrainMC::do_iteration(float*)\n";
    return search(game_state);
}

bool TrainMC::do_iteration(float evaluation, float probabilities[NUM_TOTAL_MOVES],
float dirichlet_noise[NUM_MOVES], float game_state[GAME_STATE_SIZE]) {
    cout << "TrainMC::do_iteration(float*, float*, float*, float*)\n";
    receive_evaluation(evaluation, probabilities, dirichlet_noise);
    return search(game_state);
}


// Figure out what the Python code does for this for the first search of a move, where evaluation is not needed
// We can overload, factor out the search, add "receive_evaluation" function
bool TrainMC::search(float game_state[GAME_STATE_SIZE]) {

    bool need_evaluation = false;
    while (!need_evaluation && iterations_done < max_iterations) {
        ++iterations_done;
        while (true) {

            uintf move_choice = choose_next();

            // Exploring a new node
            if (!(cur_node->has_visited(move_choice))) {
                // Create the new node
                // visits is initialized to 1
                cur = trainer->place_next(cur_node->get_game(), cur_node->get_depth(), cur_node->get_parent(), move_choice);
                cur_node = trainer->get_node(cur);
                break;
            }
            // Otherwise, move down normally
            else {
                // Add to visit count up front, although it should come after we choose_next
                cur_node->increment_visits();
                cur = trainer->find_next(cur_node->get_parent(), move_choice);
                cur_node = trainer->get_node(cur);
            }
        }
        // Check for terminal state, otherwise evaluation is needed
        if (cur_node->is_terminal()) {
            // Don't propagate if value is 0
            if (cur_node->get_result() != DRAW) {
                // Propagate evaluation
                // In a decisive terminal state, the person to play is always the loser
                // It is important we don't add to visits here, because we skip this for draws
                float cur_evaluation = -1.0;
                while (cur != root) {
                    cur_node->add_evaluation(cur_evaluation);
                    cur_evaluation *= -1;
                    cur = cur_node->get_parent();
                    cur_node = trainer->get_node(cur);
                }
                // We need to propagate to root
                cur_node->add_evaluation(cur_evaluation);
            }
        }
        // Otherwise, request an evaluation
        else {
            cur_node->write_game_state(game_state);
            need_evaluation = true;
        }
    }

    // If true, self player should request an evaluation
    // If false, number of searches is reached and move can be chosen
    return need_evaluation;

}

// Choose the best move once searches are done
uintf TrainMC::choose_move() {

    // Choose weighted random
    // cur_node should always be root after searches
    if (cur_node->get_depth() < 4 && !testing) {
        uintf total = 0, children_visits[NUM_MOVES];
        for (uintf i = 0; i < NUM_MOVES; ++i) {
            if (cur_node->has_visited(i)) {
                children_visits[i] = trainer->get_node(trainer->find_next(cur, i))->get_visits();
                total += children_visits[i];
            }
        }
        uintf id = trainer->generate() % total;
        total = 0;
        for (uintf i = 0; i < NUM_MOVES - 1; ++i) {
            if (cur_node->has_visited(i)) {
                total += children_visits[i];
                if (total >= id) return i;
            }
        }
        return NUM_MOVES - 1;
    }
    // Otherwise, choose randomly between the moves with the most visits/searches
    // Random offset is the easiest way to randomly break ties
    uintf id = trainer->generate() % NUM_MOVES, max_visits = 0, move_choice = 0;
    // the move choices are all internal, so they can be fit into NUM_MOVES
    // we might have to consider rotations when we add that
    for (uintf i = 0; i < NUM_MOVES; ++i) {
        if (cur_node->has_visited(i)) {
            uintf cur_visits = trainer->get_node(trainer->find_next(cur, i))->get_visits();
            if (cur_visits > max_visits) {
                max_visits = cur_visits;
                move_choice = (id + i) % NUM_MOVES;
            }
        }
    }

    trainer->move_down(root, move_choice);
    root = trainer->find_next(root, move_choice);
    iterations_done = 0;
    // Where to reset cur_node?

    return move_choice;
}

// Move the tree down a level
// Used when opponent moves
// this should call the tree move down method from trainer
bool TrainMC::receive_opp_move(uintf move_choice, float game_state[GAME_STATE_SIZE]) {

    // If we have visited the node before; this is the simple case
    if (trainer->get_node(root)->has_visited(move_choice)) {
        trainer->move_down(root, move_choice);
        root = trainer->find_next(root, move_choice);
        iterations_done = 0;
        // we don't need an evaluation
        return false;
    }
    // The node doesn't exist
    // Practically (with sufficient searches) this probably won't happen but it is theoretically possible
    // Maybe we could empirically track if it happens
    else {
        // Find node first, then delete tree, then place new node
        Node *root_node = trainer->get_node(root);
        trainer->delete_tree(root);
        root = trainer->place_root(root_node->get_game(), root_node->get_depth());
        cur = root;
        cur_node = trainer->get_node(root);
        cur_node->write_game_state(game_state);
        // this is the first iteration of the turn
        iterations_done = 1;
        // we need an evaluation
        return true;
    }
}

uintf TrainMC::get_root() {
    return root;
}

bool TrainMC::is_uninitialized() {
    return cur_node == nullptr;
}

const Game& TrainMC::get_game() {
    return cur_node->get_game();
}

void TrainMC::set_statics(uintf new_max_iterations, float new_c_puct, float new_epsilon) {
    max_iterations = new_max_iterations;
    c_puct = new_c_puct;
    epsilon = new_epsilon;
}