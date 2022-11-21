#include "trainmc.h"
#include "trainer.h"
#include "move.h"
#include <bitset>
#include <cmath>
#include <random>
#include <fstream>

using std::bitset;

TrainMC::TrainMC(Trainer *trainer): cur_node{nullptr}, iterations_done{1}, testing{false},
                                    logging{false}, trainer{trainer} {}

TrainMC::TrainMC(Trainer *trainer, bool): cur_node{nullptr}, iterations_done{1}, testing{false},
                                          logging{true}, trainer{trainer} {}

TrainMC::TrainMC(bool logging, Trainer *trainer, bool): cur_node{nullptr}, iterations_done{1}, testing{true},
                                                        logging{true}, trainer{trainer} {}

void TrainMC::do_first_iteration(float game_state[GAME_STATE_SIZE]) {

    // Create the root node
    root = trainer->place_root();
    cur = root;
    cur_node = trainer->get_node(root);

    cur_node->write_game_state(game_state);

}

void TrainMC::do_first_iteration(const Game &game, float game_state[GAME_STATE_SIZE]) {

    // Create the new root node
    root = trainer->place_root(game, 1);
    cur = root;
    cur_node = trainer->get_node(root);

    cur_node->write_game_state(game_state);

}

bool TrainMC::do_iteration(float game_state[GAME_STATE_SIZE]) {
    return search(game_state);
}

bool TrainMC::do_iteration(float evaluation, float probabilities[NUM_TOTAL_MOVES],
                           float dirichlet_noise[NUM_MOVES], float game_state[GAME_STATE_SIZE]) {
    receive_evaluation(evaluation, probabilities, dirichlet_noise);
    return search(game_state);
}

uintf TrainMC::choose_move() {

    uintf move_choice = 0;

    // In training, choose weighted random
    // For the first few moves
    if (cur_node->get_depth() < NUM_OPENING_MOVES && !testing) {
        uintf total = 0, children_visits[NUM_MOVES];
        for (uintf i = 0; i < NUM_MOVES; ++i) {
            if (cur_node->has_visited(i)) {
                children_visits[i] = trainer->get_node(trainer->find_next(cur, i))->get_visits();
                total += children_visits[i];
            }
        }
        uintf cur_total = 0;
        for (uintf i = 0; i < NUM_MOVES - 1; ++i) {
            if (cur_node->has_visited(i)) {
                cur_total += children_visits[i];
                if (cur_total >= total) move_choice = i;
            }
        }
        move_choice = NUM_MOVES - 1;
    }

    // Otherwise, choose randomly between the moves with the most visits/searches
    // Random offset is the easiest way to randomly break ties
    uintf id = trainer->generate() % NUM_MOVES, max_visits = 0;
    for (uintf i = 0; i < NUM_MOVES; ++i) {
        uintf cur_move = (id + i) % NUM_MOVES;
        if (cur_node->has_visited(cur_move)) {
            uintf cur_visits = trainer->get_node(trainer->find_next(cur, cur_move))->get_visits();
            if (cur_visits > max_visits) {
                max_visits = cur_visits;
                move_choice = cur_move;
            }
        }
    }

    uintf old_root = root;
    root = trainer->find_next(root, move_choice);
    trainer->move_down(old_root, move_choice);
    // Set cur_node here, but this is not needed if the game is complete
    // We can try to find a better place, but it's pretty insignificant and avoid possible hassle
    cur = root;
    cur_node = trainer->get_node(cur);
    // Do this to avoid clashing parents
    // All nodes with possibly overwritten parents
    // Should either be stale
    // Our have itself as its parent
    cur_node->null_parent();
    iterations_done = 0;

    std::fstream fs("log.txt", std::fstream::app);
    fs << cur_node->get_depth() << '\n' << Move{move_choice} << '\n' << cur_node->get_game() << '\n';

    return move_choice;

}

bool TrainMC::receive_opp_move(uintf move_choice, float game_state[GAME_STATE_SIZE],
                               const Game &game, uintf depth) {

    // If we have visited the node before
    // Simply move the tree down
    if (trainer->get_node(root)->has_visited(move_choice)) {
        uintf old_root = root;
        root = trainer->find_next(root, move_choice);
        // We should stale things afterwards now
        // Since find_next skips over stale nodes
        trainer->move_down(old_root, move_choice);
        cur = root;
        cur_node = trainer->get_node(cur);
        cur_node->null_parent();
        iterations_done = 0;
        // we don't need an evaluation
        return false;
    }

    // The node doesn't exist
    // Practically (with sufficient searches) this probably won't happen but it is theoretically possible
    // Maybe we could empirically track if it happens
    else {
        trainer->delete_tree(root);
        // Copy opponent game state into our root
        root = trainer->place_root(game, depth);
        cur = root;
        cur_node = trainer->get_node(root);
        // We need an evaluation
        cur_node->write_game_state(game_state);
        // this is the first iteration of the turn
        iterations_done = 1;
        // we need an evaluation
        return true;
    }

}

uintf TrainMC::get_root() const {
    return root;
}

const Game& TrainMC::get_game() const {
    return cur_node->get_game();
}

uintf TrainMC::get_depth() const {
    return cur_node->get_depth();
}

bool TrainMC::is_uninitialized() const {
    return cur_node == nullptr;
}

void TrainMC::set_statics(uintf new_max_iterations, float new_c_puct, float new_epsilon) {
    max_iterations = new_max_iterations;
    c_puct = new_c_puct;
    epsilon = new_epsilon;
}

void TrainMC::receive_evaluation(float evaluation, float probabilities[NUM_TOTAL_MOVES],
                                 float dirichlet_noise[NUM_MOVES]) {

    // We need to to figure out how to map the probabilities
    for (uintf i = 0; i < NUM_MOVES; ++i) {
        cur_node->set_probability(i, probabilities[i]);
    }
    
    // Apply the legal move filter
    // and keep track of the total sum so we can normalize afterwards
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
        if (cur_node->get_probability(i) > 0) {
            cur_node->adjust_probability(i, scalar, dirichlet_noise[i]);
        }
    }

    // Propagate evaluation
    float cur_evaluation = evaluation;
    if (cur_node->get_to_play() % 2 == 1) cur_evaluation *= -1.0;
    while (cur != root) {
        cur_node->add_evaluation(cur_evaluation);
        cur_evaluation *= -1.0;
        cur = cur_node->get_parent();
        cur_node = trainer->get_node(cur);
    }
    // Propagate to root
    cur_node->add_evaluation(cur_evaluation);

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
                cur_node->set_visit(move_choice);
                // Create the new node
                cur = trainer->place_next(cur_node->get_game(), cur_node->get_depth(), cur,move_choice);
                cur_node = trainer->get_node(cur);
                break;
            }
            // Otherwise, move down normally
            else {
                cur_node->increment_visits();
                cur = trainer->find_next(cur, move_choice);
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
                    cur_evaluation *= -1.0;
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

    return need_evaluation;

}

uintf TrainMC::choose_next() {

    // Random value lower than -1.0
    float max_value = -2.0;
    // Initialize this variable to be safe
    uintf move_choice = 0;

    for (uintf i = 0; i < NUM_MOVES; ++i) {
        float u;
        // Check if it is a legal move
        if (cur_node->is_legal(i)) {
            if (cur_node->has_visited(i)) {
                Node *next = trainer->get_node(trainer->find_next(cur, i));
                u = -1.0 * next->get_evaluation() / (float)next->get_visits() +
                    cur_node->get_probability(i) * sqrt((float)cur_node->get_visits() - 1.0) /
                    ((float)next->get_visits() + 1);
                
            }
            // If not visited, set action value to 0
            else {
                u = cur_node->get_probability(i) * sqrt((float)cur_node->get_visits() - 1.0);
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