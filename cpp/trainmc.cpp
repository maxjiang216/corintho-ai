#include "trainmc.h"
#include "game.h"
#include "move.h"
#include "util.h"
#include "node.h"
#include <memory>
#include <bitset>
#include <math.h>
#include <random>

using std::unique_ptr;
using std::bitset;
using std::memmove;

// Used to initialize tree
// Training
TrainMC::TrainMC(Trainer *trainer): trainer{trainer}, iterations_done{1},
testing{false}, logging{false},  {
    // We cannot put this in initialization list
    // Evaluation order is not guaranteed
    root = trainer->place_root();
    cur = root;
    cur_node = trainer->get_node(root);
}
TrainMC::TrainMC(Trainer *trainer, bool): trainer{trainer}, iterations_done{1}, testing{false}, logging{true} {
    root = trainer->place_root();
    cur = root;
    cur_node = trainer->get_node(root);
}
// Testing
TrainMC::TrainMC(Trainer *trainer, bool, bool logging): trainer{trainer}, iterations_done{1}, testing{true},
logging{logging} {
    root = trainer->place_root();
    cur = root;
    cur_node = trainer->get_node(root);
}

// First search on root node
// Guaranteed to not yield a move output
// We need to pass the memory buffers into here
void do_first_iteration() {

    // We need to write the game state into the memory buffer here
    // visits and iterations_done are initialized to 1
    // So no need to increment

}

// Choose the next child to visit
uint8 choose_next() {

    float max_value = -2.0;
    uint8 move_choice;

    for (uint8 i = 0; i < NUM_MOVES; ++i) {
        float u;
        // Check if it is a legal move
        if (cur_node->legal_moves.get(i)) {
            // if not visited, set action value to 0
            if (!(cur_node->visited.get(i))) {
                // double check the -1 here
                u = cur_node->probabilities[i] * pow(cur_node->visits - 1, 0.5);
            }
            else {
                Node *next = trainer->get_next(cur, i);
                u = -1 * cur_node-> / next->evaluation +
                    cur_node->probabilities[i] * pow(cur_node->visits - 1, 0.5) / (next->visits + 1);
            }
            // We assume there are no ties
            if u > max_value {
                max_value = u;
                move_choice = i;
            }
        }
    }

    return move_choice;

}

bool search(float evaluation, float probabilities[NUM_TOTAL_MOVES], float dirichlet_nosie[NUM_MOVES]) {

    // We need to to figure out how to map the probabilities
    
    // Now, apply the legal move filter
    // Legal moves should be found at node creation
    // There is no point in doing it lazily (?)
    // meanwhile, keep track of the total sum so we can normalize afterwards;
    float sum = 0;
    for (uint8 i = 0; i < NUM_MOVES; ++i) {
        if (!(cur_node->legal_moves.get(i))) {
            cur_node->probabilities[i] = 0;
        }
        else {
            sum += cur_node->probabilities[i];
        }
    }

    // Normalize probabilities and apply dirichlet noise
    for (uint8 i = 0; i < NUM_MOVES; ++i) {
        // Only application of dirichlet noise needs the if
        // Could we separate these? The number of branches is the same though
        if (cur_node->probabilities[i] > 0) {
            // is it faster to do them together? Then there is no in-place operations
            cur_node->probabilities[i] /= sum;
            cur_node->probabilities[i] += dirichlet_noise[i];
        }
    }

    // Propagate evaluation
    float cur_evaluation = evaluation * pow(-1, to_play);
    while (cur_node != nullptr) {
        cur_node->evaluation += cur_evaluation;
        cur_evaluation *= -1;
        cur = cur_node->parent;
        cur_node = trainer->get_node(cur_node->parent);
    }

    bool need_evaluation = false;
    while (!need_evaluation && iterations_done < max_iterations) {
        ++iterations_done;
        while (true) {

            uint8 move_choice = choose_next();

            // Exploring a new node
            if (!(cur_node->visited.get(move_choice))) {
                // Create the new node
                uint32 next = trainer->place_next(cur_node->game, cur_node->depth, cur_node->parent, move_choice);
                cur_node = cur_node->children[move_choice];
                cur_node->game.do_move(move_choice);
                cur_node->game.get_legal_moves(*(cur_node->legal_moves));
                break;
            }
            // Otherwise, move down normally
            else {
                ++(cur_node->visits);
                cur_node = cur_node->children[move_choice];
            }
        }
        // Check for terminal state, otherwise evaluation is needed
        if (cur_node->legal_moves->none()) {
            // Don't propagate if value is 0
            if (cur_node->game.outcome != 0) {
                // Propagate evaluation
                float cur_evaluation = cur_node->game.outcome;
                while (cur_node != nullptr) {
                    cur_node->evaluation += cur_evaluation;
                    // Increment visit count when the evaluation from the visit is received
                    ++(cur_node->visits);
                    cur_evaluation *= -1;
                    cur_node = cur_node->parent;
                }
            }
        }
        // Otherwise, request an evaluation
        else {
            // Game state should be written into memoryview/pointer location here
            need_evaluation = true;
        }
    }

    // If true, self player should request an evaluation
    // If false, number of searches is reached and move can be chosen
    return need_evaluation;

}

// Choose the best move once searches are done
int choose_move() {

    // Choose weighted random
    if (root->depth < 4 && !testing) {
        int total = 0;
        for (int i = 0; i < LEGAL_MOVE_NUM; ++i) {
            if (root->children[i]) total += root->children[i]->visits;
        }
        int id = rand() % total;
        for (int i = 0; i < LEGAL_MOVE_NUM; ++i) {
            if (root->children[i]) {
                id -= root->children[i]->visits;
                if (id <= 0) return i;
            }
        }
        return LEGAL_MOVE_NUM - 1;
    }
    // Otherwise, choose randomly between the moves with the most visits/searches
    // Random offset is the easiest way to randomly break ties
    int id = rand() % LEGAL_MOVE_NUM, max_visits = 0, move_choice = -1;
    for (int i = 0; i < LEGAL_MOVE_NUM; ++i) {
        Node *cur_child = root->children[(id + i) % LEGAL_MOVE_NUM].get();
        if (cur_child && cur_child->visits > max_visits) {
            max_visists = cur_child->visits;
            move_choice = (id + i) % LEGAL_MOVE_NUM;
        }
    }

    return move_choice;
}

// Move the tree down a level
// Used when opponent moves
void receive_opp_move(int move_choice) {

    // Unexplored state (this should really only happen for the first move)
    if (!root->children[move_choice]) {
        // Create the new node
        root = shared_ptr<Node>(new Node{root->game, root->depth+1, nullptr});
        cur_node->game.do_move_and_rotation(move_choice);
        cur_node->game.get_legal_moves(*(cur_node->legal_moves)); // Can we delay finding legal moves?
    }
    else {
        root = root->children[move_choice];
    }
}