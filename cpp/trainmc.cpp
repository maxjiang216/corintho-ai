#include "trainmc.h"
#include "game.h"
#include "move.h"
#include <memory>
#include <bitset>
#include <math.h>

using std::unique_ptr;
using std::bitset;
using std::memmove;

// Only used to create the root node (starting position)
TrainMC::Node::Node(): game{Game()}, visits{0}, depth{0}, evaluation{0}, legal_moves{unique_ptr<bitset<96>>()->set()}, parent{nullptr} {

    // Get legal moves in starting position. Cannot be terminal node
    game.get_legal_moves(*legal_moves);

}

TrainMC::Node::Node(Game game, int depth, Node *parent): game{game}, visits{1}, depth{depth}, parent{parent} {}

// Used to initialize tree with root node
TrainMC::TrainMC(): root{shared_ptr(new Node{})}, iterations_done{0}, cur_node{nullptr} {}

// First search on root node
void first_search() {

    cur_node = root.get();
    ++(cur_node->visits);
    ++iterations_done;
    // Here we should write the game vector into the numpy array we use for the evaluation

}

void search(float evaluation, float &noisy_probabilities[96]) {

    // Receive move probabilities
    std::memmove(&noisy_probabilities, &probabilities, 96);

    // Propagate evaluation
    float cur_evaluation = evaluation * pow(-1, to_play);
    while (cur_node != nullptr) {
        cur_node->evaluation += cur_evaluation;
        cur_evaluation *= -1;
        cur_node = cur_node->parent;
    }

    bool need_evaluation = false;
    while (!need_evaluation && iterations_done < max_iterations) {
        ++iterations_done;
        while (true) {

            // First time searching this node
            if (cur_node->children.size() == 0) {
                cur_node->children.reserve(96);
            }

            int move_choice = choose_next();

            // Exploring a new node
            if (cur_node->children[move_choice].get() == nullptr) {
                // Create the new node
                cur_node->children[move_choice] = shared_ptr<Node>(new Node{cur_node->game, cur_node->depth+1, cur_node});
                cur_node = cur_node->children[move_choice];
                cur_node->game.do_move_and_rotation(move_choice);
                cur_node->game.get_legal_moves(*(cur_node->legal_moves));
                break;
            }
            // Otherwise, move down normally
            else {
                cur_node = cur_node->children[move_choice];
                // Increment the number of visits on this node
                ++(cur_node->visits);
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
                    cur_evaluation *= -1;
                    cur_node = cur_node->parent;
                }
            }
        }
        // Otherwise, request an evaluation
        else {
            need_evaluation = true;
        }
    }
}
