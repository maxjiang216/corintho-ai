#include "trainmc.h"
#include "move.h"
#include "node.h"
#include <array>
#include <bitset>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
using std::cerr;

using std::bitset;

TrainMC::TrainMC(std::mt19937 *generator)
    : iterations_done{0}, testing{false}, generator{generator} {}

TrainMC::TrainMC(Trainer *trainer, bool)
    : iterations_done{0}, testing{true}, generator{generator} {}

void TrainMC::do_first_iteration(float game_state[GAME_STATE_SIZE]) {

  // Create the root node
  root = new Node();
  cur = root;

  cur->write_game_state(game_state);
}

bool TrainMC::do_iteration(float evaluation, float probabilities[NUM_MOVES],
                           float game_state[GAME_STATE_SIZE]) {
  receive_evaluation(evaluation, probabilities);
  return search(game_state);
}

uintf TrainMC::choose_move(float game_state[GAME_STATE_SIZE],
                           float probability_sample[NUM_MOVES]) {

  // Before moving down, read the samples from the root node
  root->write_game_state(game_state);
  // Clear probabilities first
  // Most moves are not legal
  std::memset(probability_sample, 0, NUM_MOVES * sizeof(float));
  Node *cur_child = cur->first_child;
  float denominator = 1.0 / ((float)root->visits - 1.0);
  while (cur_child != nullptr) {
    probability_sample[i] = (float)cur_child->visits * denominator;
  }

  uintf move_choice = 0;
  // Keep track of the previous sibling of the chosen node
  // For when we delete deprecated nodes
  Node *best_prev_node = nullptr;

  // In training, choose weighted random
  // For the first few moves
  if (root->depth < NUM_OPENING_MOVES && !testing) {
    uintf total = 0, target = generator->() % (cur->visits - 1);
    Node *cur_child = cur->first_child;
    // This loop will always break out
    // There is always at least one child
    while (true) {
      total += cur_child->visits;
      if (total >= target) {
        move_choice = i;
        break;
      }
      best_prev_node = cur_child;
      cur_child = cur_child->next_sibling;
    }
  }

  else {

    // Otherwise, choose the move with the most searches
    // Breaking ties with evaluation
    uintf max_visits = 0;
    float max_eval = 0.0;
    Node *cur_child = cur->first_child, prev_node = nullptr;
    while (cur_child != nullptr) {
      if (cur_child->visits > max_visits ||
          cur_child->visits == max_visits && cur_child->evaluation > max_eval) {
        move_choice = cur_child->move_id;
        best_prev_node = prev_node;
        max_visits = cur_child->visits;
        max_eval = cur_child->evaluation;
      }
      prev_node = cur_child;
      cur_child = cur_child->next_sibling;
    }
  }

  move_down(best_prev_node);

  return move_choice;
}

bool TrainMC::receive_opp_move(uintf move_choice,
                               float game_state[GAME_STATE_SIZE],
                               const Game &game, uintf depth) {

  Node *prev_node = nullptr, cur_child = root->first_child;
  while (cur_child != nullptr) {
    if (cur_child->child_num == move_choice) {
      move_down(prev_node);
      // we don't need an evaluation
      return false;
    }
    prev_node = cur_child;
    cur_child = cur_child->next_sibling;
  }

  // The node doesn't exist, delete tree
  delete root;
  // Copy opponent game state into our root
  root = new Node(game, depth);
  cur = root;
  // We need an evaluation
  cur->write_game_state(game_state);
  // this is the first iteration of the turn
  iterations_done = 1;
  // we need an evaluation
  return true;
}

const Game &TrainMC::get_game() const { return root->game; }

bool TrainMC::is_uninitialized() const { return root == nullptr; }

void TrainMC::set_statics(uintf new_max_iterations, float new_c_puct,
                          float new_epsilon) {
  max_iterations = new_max_iterations;
  c_puct = new_c_puct;
  epsilon = new_epsilon;
}

void TrainMC::receive_evaluation(float evaluation,
                                 const float probabilities[NUM_MOVES]) {

  // Apply the legal move filter
  // Legal moves can be deduced from edges
  // We should find the legal moves when the node is created
  // Since we want to avoid evaluating the node if it is terminal
  uintf edge_index = 0;
  float sum = 0.0;
  for (uintf i = 0; i < NUM_MOVES; ++i) {
    // Don't make these 0
    if (cur->edges[edge_index]->move_id == i) {
      sum += probabilities[i];
      ++edge_index;
    } else {
      probabilities[i] = 0.0;
    }
  }

  // Multiplying by this is more efficient
  float scalar = 1.0 / sum * (1 - epsilon);

  // Generate Dirichlet noise (approximation)
  // We cannot generate this at node creation as we don't store floats in the
  // node
  sum = 0.0;
  float dirichlet_noise[cur->num_legal_moves];

  for (uintf i = 0; i < cur->num_legal_moves; ++i) {
    dirichlet_noise[i] = gamma_samples[generator->() % GAMMA_BUCKETS];
    sum += dirichlet_noise[i];
  }
  float dirichlet_scalar = 1.0 / sum * epsilon;

  // Weighted average of probabilities and Dirichlet noise
  float weighted_probabilities[cur->num_legal_moves], max_probability = 0.0;
  edge_index = 0;
  for (uintf i = 0; i < NUM_MOVES; ++i) {
    if (cur->edges[edge_index]->move_id == i) {
      // Make all legal moves have positive probability
      weighted_probabilities[edge_index] =
          probabilities[i] * scalar +
          dirichlet_noise[edge_index] * dirichlet_scalar;
      max_probability =
          std::max(weighted_probabilities[edge_index], max_probability);
      ++edge_index;
    }
  }

  // Compute final scaled probabilities
  float denominator = Node.MAX_PROBABILITY / max_probability;
  uintf final_sum = 0;
  for (uintf i = 0; i < cur->num_legal_moves; ++i) {
    // Make all probabilities positive
    cur->edges[i]->probability =
        std::max(1, lround(weighted_probabilities * denominator));
    final_sum += cur->edges[i]->probability;
  }

  // Record scalar for node
  cur->denominator = 1.0 / (float)final_sum;

  // Propagate evaluation
  float cur_eval = evaluation;
  while (cur->parent != nullptr) {
    ++cur->visits;
    cur->evaluation += cur_eval;
    cur_eval *= -1.0;
    cur = cur->parent;
  }
  // Propagate to root
  ++cur->visits;
  cur->evaluation += cur_eval;
}

bool TrainMC::search(float game_state[GAME_STATE_SIZE]) {

  bool need_evaluation = false;

  while (!need_evaluation && iterations_done < max_iterations) {

    ++iterations_done;

    while (!cur->is_terminal()) {
      // Choose next

      // Random value lower than -1.0
      float max_value = -2.0;
      // Initialize this variable to be safe
      uintf move_choice = 0;

      uintf cur_child = cur->first_child, edge_index = 0;
      // Keep track of previous node to insert into linked list
      Node *prev_node = nullptr, *best_prev_node = nullptr;
      // Loop through existing children
      while (cur_child != nullptr) {
        float u;
        if (cur_child->child_num == edge_index) {
          u = -1.0 * cur_child->evaluation / (float)cur_child->visits +
              c_puct * cur->get_probability(edge_index) *
                  sqrt((float)cur->visits - 1.0) /
                  ((float)cur_child->visits + 1.0);
          prev_node = cur_child;
          cur_child = cur_child->next_sibling;
        } else {
          u = c_puct * cur->get_probability(edge_index) *
              sqrt((float)cur->visits - 1.0);
        }
        if (u > max_value) {
          best_prev_node = prev_node;
          max_value = u;
          move_choice = edge_index;
        }
        ++edge_index;
      }
      while (edge_index < cur->num_legal_moves) {
        float u = c_puct * cur->get_probability(edge_index) *
                  sqrt((float)cur->visits - 1.0);
        if (u > max_value) {
          best_prev_node =
              prev_node; // This should always be the last node in the list
          max_value = u;
          move_choice = edge_index;
        }
        ++edge_index;
      }

      // Best node should be inserted at the beginning of the list
      if (best_prev_node == nullptr) {
        cur->first_child = new Node(cur->game, cur->depth + 1, cur,
                                    cur->first_child, move_choice);
        cur = cur->first_child;
        break;
      }
      // New node somewhere else in the list
      else if (best_prev_node->child_num != move_choice) {
        best_prev_node->next_sibling =
            new Node(cur->game, cur->depth + 1, cur,
                     best_prev_node->next_sibling, move_choice);
        cur = best_prev_node->next_sibling;
        break;
      }
      // Existing node, continue searching
      else {
        cur = best_prev_node;
      }
    }

    // Check for terminal state, otherwise evaluation is needed
    if (cur_node->is_terminal()) {
      // Don't propagate if value is 0
      if (cur_node->get_result() != DRAW) {
        // Propagate evaluation
        // In a decisive terminal state, the person to play is always the
        // loser
        float cur_eval = -1.0;
        while (cur->parent != nullptr) {
          ++cur->visits;
          cur->evaluation += cur_eval;
          cur_eval *= -1.0;
          cur = cur->parent;
        }
        ++cur->visits;
        cur->evaluation += cur_eval;

      } else {
        // We need to add to the visit count
        while (cur->parent != nullptr) {
          ++cur->visits;
          cur = cur->parent;
        }
        ++cur->visits;
      }
    }
    // Otherwise, request an evaluation
    else {
      cur->write_game_state(game_state);
      need_evaluation = true;
    }
  }

  return need_evaluation;
}

void Trainer::move_down(Node *prev_node) {

  // Extricate the node we want
  Node *new_root;
  // Chosen node is first child
  if (prev_node == nullptr) {
    new_root = root->first_child;
    root->first_child = new_root->next_sibling;
  } else {
    new_root = prev_node->next_sibling;
    prev_node->next_sibling = new_root->next_sibling;
  }

  delete root;
  root = new_root;
  cur = root;
  iterations_done = 0;
}