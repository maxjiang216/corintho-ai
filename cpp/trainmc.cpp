#include "trainmc.h"
#include "move.h"
#include "node.h"
#include <array>
#include <bitset>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
using std::cerr;

using std::bitset;

TrainMC::TrainMC(std::mt19937 *generator)
    : root{nullptr}, cur{nullptr},
      eval_index{0}, searched{std::vector<Node *>()},
      iterations_done{0}, testing{false}, generator{generator} {
  searched.reserve(searches_per_eval);
}

TrainMC::TrainMC(std::mt19937 *generator, bool)
    : root{nullptr}, cur{nullptr},
      eval_index{0}, searched{std::vector<Node *>()},
      iterations_done{0}, testing{true}, generator{generator} {
  searched.reserve(searches_per_eval);
}

void TrainMC::do_first_iteration(float game_state[GAME_STATE_SIZE]) {

  // Create the root node
  root = new Node();
  cur = root;

  cur->write_game_state(game_state);
  searched.push_back(cur);
}

void TrainMC::do_first_iteration(const Game &game, uintf depth,
                                 float game_state[GAME_STATE_SIZE]) {

  // Create the root node
  root = new Node(game, depth);
  cur = root;

  cur->write_game_state(game_state);
  searched.push_back(cur);
}

bool TrainMC::do_iteration(float evaluation[], float probabilities[],
                           float game_state[]) {
  receive_evaluation(evaluation, probabilities);
  while (eval_index < searches_per_eval) {
    // We are done the search prematurely if
    // The root has no more children to search
    // Or if we have done max_iterations searches
    bool done_search = search(game_state);
    if (done_search)
      break;
  }
  // Return whether the turn is done
  return iterations_done == max_iterations;
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
    probability_sample[cur_child->child_num] =
        (float)cur_child->visits * denominator;
    cur_child = cur_child->next_sibling;
  }

  uintf move_choice = 0;
  // Keep track of the previous sibling of the chosen node
  // For when we delete deprecated nodes
  Node *best_prev_node = nullptr;

  // In training, choose weighted random
  // For the first few moves
  if (root->depth < NUM_OPENING_MOVES && !testing) {
    uintf total = 0, target = (*generator)() % (root->visits - 1);
    Node *cur_child = root->first_child;
    // This loop will always break out
    // There is always at least one child
    while (true) {
      total += cur_child->visits;
      if (total > target) {
        move_choice = cur_child->child_num;
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
    Node *cur_child = root->first_child, *prev_node = nullptr;
    while (cur_child != nullptr) {
      if (cur_child->visits > max_visits ||
          (cur_child->visits == max_visits &&
           cur_child->evaluation > max_eval)) {
        move_choice = cur_child->child_num;
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

  Node *prev_node = nullptr, *cur_child = root->first_child;
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
                          float new_epsilon, uintf new_searches_per_eval) {
  max_iterations = new_max_iterations;
  c_puct = new_c_puct;
  epsilon = new_epsilon;
  searches_per_eval = new_searches_per_eval;
}

void TrainMC::receive_evaluation(float evaluation[], float probabilities[]) {

  for (uintf j = 0; j < searched.size(); ++j) {

    cur = searched[j];

    // Apply the legal move filter
    // Legal moves can be deduced from edges
    // We should find the legal moves when the node is created
    // Since we want to avoid evaluating the node if it is terminal
    uintf edge_index = 0;
    float sum = 0.0, filtered_probs[cur->num_legal_moves];
    for (uintf i = 0; i < NUM_MOVES; ++i) {
      if (edge_index < cur->num_legal_moves &&
          cur->edges[edge_index].move_id == i) {
        filtered_probs[edge_index] = probabilities[NUM_MOVES * j + i];
        sum += filtered_probs[edge_index];
        ++edge_index;
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
      dirichlet_noise[i] = gamma_samples[(*generator)() % GAMMA_BUCKETS];
      sum += dirichlet_noise[i];
    }
    float dirichlet_scalar = 1.0 / sum * epsilon;

    // Weighted average of probabilities and Dirichlet noise
    float weighted_probabilities[cur->num_legal_moves], max_probability = 0.0;
    edge_index = 0;
    for (uintf i = 0; i < cur->num_legal_moves; ++i) {
      // Make all legal moves have positive probability
      weighted_probabilities[i] =
          filtered_probs[i] * scalar + dirichlet_noise[i] * dirichlet_scalar;
      max_probability = std::max(weighted_probabilities[i], max_probability);
    }

    // Compute final scaled probabilities
    float denominator = Node::MAX_PROBABILITY / max_probability;
    uintf final_sum = 0;
    for (uintf i = 0; i < cur->num_legal_moves; ++i) {
      // Make all probabilities positive
      cur->edges[i].probability = std::max(
          (long int)1, lround(weighted_probabilities[i] * denominator));
      final_sum += cur->edges[i].probability;
    }

    // Record scalar for node
    cur->denominator = 1.0 / (float)final_sum;

    // Propagate evaluation
    float cur_eval = evaluation[j];
    while (cur->parent != nullptr) {
      cur->evaluation += cur_eval;
      // Reset this marker
      cur->all_visited = false;
      cur_eval *= -1.0;
      cur = cur->parent;
    }
    // Propagate to root
    cur->evaluation += cur_eval;
  }
  root->all_visited = false;
  eval_index = 0;
  searched.clear();
}

bool TrainMC::search(float game_state[]) {

  bool need_evaluation = false;

  while (!need_evaluation && iterations_done < max_iterations) {

    cur = root;

    ++iterations_done;

    while (!cur->is_terminal()) {
      // Choose next

      // Random value lower than -1.0
      float max_value = -2.0;
      // Initialize this variable to be safe
      uintf move_choice = 0;

      Node *cur_child = cur->first_child;
      uintf edge_index = 0;
      // Keep track of previous node to insert into linked list
      Node *prev_node = nullptr, *best_prev_node = nullptr;
      // Factor this value out, as it is expense to compute
      float v_sqrt = sqrt((float)cur->visits - 1.0);
      // Loop through existing children
      while (cur_child != nullptr) {
        if (!cur_child->all_visited) {
          // Random value lower than -1.0
          float u = -2.0;
          if (cur_child->child_num == cur->edges[edge_index].move_id) {
            u = -1.0 * cur_child->evaluation / (float)cur_child->visits +
                c_puct * cur->get_probability(edge_index) * v_sqrt /
                    ((float)cur_child->visits + 1.0);
            prev_node = cur_child;
            cur_child = cur_child->next_sibling;
          } else {
            u = c_puct * cur->get_probability(edge_index) * v_sqrt;
          }
          if (u > max_value) {
            // This is the previous node unless it is a match
            best_prev_node = prev_node;
            max_value = u;
            move_choice = cur->edges[edge_index].move_id;
          }
        } else {
          prev_node = cur_child;
          cur_child = cur_child->next_sibling;
        }
        ++edge_index;
      }
      while (edge_index < cur->num_legal_moves) {
        float u = c_puct * cur->get_probability(edge_index) * v_sqrt;
        if (u > max_value) {
          best_prev_node =
              prev_node; // This should always be the last node in the list
          max_value = u;
          move_choice = cur->edges[edge_index].move_id;
        }
        ++edge_index;
      }
      // Mark the visit
      // If we do it up front we only do it in one place
      ++cur->visits;

      // No nodes are available
      if (max_value == -2.0) {
        cur->all_visited = true;
        // If it is the root that is all searched, we should stop
        bool done = cur == root;
        // Remove the search
        // This should be rare
        // So this is the best way to do this
        // as we need to update visits within the same cycle
        while (cur->parent != nullptr) {
          --cur->visits;
          cur = cur->parent;
        }
        // Remove search from root
        --cur->visits;
        return done;
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
    if (cur->is_terminal()) {
      // Don't propagate if value is 0
      if (cur->result != RESULT_DRAW) {
        // Propagate evaluation
        // In a decisive terminal state, the person to play is always the
        // loser
        float cur_eval = -1.0;
        while (cur->parent != nullptr) {
          cur->evaluation += cur_eval;
          cur_eval *= -1.0;
          cur = cur->parent;
        }
        cur->evaluation += cur_eval;
      }
    }
    // Otherwise, request an evaluation
    else {
      need_evaluation = true;
      // Write game in offset position
      cur->write_game_state(game_state + eval_index * GAME_STATE_SIZE);
      // Record the node
      searched.push_back(cur);
      ++eval_index;
    }
  }
  // Reset cur for next search
  cur = root;

  return iterations_done == max_iterations;
}

void TrainMC::move_down(Node *prev_node) {

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

  new_root->next_sibling = nullptr;
  new_root->parent = nullptr;

  delete root;
  root = new_root;
  cur = root;
  iterations_done = 0;
}

uintf TrainMC::count_nodes() const {
  if (root == nullptr)
    return 0;
  return root->count_nodes();
}