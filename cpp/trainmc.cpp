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

uintf TrainMC::choose_move(std::array<float, GAME_STATE_SIZE> &game_state,
                           std::array<float, NUM_MOVES> &probability_sample) {

  // Before moving down, read the samples from the root node
  Node *root_node = trainer->get_node(root);
  root_node->write_game_state(game_state);
  for (uintf i = 0; i < NUM_MOVES; ++i) {
    if (root_node->has_visited(i)) {
      probability_sample[i] =
          (float)trainer->get_node(trainer->find_next(root_node->get_seed(), i))
              ->get_visits() /
          ((float)root_node->get_visits() - 1.0);
    } else {
      probability_sample[i] = 0.0;
    }
  }

  uintf move_choice = 0;

  // In training, choose weighted random
  // For the first few moves
  if (root_node->get_depth() < NUM_OPENING_MOVES && !testing) {
    uintf total = 0, children_visits[NUM_MOVES];
    for (uintf i = 0; i < NUM_MOVES; ++i) {
      if (root_node->has_visited(i)) {
        children_visits[i] =
            trainer->get_node(trainer->find_next(root_node->get_seed(), i))
                ->get_visits();
        total += children_visits[i];
      }
    }
    uintf cur_total = 0, target = trainer->generate() % total;
    for (uintf i = 0; i < NUM_MOVES; ++i) {
      if (root_node->has_visited(i)) {
        cur_total += children_visits[i];
        if (cur_total >= target) {
          move_choice = i;
          break;
        }
      }
    }
  }

  else {

    // Otherwise, choose randomly between the moves with the most
    // visits/searches. Random offset is the easiest way to randomly break ties
    uintf id = trainer->generate() % NUM_MOVES, max_visits = 0;
    for (uintf i = 0; i < NUM_MOVES; ++i) {
      uintf cur_move = (id + i) % NUM_MOVES;
      if (root_node->has_visited(cur_move)) {
        uintf cur_visits =
            trainer
                ->get_node(trainer->find_next(root_node->get_seed(), cur_move))
                ->get_visits();
        if (cur_visits > max_visits) {
          max_visits = cur_visits;
          move_choice = cur_move;
        }
      }
    }
  }

  uintf old_root = root;
  root = trainer->find_next(root_node->get_seed(), move_choice);
  trainer->move_down(old_root, move_choice);
  // Set cur_node here, but this is not needed if the game is complete
  // We can try to find a better place, but it's pretty insignificant and avoid
  // possible hassle
  cur = root;
  cur_node = trainer->get_node(cur);
  // Do this to avoid clashing parents
  // All nodes with possibly overwritten parents
  // Should either be stale
  // Our have itself as its parent
  cur_node->null_parent();
  iterations_done = 0;

  return move_choice;
}

bool TrainMC::receive_opp_move(uintf move_choice,
                               float game_state[GAME_STATE_SIZE],
                               const Game &game, uintf depth) {

  // If we have visited the node before
  // Simply move the tree down
  if (trainer->get_node(root)->has_visited(move_choice)) {
    uintf old_root = root;
    root = trainer->find_next(trainer->get_node(root)->get_seed(), move_choice);
    // We should stale things afterwards now
    // Since find_next skips over stale nodes
    trainer->move_down(old_root, move_choice);
    cur = root;
    cur_node = trainer->get_node(cur);
    iterations_done = 0;
    // we don't need an evaluation
    return false;
  }

  // The node doesn't exist
  // Practically (with sufficient searches) this probably won't happen but it is
  // theoretically possible Maybe we could empirically track if it happens
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

uintf TrainMC::get_root() const { return root; }

const Game &TrainMC::get_game() const { return cur_node->get_game(); }

uintf TrainMC::get_depth() const { return cur_node->get_depth(); }

bool TrainMC::is_uninitialized() const { return cur_node == nullptr; }

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
        Node *next =
            trainer->get_node(trainer->find_next(cur_node->get_seed(), i));
        u = -1.0 * next->get_evaluation() / (float)next->get_visits() +
            c_puct * cur_node->get_probability(i) *
                sqrt((float)cur_node->get_visits() - 1.0) /
                ((float)next->get_visits() + 1);

      }
      // If not visited, set action value to 0
      else {
        u = c_puct * cur_node->get_probability(i) *
            sqrt((float)cur_node->get_visits() - 1.0);
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