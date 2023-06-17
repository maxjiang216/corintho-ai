#include "trainmc.h"

#include <array>
#include <bitset>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "move.h"
#include "node.h"
using std::cerr;

using std::bitset;

TrainMC::TrainMC(std::mt19937 *generator)
    : root{nullptr}, cur{nullptr},
      eval_index{0}, searched{std::vector<Node *>()}, to_eval{nullptr},
      iterations_done{0}, testing{false}, generator{generator} {
  searched.reserve(searches_per_eval);
}

TrainMC::TrainMC(std::mt19937 *generator, bool)
    : root{nullptr}, cur{nullptr},
      eval_index{0}, searched{std::vector<Node *>()}, to_eval{nullptr},
      iterations_done{0}, testing{true}, generator{generator} {
  searched.reserve(searches_per_eval);
}

void TrainMC::do_first_iteration() {
  // Create the root node
  root = new Node();
  cur = root;
  iterations_done = 1;

  cur->write_game_state(to_eval);
  searched.push_back(cur);
  eval_index = 1;
}

void TrainMC::do_first_iteration(const Game &game, uintf depth) {
  // Create the root node
  root = new Node(game, depth);
  cur = root;
  iterations_done = 1;

  cur->write_game_state(to_eval);
  searched.push_back(cur);
  eval_index = 1;
}

bool TrainMC::do_iteration(float evaluation[], float probabilities[]) {
  receive_evaluation(evaluation, probabilities);
  while (eval_index < searches_per_eval && root->result == RESULT_NONE) {
    // We are done the search prematurely if
    // The root has no more children to search
    // Or if we have done max_iterations searches
    bool done_search = search();
    if (done_search)
      break;
  }
  // Return whether the turn is done
  return iterations_done == max_iterations || root->result != RESULT_NONE;
}

uintf TrainMC::choose_move(float game_state[GAME_STATE_SIZE],
                           float probability_sample[NUM_MOVES]) {
  if (!testing) {
    // Before moving down, read the samples from the root node
    root->write_game_state(game_state);
    // Clear probabilities first
    // Most moves are not legal
    std::memset(probability_sample, 0, NUM_MOVES * sizeof(float));
  }

  uintf move_choice = 0;
  // Keep track of the previous sibling of the chosen node
  // For when we delete deprecated nodes
  Node *best_prev_node = nullptr;

  // Mating sequence available, find the first winning move
  // There should only ever be 1 winning move
  // Since after it is found, the node is not searched again for the most part
  // Ignore opening stuff here
  if (root->result == DEDUCED_WIN) {
    Node *cur_child = root->first_child, *prev_node = nullptr;
    while (cur_child != nullptr) {
      if (cur_child->result == DEDUCED_LOSS ||
          cur_child->result == RESULT_LOSS) {
        move_choice = cur_child->child_num;
        best_prev_node = prev_node;
        break;
      }
      prev_node = cur_child;
      cur_child = cur_child->next_sibling;
    }
    probability_sample[move_choice] = 1.0;
  }

  // Losing position, find the move with the most searches
  // The most searched line is likely the one with the longest and/or hardest to
  // find mate which is pratically better. Ignore opening stuff
  else if (root->result == DEDUCED_LOSS) {
    uintf max_visits = 0;
    Node *cur_child = root->first_child, *prev_node = nullptr;
    while (cur_child != nullptr) {
      if (cur_child->visits > max_visits) {
        move_choice = cur_child->child_num;
        best_prev_node = prev_node;
        max_visits = cur_child->visits;
      }
      prev_node = cur_child;
      cur_child = cur_child->next_sibling;
    }
    probability_sample[move_choice] = 1.0;
  }

  // Drawn position, find the drawing move with the most searches
  // Which again is practically more likely to get winning chances
  // Against a suboptimal opponent
  // Ignore opening stuff
  else if (root->result == DEDUCED_DRAW) {
    uintf max_visits = 0;
    Node *cur_child = root->first_child, *prev_node = nullptr;
    while (cur_child != nullptr) {
      if (cur_child->visits > max_visits && cur_child->result != DEDUCED_WIN) {
        move_choice = cur_child->child_num;
        best_prev_node = prev_node;
        max_visits = cur_child->visits;
      }
      prev_node = cur_child;
      cur_child = cur_child->next_sibling;
    }
    probability_sample[move_choice] = 1.0;
  }

  // In training, choose weighted random
  // For the first few moves
  else if (root->depth < NUM_OPENING_MOVES && !testing) {
    // We exclude losing moves from our choices
    uintf visits = 0;
    Node *cur_child = root->first_child;
    while (cur_child != nullptr) {
      if (cur_child->result != DEDUCED_WIN) {
        visits += cur_child->visits;
      }
      cur_child = cur_child->next_sibling;
    }
    uintf total = 0, target = (*generator)() % visits;
    float denominator = 1.0 / (float)visits;
    cur_child = root->first_child;
    // This loop will always break out
    // There is always at least one child
    while (true) {
      if (cur_child->result != DEDUCED_WIN) {
        total += cur_child->visits;
        probability_sample[cur_child->child_num] =
            (float)cur_child->visits * denominator;
        if (total > target) {
          move_choice = cur_child->child_num;
          break;
        }
      }
      best_prev_node = cur_child;
      cur_child = cur_child->next_sibling;
    }
    cur_child = cur_child->next_sibling;
    while (cur_child != nullptr) {
      if (cur_child->result != DEDUCED_WIN) {
        probability_sample[cur_child->child_num] =
            (float)cur_child->visits * denominator;
      }
      cur_child = cur_child->next_sibling;
    }
  }

  else {
    // Otherwise, choose the move with the most searches
    // Breaking ties with evaluation
    // We never choose losing moves
    // And treat draws as having evaluation 0
    uintf max_visits = 0;
    float max_eval = 0.0, eval;
    Node *cur_child = root->first_child, *prev_node = nullptr;
    while (cur_child != nullptr) {
      if (cur_child->result != DEDUCED_WIN) {
        eval = cur_child->evaluation;
        if (cur_child->result == RESULT_DRAW ||
            cur_child->result == DEDUCED_DRAW) {
          eval = 0.0;
        }
        if (cur_child->visits > max_visits ||
            (cur_child->visits == max_visits && eval > max_eval)) {
          move_choice = cur_child->child_num;
          best_prev_node = prev_node;
          max_visits = cur_child->visits;
          max_eval = eval;
        }
      }
      prev_node = cur_child;
      cur_child = cur_child->next_sibling;
    }
    probability_sample[move_choice] = 1.0;
  }

  move_down(best_prev_node);

  return move_choice;
}

bool TrainMC::receive_opp_move(uintf move_choice, const Game &game,
                               uintf depth) {
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
  cur->write_game_state(to_eval);
  searched.push_back(cur);
  // this is the first iteration of the turn
  iterations_done = 1;
  eval_index = 1;
  // we need an evaluation
  return true;
}

const Game &TrainMC::get_game() const {
  return root->game;
}

bool TrainMC::is_uninitialized() const {
  return root == nullptr;
}

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
      // Correct default +1 evaluation
      cur->evaluation += cur_eval - 1.0;
      // Reset this marker
      cur->all_visited = false;
      cur_eval *= -1.0;
      cur = cur->parent;
    }
    // Propagate to root
    cur->evaluation += cur_eval - 1.0;
  }
  root->all_visited = false;
  eval_index = 0;
  searched.clear();
}

bool TrainMC::search() {
  bool need_evaluation = false;

  while (!need_evaluation && iterations_done < max_iterations &&
         root->result == RESULT_NONE) {
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
      float v_sqrt = c_puct * sqrt((float)cur->visits);
      // Loop through existing children
      while (cur_child != nullptr) {
        // Random value lower than -1.0
        float u = -2.0;
        if (cur_child->child_num == cur->edges[edge_index].move_id) {
          // Don't visit all_visited nodes
          if (!cur_child->all_visited && (cur_child->result == RESULT_NONE ||
                                          cur_child->result == RESULT_DRAW ||
                                          cur_child->result == DEDUCED_DRAW)) {
            // Known draw, use evaluation 0
            if (cur_child->result != RESULT_NONE) {
              u = cur->get_probability(edge_index) * v_sqrt;
            } else {
              u = -1.0 * cur_child->evaluation / (float)cur_child->visits +
                  cur->get_probability(edge_index) * v_sqrt /
                      ((float)cur_child->visits + 1.0);
            }
          }
          prev_node = cur_child;
          cur_child = cur_child->next_sibling;
        } else {
          u = cur->get_probability(edge_index) * v_sqrt;
        }
        if (u > max_value) {
          // This is the previous node unless it is a match
          best_prev_node = prev_node;
          max_value = u;
          move_choice = cur->edges[edge_index].move_id;
        }
        ++edge_index;
      }
      while (edge_index < cur->num_legal_moves) {
        float u = cur->get_probability(edge_index) * v_sqrt;
        if (u > max_value) {
          best_prev_node =
              prev_node;  // This should always be the last node in the list
          max_value = u;
          move_choice = cur->edges[edge_index].move_id;
        }
        ++edge_index;
      }
      // Count the visit
      ++cur->visits;
      // Default eval is +1 for all positions
      // This prevents repeatedly choosing the same line
      cur->evaluation += 1.0;

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
          cur->evaluation -= 1.0;
          cur = cur->parent;
        }
        // Remove search from root
        --cur->visits;
        cur->evaluation -= 1.0;
        --iterations_done;
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
      // propagate the result
      // new deductions can only be made after a new terminal node is found
      propagate_result();
      // Count the visit
      ++cur->visits;
      // In a decisive terminal state, the person to play is always the
      // loser
      float cur_eval = -1.0;
      if (cur->result == RESULT_DRAW) {
        cur_eval = 0.0;
      }
      cur->evaluation = cur_eval;
      while (cur->parent != nullptr) {
        cur = cur->parent;
        // Correct default +1 evaluation
        cur->evaluation += cur_eval - 1.0;
        cur_eval *= -1.0;
      }
    }
    // Otherwise, request an evaluation
    else {
      // Default +1 evaluation
      cur->evaluation = 1.0;
      need_evaluation = true;
      // Write game in offset position
      cur->write_game_state(to_eval + eval_index * GAME_STATE_SIZE);
      // Record the node
      searched.push_back(cur);
      ++eval_index;
    }
  }
  // Reset cur for next search
  cur = root;

  return iterations_done == max_iterations || root->result != RESULT_NONE;
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

void TrainMC::propagate_result() {
  Node *node = cur, *cur_node;

  while (node != root) {
    // We only need one loss to deduce
    if (node->result == RESULT_LOSS || node->result == DEDUCED_LOSS) {
      node = node->parent;
      node->result = DEDUCED_WIN;
    } else {
      // There are no winning moves
      node = node->parent;
      cur_node = node->first_child;
      // Position has a drawing move
      bool has_draw = false;
      uintf edge_index = 0;
      while (cur_node != nullptr) {
        // We can deduce no further (not visited or not known)
        if (cur_node->child_num != node->edges[edge_index].move_id ||
            cur_node->result == RESULT_NONE) {
          return;
        }
        if (cur_node->result == RESULT_DRAW ||
            cur_node->result == DEDUCED_DRAW) {
          has_draw = true;
        }
        cur_node = cur_node->next_sibling;
        ++edge_index;
      }
      // Unvisited moves at the end
      if (edge_index < node->num_legal_moves)
        return;
      // If we reach this part, we are guaranteed
      // No winning moves and
      // no unknown moves
      // If there are any drawing moves, the position is a draw
      if (has_draw) {
        node->result = DEDUCED_DRAW;
      }
      // Otherwise, there are only losing moves, so the position is a loss
      else {
        node->result = DEDUCED_LOSS;
      }
    }
  }
}