#include "trainmc.h"

#include <cstdint>
#include <cmath>
#include <cstring>
#include <cassert>

#include <fstream>
#include <random>
#include <vector>

#include "move.h"
#include "node.h"

TrainMC::TrainMC(std::mt19937 *generator, int32_t max_searches, int32_t searches_per_eval, float c_puct, float epsilon, bool testing):
max_searches_{max_searches}, searches_per_eval_{searches_per_eval}, c_puct_{c_puct}, epsilon_{epsilon}, testing_{testing}, generator_{generator} {
  assert(max_searches_ > 0);
  assert(searches_per_eval_ > 0);
  assert(c_puct_ > 0.0);
  assert(epsilon_ >= 0.0 && epsilon_ <= 1.0);
  assert(generator_ != nullptr);
  // seached_ will only ever need this many elements
  searched_.reserve(searches_per_eval_);
}

Node *TrainMC::root() const noexcept {
  return root_;
}

int32_t TrainMC::root_depth() const noexcept {
  assert(root_ != nullptr);
  return root_->depth();
}

const Game& TrainMC::get_root_game() const noexcept {
  assert(root_ != nullptr);
  return root_->get_game();
}

bool TrainMC::noEvalsRequested() const noexcept {
  return searched_index_ == 0;
}

int32_t TrainMC::numNodesSearched() const noexcept {
  return searches_done_;
}

bool TrainMC::isUninitialized() const noexcept {
  return root_ == nullptr;
}

int32_t TrainMC::numNodes() const noexcept {
  if (root_ == nullptr) {
    return 0;
  }
  return root_->countNodes();
}

void TrainMC::createRoot(const Game &game, int32_t depth) {
  assert(root_ == nullptr);
  root_ = new Node(game, depth);
  cur_ = root_;
}

bool TrainMC::doIteration(float eval[], float probs[]) {
  // This is the first iteration of a game
  if (isUninitialized()) {
    // We should not be getting evaluations
    assert(eval == nullptr);
    assert(probs == nullptr);
    // Initialize the Monte Carlo search tree
    root_ = new Node();
    cur_ = root_;
    // "Search" the root node
    searches_done_ = 1;
    // Request an evaluation
    // The result is not deduced at this point
    cur_->writeGameState(to_eval_);
    searched_.push_back(cur_);
    searched_index_ = 1;
    return false;
  }
  // Otherwise, we should have evaluations
  assert(eval != nullptr);
  assert(probs != nullptr);
  receiveEval(eval, probs);
  while (searches_done_ < max_searches_ && !root_->known()) {
    // search returns if we should continue searching
    if (search()) {
      break;
    }
  }
  assert(searches_done_ <= max_searches_);
  return searches_done_ == max_searches_ || root_->known();
}

int32_t TrainMC::chooseMove(float game_state[kGameStateSize], float prob_sample[kNumMoves]) noexcept {
  assert(game_state != nullptr);
  assert(prob_sample != nullptr);
  assert(!isUninitialized());
  if (!testing_) {  // Write samples if we are not testing
    // Before moving down, read the samples from the root node
    root_->writeGameState(game_state);
    // Clear the probability sample
    // We will write the probabilities when we determine which move to do
    std::memset(prob_sample, 0, kNumMoves * sizeof(float));
  }
  int32_t choice = 0;
  // Keep track of the previous sibling of the chosen node
  // For when we delete deprecated nodes
  Node *best_prev = nullptr;
  // If the root is a deduced win, find the first winning move
  // There should only ever be 1 winning move, since after it is found,
  // the node is not searched again
  // Temperature is 0 in this case, even in the opening.
  if (root_->won()) {
    Node *cur = root_->first_child();
    Node *prev = nullptr;
    while (cur != nullptr) {
      // Winning moves lead to lost positions
      if (cur->lost()) {
        choice = cur->child_id();
        best_prev = prev;
        break;
      }
      prev = cur;
      cur = cur->next_sibling();
    }
    // Set the probability of the winning move to 1
    prob_sample[choice] = 1.0;

  }
  // For losing positions, find the move with the most searches
  // The most searched line is likely the one with the longest and/or hardest to
  // find mate which is practically better.
  // Temperature is 0 in this case, even in the opening.
  else if (root_->lost()) {
    int32_t max_visits = 0;
    Node *cur = root_->first_child();
    Node *prev = nullptr;
    while (cur != nullptr) {
      if (cur->visits() > max_visits) {
        choice = cur->child_id();
        best_prev = prev;
        max_visits = cur->visits();
      }
      prev = cur;
      cur = cur->next_sibling();
    }
    prob_sample[choice] = 1.0;
  }
  // In a drawn position, find the drawing move with the most searches
  // that is not a losing move.
  // This again is better practically.
  // Temperature is 0.
  else if (root_->drawn()) {
    int32_t max_visit = 0;
    Node *cur = root_->first_child();
    Node *prev = nullptr;
    while (cur != nullptr) {
      if (cur->visits() > max_visit && !cur->won()) {
        choice = cur->child_id();
        best_prev = prev;
        max_visit = cur->visits();
      }
      prev = cur;
      cur = cur->next_sibling();
    }
    prob_sample[choice] = 1.0;
  }
  // For opening moves during training, temperature is 1.
  else if (root_->depth() < kNumOpeningMoves && !testing_) {
    assert(!root_->known());
    // Count the number of visits to non-losing moves
    // Which will be the denominator for the probabilities
    int32_t visits = 0;
    Node *cur = root_->first_child();
    while (cur != nullptr) {
      // Exclude losing moves
      if (!cur->won()) {
        visits += cur->visits();
      }
      cur = cur->next_sibling();
    }
    float denominator = 1.0 / static_cast<float>(visits);
    cur = root_->first_child();
    // Get the probability sample
    while (cur != nullptr) {
      // Exclude losing moves
      if (!cur->won()) {
        prob_sample[cur->child_id()] = static_cast<float>(cur->visits()) * denominator;
      }
      best_prev = cur;
      cur = cur->next_sibling();
    }
    // Choose a random move weighted by the number of visits
    int32_t target = (*generator_)() % visits;
    int32_t total = 0;
    while (cur != nullptr) {
      // Exclude losing moves
      if (!cur->won()) {
        total += cur->visits();
        if (total > target) {
          choice = cur->child_id();
          break;
        }
      }
      best_prev = cur;
      cur = cur->next_sibling();
    }
  }
  // Otherwise, choose the move with the most searches
  // Break ties with evaluation
  // We never choose losing moves and treat draws as having evaluation 0
  else {
    assert(!root_->known());
    int32_t max_visits = 0;
    float max_eval = 0.0;
    Node *cur = root_->first_child();
    Node *prev = nullptr;
    while (cur != nullptr) {
      if (!cur->won()) {
        float eval = cur->evaluation();
        if (cur->result() == kResultDraw || cur->result() == kDeducedDraw) {
          eval = 0.0;
        }
        if (cur->visits() > max_visits || (cur->visits() == max_visits && eval > max_eval)) {
          choice = cur->child_id();
          best_prev = prev;
          max_visits = cur->visits();
          max_eval = eval;
        }
      }
      prev = cur;
      cur = cur->next_sibling();
    }
    prob_sample[choice] = 1.0;
  }

  // Move down the tree
  moveDown(best_prev);

  return choice;
}

bool TrainMC::receiveOpponentMove(uintf move_choice, const Game &game,
                               uintf depth) {
  Node *prev_node = nullptr, *cur_child = root->first_child();
  while (cur_child != nullptr) {
    if (cur_child->child_id() == move_choice) {
      moveDown(prev_node);
      // we don't need an evaluation
      return false;
    }
    prev_node = cur_child;
    cur_child = cur_child->next_sibling();
  }

  // The node doesn't exist, delete tree
  delete root;
  // Copy opponent game state into our root
  root = new Node(game, depth);
  cur = root;
  // We need an evaluation
  cur->writeGameState(to_eval_);
  searched_.push_back(cur);
  // this is the first iteration of the turn
  iterations_done = 1;
  searched_index_ = 1;
  // we need an evaluation
  return true;
}

const Game &TrainMC::get_game() const {
  return root->game();
}

bool TrainMC::is_uninitialized() const {
  return root == nullptr;
}

void TrainMC::set_statics(uintf new_max_iterations, float new_c_puct,
                          float new_epsilon, uintf new_searches_per_eval) {
  max_iterations = new_max_iterations;
  c_puct = new_c_puct;
  epsilon = new_epsilon;
  searches_per_eval_ = new_searches_per_eval;
}

void TrainMC::receiveEval(float evaluation[], float probabilities[]) {
  for (uintf j = 0; j < searched_.size(); ++j) {
    cur = searched_[j];

    // Apply the legal move filter
    // Legal moves can be deduced from edges
    // We should find the legal moves when the node is created
    // Since we want to avoid evaluating the node if it is terminal
    uintf edge_index = 0;
    float sum = 0.0, filtered_probs[cur->num_legal_moves()];
    for (uintf i = 0; i < kNumMoves; ++i) {
      if (edge_index < cur->num_legal_moves() &&
          cur->move_id(edge_index) == i) {
        filtered_probs[edge_index] = probabilities[kNumMoves * j + i];
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
    float dirichlet_noise[cur->num_legal_moves()];

    for (uintf i = 0; i < cur->num_legal_moves(); ++i) {
      dirichlet_noise[i] = gamma_samples[(*generator_)() % GAMMA_BUCKETS];
      sum += dirichlet_noise[i];
    }
    float dirichlet_scalar = 1.0 / sum * epsilon;

    // Weighted average of probabilities and Dirichlet noise
    float weighted_probabilities[cur->num_legal_moves()], max_probability = 0.0;
    edge_index = 0;
    for (uintf i = 0; i < cur->num_legal_moves(); ++i) {
      // Make all legal moves have positive probability
      weighted_probabilities[i] =
          filtered_probs[i] * scalar + dirichlet_noise[i] * dirichlet_scalar;
      max_probability = std::max(weighted_probabilities[i], max_probability);
    }

    // Compute final scaled probabilities
    float denominator = Node::kMaxProbability / max_probability;
    uintf final_sum = 0;
    for (uintf i = 0; i < cur->num_legal_moves(); ++i) {
      // Make all probabilities positive
      cur->set_probability(
          i, std::max((long int)1,
                      lround(weighted_probabilities[i] * denominator)));
      final_sum += cur->probability(i);
    }

    // Record scalar for node
    cur->set_denominator(1.0 / (float)final_sum);

    // Propagate evaluation
    float cur_eval = evaluation[j];
    while (cur->parent() != nullptr) {
      // Correct default +1 evaluation
      cur->increase_evaluation(cur_eval - 1.0);
      // Reset this marker
      cur->set_all_visited(false);
      cur_eval *= -1.0;
      cur = cur->parent();
    }
    // Propagate to root
    cur->increase_evaluation(cur_eval - 1.0);
  }
  root->set_all_visited(false);
  searched_index_ = 0;
  searched_.clear();
}

bool TrainMC::search() {
  bool need_evaluation = false;

  while (!need_evaluation && iterations_done < max_iterations &&
         root->result() == kResultNone) {
    cur = root;

    ++iterations_done;

    while (!cur->terminal()) {
      // Choose next

      // Random value lower than -1.0
      float max_value = -2.0;
      // Initialize this variable to be safe
      uintf move_choice = 0;

      Node *cur_child = cur->first_child();
      uintf edge_index = 0;
      // Keep track of previous node to insert into linked list
      Node *prev_node = nullptr, *best_prev_node = nullptr;
      // Factor this value out, as it is expense to compute
      float v_sqrt = c_puct * sqrt(static_cast<float>(cur->visits()));
      // Loop through existing children
      while (cur_child != nullptr) {
        // Random value lower than -1.0
        float u = -2.0;
        if (cur_child->child_id() == cur->move_id(edge_index)) {
          // Don't visit all_visited nodes
          if (!cur_child->all_visited() &&
              (cur_child->result() == kResultNone ||
               cur_child->result() == kResultDraw ||
               cur_child->result() == kDeducedDraw)) {
            // Known draw, use evaluation 0
            if (cur_child->result() != kResultNone) {
              u = cur->probability(edge_index) * v_sqrt;
            } else {
              u = -1.0 * cur_child->evaluation() /
                      static_cast<float>(cur_child->visits()) +
                  cur->probability(edge_index) * v_sqrt /
                      (static_cast<float>(cur_child->visits()) + 1.0);
            }
          }
          prev_node = cur_child;
          cur_child = cur_child->next_sibling();
        } else {
          u = cur->probability(edge_index) * v_sqrt;
        }
        if (u > max_value) {
          // This is the previous node unless it is a match
          best_prev_node = prev_node;
          max_value = u;
          move_choice = cur->move_id(edge_index);
        }
        ++edge_index;
      }
      while (edge_index < cur->num_legal_moves()) {
        float u = cur->probability(edge_index) * v_sqrt;
        if (u > max_value) {
          best_prev_node =
              prev_node;  // This should always be the last node in the list
          max_value = u;
          move_choice = cur->move_id(edge_index);
        }
        ++edge_index;
      }
      // Count the visit
      cur->increment_visits();
      // Default eval is +1 for all positions
      // This prevents repeatedly choosing the same line
      cur->increase_evaluation(1.0);

      // No nodes are available
      if (max_value == -2.0) {
        cur->set_all_visited();
        // If it is the root that is all searched_, we should stop
        bool done = cur == root;
        // Remove the search
        // This should be rare
        // So this is the best way to do this
        // as we need to update visits within the same cycle
        while (cur->parent() != nullptr) {
          cur->decrement_visits();
          cur->decrease_evaluation(1.0);
          cur = cur->parent();
        }
        // Remove search from root
        cur->decrement_visits();
        cur->decrease_evaluation(1.0);
        --iterations_done;
        return done;
      }
      // Best node should be inserted at the beginning of the list
      if (best_prev_node == nullptr) {
        cur->set_first_child(new Node(cur->game(), cur, cur->first_child(),
                                      move_choice, cur->depth() + 1));
        cur = cur->first_child();
        break;
      }
      // New node somewhere else in the list
      else if (best_prev_node->child_id() != move_choice) {
        best_prev_node->set_next_sibling(
            new Node(cur->game(), cur, best_prev_node->next_sibling(),
                     move_choice, cur->depth() + 1));
        cur = best_prev_node->next_sibling();
        break;
      }
      // Existing node, continue searching
      else {
        cur = best_prev_node;
      }
    }

    // Check for terminal state, otherwise evaluation is needed
    if (cur->terminal()) {
      // propagate the result
      // new deductions can only be made after a new terminal node is found
      propagateTerminal();
      // Count the visit
      cur->increment_visits();
      // In a decisive terminal state, the person to play is always the
      // loser
      float cur_eval = -1.0;
      if (cur->result() == kResultDraw) {
        cur_eval = 0.0;
      }
      cur->set_evaluation(cur_eval);
      while (cur->parent() != nullptr) {
        cur = cur->parent();
        // Correct default +1 evaluation
        cur->increase_evaluation(cur_eval - 1.0);
        cur_eval *= -1.0;
      }
    }
    // Otherwise, request an evaluation
    else {
      // Default +1 evaluation
      cur->set_evaluation(1.0);
      need_evaluation = true;
      // Write game in offset position
      cur->writeGameState(to_eval_ + searched_index_ * kGameStateSize);
      // Record the node
      searched_.push_back(cur);
      ++searched_index_;
    }
  }
  // Reset cur for next search
  cur = root;

  return iterations_done == max_iterations || root->result() != kResultNone;
}

void TrainMC::moveDown(Node *prev_node) {
  // Extricate the node we want
  Node *new_root;
  // Chosen node is first child
  if (prev_node == nullptr) {
    new_root = root->first_child();
    root->set_first_child(new_root->next_sibling());
  } else {
    new_root = prev_node->next_sibling();
    prev_node->set_next_sibling(new_root->next_sibling());
  }

  new_root->null_next_sibling();
  new_root->null_parent();

  delete root;
  root = new_root;
  cur = root;
  iterations_done = 0;
}

uintf TrainMC::count_nodes() const {
  if (root == nullptr)
    return 0;
  return root->countNodes();
}

void TrainMC::propagateTerminal() {
  Node *node = cur, *cur_node;

  while (node != root) {
    // We only need one loss to deduce
    if (node->result() == kResultLoss || node->result() == kDeducedLoss) {
      node = node->parent();
      node->set_result(kDeducedWin);
    } else {
      // There are no winning moves
      node = node->parent();
      cur_node = node->first_child();
      // Position has a drawing move
      bool has_draw = false;
      uintf edge_index = 0;
      while (cur_node != nullptr) {
        // We can deduce no further (not visited or not known)
        if (cur_node->child_id() != node->move_id(edge_index) ||
            cur_node->result() == kResultNone) {
          return;
        }
        if (cur_node->result() == kResultDraw ||
            cur_node->result() == kDeducedDraw) {
          has_draw = true;
        }
        cur_node = cur_node->next_sibling();
        ++edge_index;
      }
      // Unvisited moves at the end
      if (edge_index < node->num_legal_moves())
        return;
      // If we reach this part, we are guaranteed
      // No winning moves and
      // no unknown moves
      // If there are any drawing moves, the position is a draw
      if (has_draw) {
        node->set_result(kDeducedDraw);
      }
      // Otherwise, there are only losing moves, so the position is a loss
      else {
        node->set_result(kDeducedLoss);
      }
    }
  }
}