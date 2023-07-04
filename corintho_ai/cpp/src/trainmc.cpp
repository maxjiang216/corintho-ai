#include "trainmc.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

#include <fstream>
#include <limits>
#include <random>
#include <vector>

#include <gsl/gsl>

#include "move.h"
#include "node.h"

TrainMC::TrainMC(std::mt19937 *generator, int32_t max_searches,
                 int32_t searches_per_eval, float c_puct, float epsilon,
                 bool testing)
    : max_searches_{max_searches}, searches_per_eval_{searches_per_eval},
      c_puct_{c_puct}, epsilon_{epsilon}, testing_{testing}, generator_{
                                                                 generator} {
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

const Game &TrainMC::get_root_game() const noexcept {
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

int32_t TrainMC::chooseMove(float game_state[kGameStateSize],
                            float prob_sample[kNumMoves]) noexcept {
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
        prob_sample[cur->child_id()] =
            static_cast<float>(cur->visits()) * denominator;
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
        if (cur->visits() > max_visits ||
            (cur->visits() == max_visits && eval > max_eval)) {
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

bool TrainMC::receiveOpponentMove(int32_t move_choice, const Game &game,
                                  int32_t depth) {
  Node *cur = root_->first_child();
  Node *prev = nullptr;
  while (cur != nullptr) {
    if (cur->child_id() == move_choice) {
      moveDown(prev);
      return false;
    }
    prev = cur;
    cur = cur->next_sibling();
  }
  // Haven't searched this move yet
  // The current tree is not needed
  delete root_;
  // Copy opponent game state into our root
  root_ = nullptr;
  createRoot(game, depth);
  // We need an evaluation
  cur_->writeGameState(to_eval_);
  searched_.push_back(cur_);
  searches_done_ = 1;
  searched_index_ = 1;
  return true;
}

void TrainMC::receiveEval(float eval[], float probs[]) noexcept {
  assert(eval != nullptr);
  assert(probs != nullptr);
  assert(searched_index_ > 0);
  for (int32_t i = 0; i < searched_.size(); ++i) {
    cur_ = searched_[i];

    // Apply the legal move filter
    // Legal moves can be deduced from edges
    int32_t edge_index = 0;
    float sum = 0.0;
    float filtered_probs[cur_->num_legal_moves()];
    for (int32_t j = 0; j < kNumMoves; ++j) {
      if (edge_index < cur_->num_legal_moves() &&
          cur_->move_id(edge_index) == j) {
        filtered_probs[edge_index] = probs[kNumMoves * i + j];
        sum += filtered_probs[edge_index];
        ++edge_index;
        if (edge_index == cur_->num_legal_moves()) {
          break;
        }
      }
    }
    // Generate Dirichlet noise
    float dirichlet_sum = 0.0;
    float dirichlet_noise[cur_->num_legal_moves()];
    for (int32_t j = 0; j < cur_->num_legal_moves(); ++j) {
      dirichlet_noise[j] = gamma_samples[(*generator_)() % GAMMA_BUCKETS];
      dirichlet_sum += dirichlet_noise[j];
    }
    // Normalizing factors
    // Factoring this out saves division operations
    float scalar = 1.0 / sum * (1 - epsilon_);
    float dirichlet_scalar = 1.0 / dirichlet_sum * epsilon_;
    // Combine probabilities and Dirichlet noise
    float weighted_probs[cur_->num_legal_moves()];
    float max_prob = 0.0;
    edge_index = 0;
    for (int32_t j = 0; j < cur_->num_legal_moves(); ++j) {
      weighted_probs[j] =
          filtered_probs[j] * scalar + dirichlet_noise[j] * dirichlet_scalar;
      max_prob = std::max(weighted_probs[j], max_prob);
    }
    // Scale up probabilities and convert to integers
    float denom = Node::kMaxProbability / max_prob;
    int32_t final_sum = 0;
    for (int32_t j = 0; j < cur_->num_legal_moves(); ++j) {
      // Make all probabilities positive
      cur_->set_probability(
          j, std::max(1, gsl::narrow_cast<int32_t>(lround(
                             static_cast<double>(weighted_probs[j]) * denom))));
      final_sum += cur_->probability(j);
    }
    cur_->set_denominator(1.0 / (float)final_sum);
    // Propagate the evaluation up the tree
    float cur_eval = eval[i];
    while (cur_->parent() != nullptr) {
      // Correct default +1 evaluation
      cur_->increase_evaluation(cur_eval - 1.0);
      // Reset this marker
      cur_->set_all_visited(false);
      cur_eval *= -1.0;
      cur_ = cur_->parent();
    }
    // Propagate to the root
    cur_->increase_evaluation(cur_eval - 1.0);
  }
  root_->set_all_visited(false);
  searched_index_ = 0;
  searched_.clear();
}

void TrainMC::moveDown(Node *prev) noexcept {
  // Extricate the node we want
  Node *new_root;
  // Chosen node is first child
  if (prev == nullptr) {
    new_root = root_->first_child();
    root_->set_first_child(new_root->next_sibling());
  } else {
    new_root = prev->next_sibling();
    prev->set_next_sibling(new_root->next_sibling());
  }
  // Remove new root from the tree
  // So that it is not deleted with the old root
  new_root->null_next_sibling();
  new_root->null_parent();
  delete root_;
  root_ = new_root;
  cur_ = root_;
  searches_done_ = 0;
}

void TrainMC::propagateTerminal() noexcept {
  // We can only deduce more results from new terminal nodes
  assert(cur_->terminal());
  Node *cur = cur_;
  while (cur != root_) {
    // We only need one loss to deduce a win
    if (cur->lost()) {
      cur = cur->parent();
      cur->set_result(kDeducedWin);
    } else {
      cur = cur->parent();
      Node *cur_child = cur->first_child();
      // Position has a drawing move
      bool has_draw = false;
      int32_t edge_index = 0;
      while (cur_child != nullptr) {
        // If there is an unknown or unvisited move, we can deduce no further
        if (cur_child->child_id() != cur->move_id(edge_index) ||
            !cur_child->known()) {
          return;
        }
        if (cur->drawn()) {
          has_draw = true;
        }
        cur_child = cur_child->next_sibling();
        ++edge_index;
      }
      // There are unvisited moves after the last visited move
      if (edge_index < cur->num_legal_moves())
        return;
      // No winning or unknown moves
      // If there are any drawing moves, the position is a draw
      if (has_draw) {
        cur->set_result(kDeducedDraw);
      }
      // Otherwise, there are only losing moves, so the position is a loss
      else {
        cur->set_result(kDeducedLoss);
      }
    }
  }
}

bool TrainMC::search() {
  bool need_eval = false;
  // While we have not found a node needing evaluation,
  // have not done too many searches, and
  // the result of the root is not known
  while (!need_eval && searches_done_ < max_searches_ && !root_->known()) {
    assert(cur_ == root_);
    const float neg_inf = -std::numeric_limits<float>::infinity();
    ++searches_done_;
    while (!cur_->terminal()) {
      // Choose the next node to move down to
      float max_eval = neg_inf;
      int32_t choice = 0;
      Node *cur_child = cur_->first_child();
      int32_t edge_index = 0;
      // Keep track of previous node to insert into linked list
      Node *prev = nullptr;
      Node *best_prev = nullptr;
      // Factor this value out, as it is expense to compute
      float v_sqrt = c_puct_ * sqrt(static_cast<float>(cur_->visits()));
      while (cur_child != nullptr || edge_index < cur_->num_legal_moves()) {
        float u = neg_inf;
        // This node has already been visited
        if (cur_child != nullptr &&
            cur_child->child_id() == cur_->move_id(edge_index)) {
          // Don't all_visited nodes or won or lost positions
          // We search draws since the number of searches they have
          // makes a difference in choose_move
          // as they are not automatically chosen or excluded
          if ((!cur_child->known() || cur_child->drawn()) &&
              !cur_child->all_visited()) {
            // Known draw, use evaluation 0
            if (cur_child->drawn()) {
              u = cur_->probability(edge_index) * v_sqrt;
            } else {
              u = -1.0 * cur_->evaluation() /
                      static_cast<float>(cur_child->visits()) +
                  cur_->probability(edge_index) * v_sqrt /
                      (static_cast<float>(cur_child->visits()) + 1.0);
            }
          }
          prev = cur_child;
          cur_child = cur_child->next_sibling();
          // This node has not been visited, ignore the evaluation term in the
          // UCB formula This is essentially using a default evaluation of 0
          // (but we avoid division by 0)
        } else {
          u = cur_->probability(edge_index) * v_sqrt;
        }
        if (u > max_eval) {
          // If the node has not been visited, prev is the previous node
          // and can be used to insert the new node
          // otherwise it is the current node, but if this is the best node,
          // best_prev is not needed since we don't insert
          best_prev = prev;
          max_eval = u;
          choice = cur_->move_id(edge_index);
        }
        ++edge_index;
      }
      cur_->increment_visits();
      // We use a default evaluation of 1.0 before we have a neural net
      // evaluation This helps diversify the searches In particular, the second
      // player has a large advantage in Corintho So most positions the first
      // player plays has only losing moves In such a position, a move getting a
      // 0.0 evaluation search will improve its evaluation and will very likely
      // be searched again before neural net evaluations are received This
      // drastically decreases the effective number of searches Most
      // pessimistically, with 1600 searches and 16 searches per evaluation, we
      // effectively get 100 searches Change the default value from 0.0 to 1.0
      // empirically drastically improved improvement rate. Note that every node
      // visited gets this default, as opposed to a leaf node getting it and
      // having it propagate the normal way. This ensures that visited nodes are
      // maximally unlikely to get visited again.
      cur_->increase_evaluation(1.0);
      // If no nodes were searched, this means all nodes are all_visited
      // or won or lost positions
      // This node is then all_visited
      if (max_eval == neg_inf) {
        cur_->set_all_visited();
        // If it is the root that is all searched_
        // we should stop the search and get evaluations immediately
        bool done = (cur_ == root_);
        // Since no search was done
        // We need to undo the visit count and default 1.0 evaluation
        while (cur_->parent() != nullptr) {
          cur_->decrement_visits();
          cur_->decrease_evaluation(1.0);
          cur_ = cur_->parent();
        }
        // Remove search from root
        cur_->decrement_visits();
        cur_->decrease_evaluation(1.0);
        --searches_done_;
        return done;
      }
      // New node at the beginning of the list
      if (best_prev == nullptr) {
        cur_->set_first_child(new Node(cur_child->game(), cur_child,
                                       cur_child->first_child(), choice,
                                       cur_child->depth() + 1));
        cur_ = cur_->first_child();
        break;
      }
      // New node somewhere else in the list
      else if (best_prev->child_id() != choice) {
        best_prev->set_next_sibling(new Node(cur_->get_game(), cur_,
                                             best_prev->next_sibling(), choice,
                                             cur_->depth() + 1));
        cur_ = best_prev->next_sibling();
        break;
      }
      // Existing node, continue searching
      // In this case, best_prev is the best child node
      else {
        cur_ = best_prev;
      }
    }
    // Terminal node
    // This is usually a new node
    // But may be a drawn node that has been searched before
    if (cur_->terminal()) {
      // Propagate the result
      propagateTerminal();
      // In a decisive terminal state, the person to play is always the loser
      // Otherwise the evaluation is 0.0 for a draw.
      float cur_eval = -1.0;
      if (cur_->drawn()) {
        cur_eval = 0.0;
      }
      cur_->set_evaluation(cur_eval);
      while (cur_->parent() != nullptr) {
        cur_ = cur_->parent();
        // Correct default +1.0 evaluation
        cur_->increase_evaluation(cur_eval - 1.0);
        cur_eval *= -1.0;
      }
    }
    // Otherwise, request an evaluation for the new node
    else {
      // Default +1 evaluation for new node
      cur_->set_evaluation(1.0);
      need_eval = true;
      // Write game in correct position
      cur_->writeGameState(to_eval_ + searched_index_ * kGameStateSize);
      // Record the node in searched_
      searched_.push_back(cur_);
      ++searched_index_;
    }
  }
  // Reset cur for next search
  cur_ = root_;
  // Conditions for searches being done for this move
  return searches_done_ == max_searches_ || root_->known();
}