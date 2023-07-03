#include "node.h"

#include <bitset>
#include <ostream>

#include <gsl/gsl>

#include "game.h"
#include "move.h"
#include "util.h"

Node::Node() : child_id_{0}, depth_{0} {
  // initializeEdges can throw an exception from new
  initializeEdges();
}

Node::~Node() {
  delete[] edges_;
  delete next_sibling_;
  delete first_child_;
}

Node::Node(const Game &game, int32_t depth)
    : game_{game}, child_id_{0}, depth_{gsl::narrow_cast<int8_t>(depth)} {
  // initializeEdges can throw an exception from new
  initializeEdges();
}

Node::Node(const Game &game, Node *parent, Node *next_sibling, int32_t move_id,
           int32_t depth)
    : game_{game}, parent_{parent},
      next_sibling_{next_sibling}, child_id_{gsl::narrow_cast<int8_t>(move_id)},
      depth_{gsl::narrow_cast<int8_t>(depth)} {
  game_.doMove(move_id);
  // initializeEdges can throw an exception from new
  initializeEdges();
}

Game Node::game() const noexcept {
  return game_;
}

Node *Node::parent() const noexcept {
  return parent_;
}

Node *Node::next_sibling() const noexcept {
  return next_sibling_;
}

Node *Node::first_child() const noexcept {
  return first_child_;
}

float Node::evaluation() const noexcept {
  return evaluation_;
}

int32_t Node::visits() const noexcept {
  return visits_;
}

Result Node::result() const noexcept {
  return result_;
}

int32_t Node::child_id() const noexcept {
  return child_id_;
}

int32_t Node::num_legal_moves() const noexcept {
  return num_legal_moves_;
}

int32_t Node::depth() const noexcept {
  return depth_;
}

bool Node::all_visited() const noexcept {
  return all_visited_;
}

int32_t Node::move_id(int32_t i) const noexcept {
  assert(i < num_legal_moves_);
  return edges_[i].move_id;
}

float Node::probability(int32_t i) const noexcept {
  assert(i < num_legal_moves_);
  assert(denominator_ > 0.0);
  return static_cast<float>(edges_[i].probability) * denominator_;
}

bool Node::terminal() const noexcept {
  return result_ == kResultLoss || result_ == kResultDraw;
}

bool Node::known() const noexcept {
  return result_ != kResultNone;
}

bool Node::won() const noexcept {
  // A terminal position can never be winning for the current player
  return result_ == kDeducedWin;
}

bool Node::lost() const noexcept {
  return result_ == kResultLoss || result_ == kDeducedLoss;
}

bool Node::drawn() const noexcept {
  return result_ == kResultDraw || result_ == kDeducedDraw;
}

const Game &Node::get_game() const noexcept {
  return game_;
}

void Node::set_next_sibling(Node *next_sibling) noexcept {
  next_sibling_ = next_sibling;
}

void Node::set_first_child(Node *first_child) noexcept {
  first_child_ = first_child;
}

void Node::set_evaluation(float evaluation) noexcept {
  evaluation_ = evaluation;
}

void Node::set_denominator(float denominator) noexcept {
  denominator_ = denominator;
}

void Node::set_visits(int32_t visits) noexcept {
  visits_ = gsl::narrow_cast<int16_t>(visits);
}

void Node::set_result(Result result) noexcept {
  result_ = result;
}

void Node::set_all_visited(bool all_visited) noexcept {
  all_visited_ = all_visited;
}

void Node::set_probability(int32_t i, int32_t probability) noexcept {
  assert(i < num_legal_moves_);
  edges_[i].probability = gsl::narrow_cast<uint16_t>(probability);
}

void Node::increment_visits() noexcept {
  ++visits_;
}

void Node::decrement_visits() noexcept {
  --visits_;
}

void Node::increase_evaluation(float d) noexcept {
  evaluation_ += d;
}

void Node::decrease_evaluation(float d) noexcept {
  evaluation_ -= d;
}

void Node::null_parent() noexcept {
  parent_ = nullptr;
}

void Node::null_next_sibling() noexcept {
  next_sibling_ = nullptr;
}

int32_t Node::countNodes() const noexcept {
  int32_t counter = 1;
  Node *cur_child = first_child_;
  while (cur_child != nullptr) {
    counter += cur_child->countNodes();
    cur_child = cur_child->next_sibling_;
  }
  return counter;
}

bool Node::getLegalMoves(std::bitset<kNumMoves> &legal_moves) const noexcept {
  return game_.getLegalMoves(legal_moves);
}

void Node::writeGameState(float game_state[kGameStateSize]) const noexcept {
  game_.writeGameState(game_state);
}

void Node::printMainLine(std::ostream &logging_file) const {
  Node *cur_child = first_child_;
  Node *best_child = nullptr;
  int32_t max_visits = 0;
  int32_t edge_index = 0;
  float max_eval = 0.0;
  float prob = 0.0;
  while (cur_child != nullptr) {
    // This edge has a corresponding child
    // i.e. it has been visited
    if (move_id(edge_index) == cur_child->child_id_) {
      // If we have deduced a result, choose that move
      if (cur_child->result_ == kDeducedLoss ||
          cur_child->result_ == kResultLoss) {
        best_child = cur_child;
        max_visits = cur_child->visits_;
        prob = probability(edge_index);
        break;
      }
      // Choose the child with the most visits
      // Break ties by choosing the child with the highest evaluation
      if (cur_child->visits_ > max_visits ||
          (cur_child->visits_ == max_visits &&
           cur_child->evaluation_ > max_eval)) {
        best_child = cur_child;
        max_visits = cur_child->visits_;
        max_eval = cur_child->evaluation_;
        prob = probability(edge_index);
      }
      cur_child = cur_child->next_sibling_;
    }
    ++edge_index;
  }
  if (best_child != nullptr) {
    logging_file << static_cast<int32_t>(best_child->depth_) << ". "
                 << Move{best_child->child_id_} << " v: " << max_visits
                 << " e: ";
    if (best_child->result_ != kResultNone) {
      logging_file << strResult(best_child->result_);
    } else {
      logging_file << max_eval / (float)max_visits;
    }
    logging_file << " p: " << prob << '\t';
    best_child->printMainLine(logging_file);
  }
}

void Node::printKnownLines(std::ostream &logging_file) const {
  if (result_ != kResultNone) {
    logging_file << static_cast<int32_t>(depth_) << ". " << Move{child_id_}
                 << ' ' << strResult(result_) << " ( ";
    Node *cur_child = first_child_;
    while (cur_child != nullptr) {
      cur_child->printKnownLines(logging_file);
      cur_child = cur_child->next_sibling_;
    }
    logging_file << " ) ";
  }
}

void Node::initializeEdges() {
  std::bitset<kNumMoves> legal_moves;
  bool is_lines = game_.getLegalMoves(legal_moves);
  num_legal_moves_ = legal_moves.count();
  // Terminal node
  if (num_legal_moves_ == 0) {
    // We have to set visits_ to 0 for terminal nodes
    // Otherwise, we will overcount by 1
    visits_ = 0;
    // Current player has lost if there are lines
    if (is_lines) {
      result_ = kResultLoss;
      return;
    }
    // If there are no lines and no legal moves, the game is a draw
    result_ = kResultDraw;
    return;
  }
  // Otherwise, allocate edges for the legal moves
  edges_ = new Edge[num_legal_moves_];
  int32_t edge_index = 0;
  // Fill the array with legal moves
  for (int32_t i = 0; i < kNumMoves; ++i) {
    if (legal_moves[i]) {
      edges_[edge_index] = Edge(i, 0);
      ++edge_index;
    }
  }
}