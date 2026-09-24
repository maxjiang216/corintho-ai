#include "node.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>

#include <bitset>
#include <ostream>
#include <utility>

#include <gsl/gsl>

#include "game.h"
#include "move.h"
#include "util.h"

Node::Node() : child_id_{0}, depth_{0} {
  // initializeEdges can throw an exception from new
  initializeEdges();
}

Node::~Node() {
  if (block_ != nullptr) {
    Arena::get().deallocate(block_->stats, statsBytes(block_->capacity));
    delete block_;
  }
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
    : game_{game}, parent_{parent}, next_sibling_{next_sibling},
      child_id_{gsl::narrow_cast<int8_t>(move_id)},
      depth_{gsl::narrow_cast<int8_t>(depth)} {
  game_.doMove(move_id);
  // initializeEdges can throw an exception from new
  initializeEdges();
  // Take a slot in the parent's child statistics, then fill it. After
  // initializeEdges, so a terminal result is already known. TrainMC also
  // builds a new root this way (parent nullptr) when it discards the tree
  // after a move with no visits; a root keeps its statistics in its own
  // fields only.
  if (parent_ != nullptr) {
    slot_ = parent_->addChild(this);
    registerStats();
  }
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
  return block_->edges[i].move_id();
}

float Node::probability(int32_t i) const noexcept {
  assert(i < num_legal_moves_);
  assert(denominator_ > 0.0);
  return static_cast<float>(block_->edges[i].probability()) * denominator_;
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
  if (parent_ != nullptr)
    *statsSlot().evaluation = evaluation_;
}

void Node::set_denominator(float denominator) noexcept {
  assert(denominator > 0.0);
  denominator_ = denominator;
}

void Node::set_visits(int32_t visits) noexcept {
  visits_ = gsl::narrow_cast<int16_t>(visits);
  if (parent_ != nullptr)
    *statsSlot().visits = static_cast<float>(visits_);
}

void Node::set_result(Result result) noexcept {
  result_ = result;
  syncFlags();
}

void Node::set_all_visited(bool all_visited) noexcept {
  // Backup clears this on every node of the path, and it is almost always
  // already clear, so skip the write into the parent when nothing changes
  if (all_visited_ == all_visited)
    return;
  all_visited_ = all_visited;
  syncFlags();
}

void Node::set_probability(int32_t i, int32_t probability) noexcept {
  assert(i < num_legal_moves_);
  block_->edges[i].set_probability(probability);
}

void Node::promoteBestEdge(int32_t first) noexcept {
  assert(first < num_legal_moves_);
  // One integer per edge that orders by probability weight, then by lower
  // move ID. Move IDs are distinct, so ranks are too, and the scan is a plain
  // running maximum rather than a two-field comparison.
  auto rank = [](Edge e) -> uint32_t {
    return (static_cast<uint32_t>(e.probability()) << 7) |
           (127U - static_cast<uint32_t>(e.move_id()));
  };
  int32_t best = first;
  uint32_t best_rank = rank(block_->edges[first]);
  for (int32_t i = first + 1; i < num_legal_moves_; ++i) {
    const uint32_t r = rank(block_->edges[i]);
    if (r > best_rank) {
      best_rank = r;
      best = i;
    }
  }
  std::swap(block_->edges[first], block_->edges[best]);
}

// Each update changes this node's own field, then stores only that field into
// the parent's statistics. The store never reads the parent's line, so a miss
// on it (common in backup, whose path is cold) does not stall: see worklog
// entry 25, where updating the parent's entry in place instead turned these
// store misses into load misses.
void Node::increment_visits() noexcept {
  ++visits_;
  if (parent_ != nullptr)
    *statsSlot().visits = static_cast<float>(visits_);
}

void Node::decrement_visits() noexcept {
  --visits_;
  if (parent_ != nullptr)
    *statsSlot().visits = static_cast<float>(visits_);
}

void Node::add_visit(float evaluation) noexcept {
  ++visits_;
  evaluation_ += evaluation;
  if (parent_ != nullptr) {
    const StatsSlot s = statsSlot();
    *s.visits = static_cast<float>(visits_);
    *s.evaluation = evaluation_;
  }
}

void Node::increase_evaluation(float d) noexcept {
  evaluation_ += d;
  if (parent_ != nullptr)
    *statsSlot().evaluation = evaluation_;
}

void Node::decrease_evaluation(float d) noexcept {
  evaluation_ -= d;
  if (parent_ != nullptr)
    *statsSlot().evaluation = evaluation_;
}

Node::ChildStats Node::child_stats() const noexcept {
  if (block_ == nullptr || block_->stats == nullptr)
    return ChildStats{nullptr, nullptr, nullptr, nullptr, 0, nullptr, 0.0F};
  const int32_t cap = block_->capacity;
  unsigned char *base = block_->stats;
  auto *child = reinterpret_cast<Node **>(base);
  auto *evaluation = reinterpret_cast<float *>(child + cap);
  auto *visits = evaluation + cap;
  auto *flags = reinterpret_cast<uint8_t *>(visits + cap);
  return ChildStats{child,      evaluation,  visits,      flags,
                    block_->num_children, block_->edges, denominator_};
}

int8_t Node::addChild(Node *child) {
  assert(block_ != nullptr);
  EdgeBlock &b = *block_;
  if (b.num_children == b.capacity) {
    // Most expanded nodes stop at one or two children, so capacity grows
    // 3 -> 7 -> 15 -> the legal move count, each step filling a 64-, 128- or
    // 256-byte slot. A node with children has on average ~26 legal moves, and
    // over half of them never get a second child. The 15 step matters too:
    // without it, peak memory was 32 MB higher at 2000 games (entry 16).
    static_assert(statsBytes(3) <= Arena::kSmall);
    static_assert(statsBytes(7) <= Arena::kLarge);
    static_assert(statsBytes(15) <= 2 * Arena::kLarge);
    static_assert(statsBytes(kMaxEdges) <= Arena::kMaxBlock);
    const int32_t old_cap = b.capacity;
    const int32_t new_cap = std::min<int32_t>(
        old_cap == 0   ? 3
        : old_cap == 3 ? 7
        : old_cap == 7 ? 15
                       : num_legal_moves_,
        num_legal_moves_);
    assert(new_cap > old_cap);
    auto *fresh = static_cast<unsigned char *>(
        Arena::get().allocate(statsBytes(new_cap)));
    if (old_cap > 0) {
      // Same four arrays, each now new_cap long
      unsigned char *old = b.stats;
      const size_t n = static_cast<size_t>(b.num_children);
      std::memcpy(fresh, old, n * sizeof(Node *));
      std::memcpy(fresh + new_cap * sizeof(Node *),
                  old + old_cap * sizeof(Node *), n * sizeof(float));
      std::memcpy(fresh + new_cap * (sizeof(Node *) + sizeof(float)),
                  old + old_cap * (sizeof(Node *) + sizeof(float)),
                  n * sizeof(float));
      std::memcpy(fresh + new_cap * (sizeof(Node *) + 2 * sizeof(float)),
                  old + old_cap * (sizeof(Node *) + 2 * sizeof(float)), n);
      Arena::get().deallocate(old, statsBytes(old_cap));
    }
    b.stats = fresh;
    b.capacity = gsl::narrow_cast<int8_t>(new_cap);
  }
  const int8_t slot = b.num_children++;
  reinterpret_cast<Node **>(b.stats)[slot] = child;
  return slot;
}

Node::StatsSlot Node::statsSlot() const noexcept {
  assert(parent_ != nullptr);
  const EdgeBlock &b = *parent_->block_;
  const int32_t cap = b.capacity;
  auto *evaluation = reinterpret_cast<float *>(b.stats + cap * sizeof(Node *));
  auto *visits = evaluation + cap;
  auto *flags = reinterpret_cast<uint8_t *>(visits + cap);
  return StatsSlot{evaluation + slot_, visits + slot_, flags + slot_};
}

uint8_t Node::selectionFlags() const noexcept {
  return static_cast<uint8_t>(
      (((known() && !drawn()) || all_visited_) ? kSkipChild : 0) |
      (drawn() ? kDrawnChild : 0));
}

void Node::syncFlags() noexcept {
  if (parent_ != nullptr)
    *statsSlot().flags = selectionFlags();
}

void Node::registerStats() noexcept {
  // A new node's own fields hold its initial values (visits 1, evaluation 0)
  const StatsSlot s = statsSlot();
  *s.evaluation = evaluation_;
  *s.visits = static_cast<float>(visits_);
  *s.flags = selectionFlags();
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

void Node::printMainLine(std::ostream *log_file) const {
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
        max_visits = cur_child->visits();
        prob = probability(edge_index);
        break;
      }
      // Choose the child with the most visits
      // Break ties by choosing the child with the highest evaluation
      if (cur_child->visits() > max_visits ||
          (cur_child->visits() == max_visits &&
           cur_child->evaluation() > max_eval)) {
        best_child = cur_child;
        max_visits = cur_child->visits();
        max_eval = cur_child->evaluation();
        prob = probability(edge_index);
      }
      cur_child = cur_child->next_sibling_;
    }
    ++edge_index;
  }
  if (best_child != nullptr) {
    *log_file << static_cast<int32_t>(best_child->depth_) << ". "
              << Move{best_child->child_id_} << " V: " << max_visits << " E: ";
    if (best_child->result_ != kResultNone) {
      *log_file << strResult(best_child->result_);
    } else {
      *log_file << max_eval / (float)max_visits;
    }
    *log_file << " p: " << prob << '\t';
    best_child->printMainLine(log_file);
  }
}

void Node::printKnownLines(std::ostream *log_file) const {
  if (result_ != kResultNone) {
    *log_file << static_cast<int32_t>(depth_) << ". " << Move{child_id_} << ' '
              << strResult(result_) << " ( ";
    Node *cur_child = first_child_;
    while (cur_child != nullptr) {
      cur_child->printKnownLines(log_file);
      cur_child = cur_child->next_sibling_;
    }
    *log_file << " ) ";
  }
}

void Node::initializeEdges() {
  MoveMask legal_moves;
  bool is_lines = game_.getLegalMoves(legal_moves);
  num_legal_moves_ = gsl::narrow_cast<int8_t>(legal_moves.count());
  // Terminal node
  if (num_legal_moves_ == 0) {
    // Don't set visits to 0. Not sure why we added this.
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
  block_ = new EdgeBlock;
  int32_t edge_index = 0;
  // Iterate the set bits rather than testing all 96. The test was one
  // unpredictable branch per legal move, about 26 mispredicts per call.
  forEachMove(legal_moves, [this, &edge_index](int32_t id) {
    block_->edges[edge_index] = Edge(id, 0);
    ++edge_index;
  });
}