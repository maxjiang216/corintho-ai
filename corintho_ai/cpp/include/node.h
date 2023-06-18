#ifndef NODE_H
#define NODE_H

#include <cstdint>

#include <bitset>
#include <ostream>

#include <gsl/gsl>

#include "game.h"
#include "util.h"

/// @brief A node in the Monte Carlo search tree
/// @details The Monte Carlo tree is a tree of nodes. Each node
/// conatins a game position. Its children are states
/// that can be reached by making a move from the parent.
/// A parent node has a pointer to its first child, and
/// each child has a pointer to its next sibling.
/// @note This is the memory bottleneck of the program. Many details of the
/// implementation are designed to reduce the memory footprint of this class.
/// @note The alignment is set to 64 bytes to reduce cache misses. The size of
/// the class is currently just under 64 bytes.
class alignas(64) Node {
 public:
  /// @brief Maximum value of a edge probability weight
  /// @details This is used to scale up the probability weights
  static constexpr float kMaxProbability = 511.0;
  /// @brief Default constructor constructs a node with the starting position
  /// @details This is used to initialize a Monte Carlo search tree.
  /// The starting position is never terminal.
  Node();
  // Delete these constructors as we do not need or want to deep copy nodes
  Node(const Node &) = delete;
  Node(Node &&) noexcept = delete;
  Node &operator=(const Node &) = delete;
  Node &operator=(Node &&) = delete;
  ~Node();
  /// @brief Construct a node from a game position
  /// @param depth Number of turns from the starting position
  /// @details This is used to copy a game state when the opponent
  /// chooses an unforeseen move.
  /// Any position we use in this way cannot be terminal,
  /// or else the game would have ended and we would not have received
  /// the position.
  Node(const Game &game, int32_t depth);
  /// @brief Construct a node from its parent
  /// @details This is the most commonly used constructor during training.
  /// It is used to add a new node to the tree when considering a new move.
  /// @param depth The depth of the position after applying the move
  /// or 1 more than the depth of parent
  Node(const Game &game, Node *parent, Node *next_sibling, int32_t move_id,
       int32_t depth) noexcept;

  Game game() const noexcept;
  Node *parent() const noexcept;
  Node *next_sibling() const noexcept;
  Node *first_child() const noexcept;
  float evaluation() const noexcept;
  int32_t visits() const noexcept;
  Result result() const noexcept;
  int32_t child_id() const noexcept;
  int32_t num_legal_moves() const noexcept;
  int32_t depth() const noexcept;
  bool all_visited() const noexcept;
  int32_t move_id(int32_t i) const noexcept;
  float probability(int32_t i) const noexcept;
  /// @brief Whether the game is in a terminal position
  bool terminal() const noexcept;

  void set_next_sibling(Node *next_sibling) noexcept;
  void set_first_child(Node *first_child) noexcept;
  void set_evaluation(float evaluation) noexcept;
  void set_denominator(float denominator) noexcept;
  void set_visits(int32_t visits) noexcept;
  void set_result(Result result) noexcept;
  /// @brief Set all_visited
  /// @param all_visited Default is true to match other standard set functions
  void set_all_visited(bool all_visited = true) noexcept;
  void set_probability(int32_t i, int32_t probability) noexcept;
  void increment_visits() noexcept;
  void decrement_visits() noexcept;
  void increase_evaluation(float d) noexcept;
  void decrease_evaluation(float d) noexcept;
  void null_parent() noexcept;
  void null_next_sibling() noexcept;

  int32_t countNodes() const;

  /// @returns If there are lines in the position
  bool getLegalMoves(std::bitset<kNumMoves> &legal_moves) const noexcept;
  void writeGameState(float game_state[kGameStateSize]) const noexcept;
  /// @brief Print the main line of the game
  /// @param logging_file File to write line into
  /// @details The main line is the sequence of moves with the
  /// most visits. Chooses moves as the Monte Carlo search tree would.
  void printMainLine(std::ostream &logging_file) const;
  /// @brief Print all lines where the result is deduced
  /// @param logging_file File to write lines into
  void printKnownLines(std::ostream &logging_file) const;

 private:
  struct Edge {
    /// @brief The ID of the move used to reach the child
    /// @details The maximal move ID is 95, which fits in a 7-bit integer.
    uint16_t move_id : 7;
    /// @brief The probability weight of this move
    /// @details We scale the weights to be between 0 and 511
    uint16_t probability : 9;
    Edge() = default;
    Edge(int32_t move_id, int32_t probability)
        : move_id{gsl::narrow_cast<uint16_t>(move_id)},
          probability{gsl::narrow_cast<uint16_t>(probability)} {}
  };

  /// @brief Initialize the edges of this node
  void initializeEdges() noexcept;

  /// @brief The game position
  Game game_{};
  /// @brief A pointer to the parent node
  Node *parent_{nullptr};
  /// @brief A pointer to the next sibling node
  /// @details The children of a node are stored as a linked list.
  /// This saves us from having to allocate memory for an array of
  /// pointers to children at the cost of having to traverse the list
  /// to find a child. This is an idea taken from the Leela Zero implementation.
  Node *next_sibling_{nullptr};
  /// @brief A pointer to the first child node
  /// @note This would more ideally be a std::shared_ptr, but
  /// that would increase the size of the class over 64 bytes.
  Node *first_child_{nullptr};
  /// @brief An array of edges to the children of this node
  /// @details The edges are stored in a variable length array
  /// so that we only allocate as much memory as we need (num_legal_moves).
  /// This is an idea taken from the Leela Zero implementation.
  /// Ideally, this would be a vector, but that would increase the size of the
  /// class over 64 bytes.
  Edge *edges_{nullptr};
  /// @brief The evaluation of this node
  /// @details This is a sum of the initial evaluation and all the
  /// evaluations propagated from the children.
  float evaluation_{0.0};
  /// @brief The sum of the weights of the edges
  /// @details This is used to normalize the probabilities of the edges.
  /// In this way, we can scale the weights of the edges up to get more precise
  /// probabilities while using a small number of bits to store the weights.
  float denominator_{0.0};
  /// @brief The number of times this node has been visited
  /// @note int16_t is the largest size that can be used without
  /// increasing the size of the class over 64 bytes. It allows
  /// for 32,767 visits. This is more than enough for training.
  /// During play, this number may be exceeded. Search should stop for any
  /// node that has been visited 32,767 times.
  /// @note It is generally initialized to 1, since nodes are usually created
  /// just before it is visited. This saves us an increment.
  int16_t visits_{1};
  /// @brief The result of the game in this position
  /// @details The value is kResultNone unless it is a terminal position or
  /// we have deduced the result from the children (played out until the end)
  /// There are only 7 possible results, so we store them in an 8-bit integer,
  /// aliased by Result.
  Result result_{kResultNone};
  /// @brief The ID of the move used to reach this node from its parent
  /// @details The maximal move ID is 96, which fits in an 8-bit integer.
  const int8_t child_id_;
  /// @brief The number of legal moves in this position
  /// @details This is used to determine the size of the array of edges.
  /// The maximal number of legal moves is 48, which fits in an 8-bit integer.
  int8_t num_legal_moves_{0};
  /// @brief The depth of this node
  /// @details The depth is the number of moves from the starting position.
  /// The maximal depth is 40, which fits in an 8-bit integer.
  const int8_t depth_;
  /// @brief Whether all the children of this node have been visited
  /// @details This is used to determine whether we should stop searching this
  /// node. It is set to true when the node is created, as it has no children
  /// and nodes are visited when they are created.
  bool all_visited_{true};
};

#endif