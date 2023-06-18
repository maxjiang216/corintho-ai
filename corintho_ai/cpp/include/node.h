#ifndef NODE_H
#define NODE_H

#include <cstdint>

#include <bitset>

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
  /// @brief Default constructor constructs a node with the starting position
  /// @details This is used to initialize a Monte Carlo search tree.
  Node();
  ~Node();
  /// @brief Construct a node from a game position
  /// @param depth Number of turns from the starting position
  /// @details This is used to copy a game state when the opponent
  /// chooses an unforeseen move.
  Node(const Game &game, uint8s depth);
  /// @brief Construct a node from its parent
  /// @details This is the most commonly used constructor during training.
  /// It is used to add a new node to the tree when considering a new move.
  Node(const Game &game, uint8s depth, Node *parent, Node *next_sibling,
       uint8s move_choice);

  // Returns whether there are lines
  bool get_legal_moves(std::bitset<kNumMoves> &legal_moves) const;

  // Accessors
  bool is_terminal() const;
  float get_probability(uintf edge_index) const;

  void write_game_state(float game_state[kGameStateSize]) const;

  uintf count_nodes() const;

  void print_main_line(std::ostream *logging_file) const;
  void print_known(std::ostream *logging_file) const;

 private:
  static constexpr float MAX_PROBABILITY = 511.0;

  struct Edge {
    uint16s move_id : 7, probability : 9;
    Edge() = default;
    Edge(uintf move_id, uintf probability)
        : move_id{(uint16s)move_id}, probability{(uint16s)probability} {}
  };

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
  Result result_;
  /// @brief The ID of the move used to reach this node from its parent
  /// @details The maximal move ID is 96, which fits in an 8-bit integer.
  int8_t child_id_{-1};
  /// @brief The number of legal moves in this position
  /// @details This is used to determine the size of the array of edges.
  /// The maximal number of legal moves is 48, which fits in an 8-bit integer.
  int8_t num_legal_moves_{0};
  /// @brief The depth of this node
  /// @details The depth is the number of moves from the starting position.
  /// The maximal depth is 40, which fits in an 8-bit integer.
  uint8s depth_;
  /// @brief Whether all the children of this node have been visited
  /// @details This is used to determine whether we should stop searching this
  /// node.
  bool all_visited_;

  void initialize_edges();

  friend class TrainMC;
  friend class SelfPlayer;
  friend class PlayMC;
};

#endif