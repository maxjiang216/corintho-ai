#ifndef GAME_H
#define GAME_H

#include <cstdint>

#include <bitset>
#include <ostream>

#include "move.h"
#include "util.h"

/// @brief Represents a Corintho game state
/// @note The class is designed to be as memory efficient as possible
/// since it is contained in each node of the Monte Carlo Search Tree
/// which is the main memory bottleneck of the program
/// @note The game state does not know if it is a terminal state
/// This is deduced by the Node class from the return value of getLegalMoves
/// (when there are no legal moves)
class Game {
 public:
  /// @brief The default constructor creates a game in the starting position
  Game() noexcept = default;
  Game(const Game &) noexcept = default;
  Game(Game &&) noexcept = default;
  Game &operator=(const Game &) noexcept = default;
  Game &operator=(Game &&) noexcept = default;
  ~Game() = default;
  // Create a new game from an arbitrary state
  // Used in the web app to feed an arbitrary game state to the MCST
  Game(int32_t board[4 * kBoardSize], int32_t to_play,
       int32_t pieces[6]) noexcept;

  /// @brief Mutates legal_moves to indicate which moves are legal
  /// @param legal_moves A bitset of size kNumMoves
  /// @return Whether there are any "lines" in the current position
  bool getLegalMoves(MoveMask &legal_moves) const noexcept;
  /// @brief Convenience overload for callers outside the hot path
  bool getLegalMoves(std::bitset<kNumMoves> &legal_moves) const noexcept;
  /// @brief Reference implementation of getLegalMoves, for differential
  /// testing
  /// @details A verbatim copy of the pre-bitboard implementation, with its own
  /// private copies of every helper so that a bug in a shared helper cannot
  /// corrupt both sides and hide itself. Defined in game_reference.cpp, which
  /// is built only by bench/Makefile and does not ship in the training module.
  /// Delete once the bitboard stages are complete and verified.
  bool
  getLegalMovesReference(std::bitset<kNumMoves> &legal_moves) const noexcept;

  /// @brief Whether a line stands on the board (cheaper than getLegalMoves)
  bool hasLine() const noexcept;
  /// @brief Legal moves when no line stands (the caller knows, e.g. from
  /// hasLine): placement and movement rules only, no line search
  void getLegalMovesNoLines(MoveMask &legal_moves) const noexcept;
  /// @brief An exact key for the position (board, both reserves, side to move)
  /// @details For transposition tables (solver.h). Two games have equal keys
  /// exactly when they are the same position.
  void key(uint64_t &board, uint64_t &rest) const noexcept;
  /// @brief The spaces that belong to a line on the board, by top type
  /// @details Bit i of by_type[t] is set when space i is part of a line
  /// topped by type t (base, column, capital). For network input features
  /// (pipeline/arch); the engine does not use it.
  void lineSpaces(uint16_t by_type[3]) const noexcept;
  /// @brief Write a representation of the game state to a float array
  /// @param game_state A float array of size kGameStateSize, used for input to
  /// the neural network
  void writeGameState(float game_state[kGameStateSize]) const noexcept;
  /// @brief Applies a move to the game state
  /// @warning Does not check if the move is legal
  /// @param move_id The ID of the move to apply
  void doMove(int32_t move_id) noexcept;

  friend std::ostream &operator<<(std::ostream &stream, const Game &game);

 private:
  /// @brief Per-space state, computed once per legal-move generation
  /// @details top(), bottom(), empty() and frozen() were each recomputed from
  /// the bitset on every query, and legal move generation queries them roughly
  /// 150 times per call: once per space in each of the four line detectors,
  /// and again inside canPlace/canMove for each of the 96 candidate moves. The
  /// board cannot change during a single generation, so the answers are
  /// computed once up front and read from here instead.
  ///
  /// Everything is a 16-bit plane, bit i for space i, extracted from the
  /// board word with shifts and masks. The per-space top and bottom are only
  /// needed on rare paths (line breaking, isLegalMove), so they are derived
  /// from the planes on demand rather than stored.
  struct SpaceInfo {
    /// @brief Bit per space, set when the space holds no pieces
    uint16_t empty;
    /// @brief Bit per space, set when the space is frozen
    uint16_t frozen;
    /// @brief Bit per space, set when that space CONTAINS this piece type
    /// @details Distinct from top_plane: a space holding a column and a
    /// capital contains a column but is topped by the capital. Placement
    /// rules ask about containment, line detection about topping.
    uint16_t has[3];
    /// @brief Bit per space, set when that space's top is this piece type
    /// @details Line detection works on these directly.
    uint16_t top_plane[3];

    /// @brief Index of the top piece of the stack on space, or -1 if empty
    int32_t top(int32_t space) const noexcept {
      const uint32_t bit = 1U << space;
      return (has[kCapital] & bit)  ? kCapital
             : (has[kColumn] & bit) ? kColumn
             : (has[kBase] & bit)   ? kBase
                                    : -1;
    }
    /// @brief Index of the bottom piece of the stack on space, or 3 if empty
    int32_t bottom(int32_t space) const noexcept {
      const uint32_t bit = 1U << space;
      return (has[kBase] & bit)      ? kBase
             : (has[kColumn] & bit)  ? kColumn
             : (has[kCapital] & bit) ? kCapital
                                     : 3;
    }
  };

  /// @brief Fill a SpaceInfo from the current board
  void computeSpaceInfo(SpaceInfo &info) const noexcept;

  /// @brief Private accessor for the board
  /// @details Finds the correct index in the bitset
  /// for a given row, column, and piece type
  /// This interface saves the hassle and error-proneness
  /// of manually calculating the index each time
  bool board(Space space, PieceType piece_type) const noexcept;
  /// @brief Checks if a space is frozen
  /// @return Whether the space is frozen
  bool frozen(Space space) const noexcept;
  /// @brief Checks if a space is empty
  /// @return Whether the space is empty
  bool empty(Space space) const noexcept;
  /// @brief Finds the top piece of a stack
  /// @details This is used in legal move generation
  /// @return The index of the top piece of the stack
  /// or -1 if the stack is empty
  int32_t top(Space space) const noexcept;
  /// @brief Finds the bottom piece of a stack
  /// @details This is used in legal move generation
  /// @return The index of the bottom piece of the stack
  /// or 3 if the stack is empty
  int32_t bottom(Space space) const noexcept;

  /// @brief Mutator for the board
  /// @param state Value to set the entry to
  /// @note Default is true to match bitset default
  void set_board(Space space, PieceType piece_type,
                 bool state = true) noexcept;
  /// @brief Mutator for the frozenness of a space
  /// @param state Value to set the entry to
  /// @note Default is true to match bitset default
  void set_frozen(Space space, bool state = true) noexcept;

  /// @brief Checks if a piece can be placed on the space
  /// @param piece_type The type of piece to place
  /// @return Whether the piece can be placed
  bool canPlace(const MoveInfo &move, const SpaceInfo &info) const noexcept;
  /// @brief Checks if a tower can be moved
  /// @return Whether the tower can be moved
  bool canMove(const MoveInfo &move, const SpaceInfo &info) const noexcept;
  /// @brief Checks if a move is legal according to basic rules
  /// @warning Does not check if the move is legal according to line breaking
  /// @param move_id The ID of the move to check
  /// @return Whether the move is legal
  bool isLegalMove(int32_t move_id, const SpaceInfo &info) const noexcept;
  /// @brief Overload that computes its own SpaceInfo, for the doMove assert
  bool isLegalMove(int32_t move_id) const noexcept;
  /// @brief A line present on the board, as a shape and the type topping it
  struct PresentLine {
    int8_t shape;
    int8_t type;
  };

  /// @brief Find every line shape currently holding a single top type
  /// @details No subsumption: a four and both of its threes are all recorded,
  /// because covering one end of a four leaves a three standing.
  /// @return The number of lines written to `out`
  int32_t findLines(const SpaceInfo &info, PresentLine *out) const noexcept;

  /// @brief Every move legal by placement and movement rules, ignoring lines
  /// @details All 48 placements reduce to three mask expressions and all 48
  /// moves to a shifted AND per direction, replacing 96 per-move calls.
  MoveMask basicLegalMoves(const SpaceInfo &info) const noexcept;

  /// @brief The set of moves that break every line currently on the board
  /// @details One mask per line, ANDed together, rather than testing every
  /// move against every line. Four of the five ways to break a line are
  /// constant per shape and type and come straight from kLineBreakTable. The
  /// fifth -- moving a stack onto the extending space -- is the only one that
  /// depends on the board, and is masked by which moves currently start from a
  /// space topped by the line's type.
  MoveMask lineBreakers(const SpaceInfo &info, const PresentLine *lines,
                        int32_t num_lines) const noexcept;

  /// @brief The Corintho game board, stored as a bitset.
  /// @details 4x4 board with 4 bits per space (3 for pieces, 1 for frozenness)
  /// @note Stored as a bitset for memory efficiency
  std::bitset<4 * kBoardSize> board_{};
  /// @brief The pieces that each player has available to place
  /// @details Stored as an array of 6 int8_t's for
  /// where the first 3 are the first player's pieces
  int8_t pieces_[6]{4, 4, 4, 4, 4, 4};
  /// @brief Whose turn it is to play (0 for first player, 1 for second player)
  /// @note This is stored as an int8_t instead of a bool
  /// so that arithmetic operations can be performed on it
  int8_t to_play_{};
};

#endif
