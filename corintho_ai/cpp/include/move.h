#ifndef MOVE_H
#define MOVE_H

#include <ostream>

#include "util.h"

/**
 * @class Move
 * @brief This class represents a move in Corintho.
 *
 * There are two types of moves: place and move.
 * A place move places a piece on the board. It is specified by the piece type,
 * row, and column. A move move moves a piece on the board. It is specified by
 * the starting row and column and the destination row and column. Each move has
 * a unique ID (0-95). We use this to refer to moves in the Monte Carlo Tree
 * Search.
 */
class Move {
public:
  enum class MoveType { kPlace, kMove };

  Move() = delete;
  Move(const Move &move) noexcept = default;
  Move(Move &&move) noexcept = default;
  Move &operator=(const Move &move) noexcept = default;
  Move &operator=(Move &&move) noexcept = default;
  ~Move() = default;
  /// Construct a move from its ID
  explicit Move(int32_t id) noexcept;

  MoveType move_type() const noexcept { return move_type_; }
  int32_t piece_type() const noexcept { return piece_type_; }
  /// @brief The row of the space being moved from or placed on
  int32_t row1() const noexcept { return row1_; }
  /// @brief The column of the space being moved from or placed on
  int32_t col1() const noexcept { return col1_; }
  /// @brief The row of the space being moved to, not used for place moves
  int32_t row2() const noexcept { return row2_; }
  /// @brief The column of the space being moved to, not used for place moves
  int32_t col2() const noexcept { return col2_; }

  friend std::ostream &operator<<(std::ostream &os, const Move &move);

private:
  MoveType move_type_;
  int32_t piece_type_;
  int32_t row1_;
  int32_t col1_;
  int32_t row2_{-1};
  int32_t col2_{-1};
};

// Get the ID of a place move
int32_t encodePlace(int32_t ptype, int32_t row, int32_t col);
// Get the ID of a move move
int32_t encodeMove(int32_t row1, int32_t col1, int32_t row2, int32_t col2);
// Convert column index to its name (a, b, c, d)
char getColName(int32_t col);

#endif
