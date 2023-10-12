#ifndef MOVE_H
#define MOVE_H

#include <cstdint>

#include <ostream>

#include "util.h"

/**
 * @class Move
 * @brief This class represents a move in Corintho.
 *
 * There are two types of moves: place and move.
 * A place move places a piece on the board. It is specified by the piece type,
 * row, and column. A move move moves a piece on the board. It is specified by
 * the starting row and column and the destination row and column. Each move
 * has a unique ID (0-95). We use this to refer to moves in the Monte Carlo
 * Tree Search.
 */
class Move {
 public:
  enum class MoveType { kPlace, kMove };

  Move() = delete;
  Move(const Move &) noexcept = default;
  Move(Move &&) noexcept = default;
  Move &operator=(const Move &) noexcept = default;
  Move &operator=(Move &&) noexcept = default;
  ~Move() = default;
  /// @brief Construct a move from its ID
  explicit Move(int32_t id) noexcept;
  /// @brief Construct a place move
  Move(Space space, PieceType piece_type) noexcept;
  /// @brief Construct a move move with two spaces
  Move(Space spaceFrom, Space spaceTo) noexcept;

  MoveType move_type() const noexcept { return move_type_; }
  PieceType piece_type() const noexcept { return piece_type_; }
  /// @brief The row of the space being moved from, not used for place moves
  int32_t row_from() const noexcept { return space_from_.row; }
  /// @brief The column of the space being moved from, not used for place moves
  int32_t col_from() const noexcept { return space_from_.col; }
  /// @brief The space being moved from, not used for place moves
  Space space_from() const noexcept { return space_from_; }
  /// @brief The row of the space being moved to or placed on
  int32_t row_to() const noexcept { return space_to_.row; }
  /// @brief The column of the space being moved to or placed on
  int32_t col_to() const noexcept { return space_to_.col; }
  /// @brief The space being moved to or placed on
  Space space_to() const noexcept { return space_to_; }
  
  friend std::ostream &operator<<(std::ostream &os, const Move &move);

 private:
  const MoveType move_type_;
  PieceType piece_type_;
  Space space_from_{-1, -1};
  Space space_to_;
};

// Get the ID of a place move
int32_t encodePlace(Space space, PieceType piece_type) noexcept;
// Get the ID of a move move
int32_t encodeMove(Space spaceFrom, Space spaceTo) noexcept;
// Convert column index to its name (a, b, c, d)
char getColName(int32_t col);

#endif
