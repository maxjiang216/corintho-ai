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
  /// @brief Construct a move from its ID
  explicit Move(int32_t id) noexcept;
  /// @brief Construct a place move
  Move(Space space, PieceType piece_type) noexcept;
  /// @brief Construct a move move
  Move(Space spaceFrom, Space spaceTo) noexcept;

  MoveType move_type() const noexcept { return move_type_; }
  PieceType piece_type() const noexcept { return piece_type_; }
  /// @brief The row of the space being moved from, not used for place moves
  int32_t rowFrom() const noexcept { return spaceFrom_.row; }
  /// @brief The column of the space being moved from, not used for place moves
  int32_t colFrom() const noexcept { return spaceFrom_.row; }
  /// @brief The space being moved from, not used for place moves
  Space spaceFrom() const noexcept { return spaceFrom_; }
  /// @brief The row of the space being moved to or placed on
  int32_t rowTo() const noexcept { return spaceTo_.row; }
  /// @brief The column of the space being moved to or placed on
  int32_t colTo() const noexcept { return spaceTo_.row; }
  /// @brief The space being moved to or placed on
  Space spaceTo() const noexcept { return spaceTo_; }

  friend std::ostream &operator<<(std::ostream &os, const Move &move);

 private:
  MoveType move_type_;
  PieceType piece_type_;
  Space spaceFrom_{-1, -1};
  Space spaceTo_;
};

// Get the ID of a place move
int32_t encodePlace(Space space, PieceType piece_type);
// Get the ID of a move move
int32_t encodeMove(Space spaceFrom, Space spaceTo);
// Convert column index to its name (a, b, c, d)
char getColName(int32_t col);

#endif
