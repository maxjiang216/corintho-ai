#ifndef MOVE_H
#define MOVE_H

#include <cstdint>

#include <array>
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
/// @brief A move decoded into flat board indices
/// @details The hot paths -- legality and line breaking -- want cell indices,
/// not a Move object. Decoding an ID arithmetically costs several integer
/// divisions by 3 behind a data-dependent branch chain, and it was measured at
/// 10.7% of all instructions once it stopped being inlined.
struct MoveInfo {
  /// @brief Board index of the source, row * 4 + col, or -1 for a place
  int8_t from;
  /// @brief Board index of the destination, row * 4 + col
  int8_t to;
  /// @brief Piece placed, or -1 for a move-move
  int8_t piece;
  /// @brief True for a place, false for a move-move
  bool is_place;
};

/// @brief Build the move table by running the ID arithmetic at compile time
/// @details Deliberately NOT hand-written. A transcribed constant table is how
/// line_breakers acquired thirteen transposition errors (see
/// worklog/RULES-CHECKLIST.md item 4); generating it from the same formulas the
/// decoder uses means it cannot disagree with the encoding.
constexpr std::array<MoveInfo, kNumMoves> makeMoveTable() {
  std::array<MoveInfo, kNumMoves> table{};
  for (int32_t id = 0; id < kNumMoves; ++id) {
    if (id >= 48) {  // Place
      table[id] = MoveInfo{-1,
                           static_cast<int8_t>(((id % 16) / 4) * 4 + id % 4),
                           static_cast<int8_t>((id - 48) / 16), true};
    } else if (id < 12) {  // Right
      const int32_t r = id / 3, c = id % 3;
      table[id] = MoveInfo{static_cast<int8_t>(r * 4 + c),
                           static_cast<int8_t>(r * 4 + c + 1), -1, false};
    } else if (id < 24) {  // Down
      const int32_t r = (id - 12) / 4, c = id % 4;
      table[id] = MoveInfo{static_cast<int8_t>(r * 4 + c),
                           static_cast<int8_t>((r + 1) * 4 + c), -1, false};
    } else if (id < 36) {  // Left
      const int32_t r = (id - 24) / 3, c = id % 3 + 1;
      table[id] = MoveInfo{static_cast<int8_t>(r * 4 + c),
                           static_cast<int8_t>(r * 4 + c - 1), -1, false};
    } else {  // Up
      const int32_t r = (id - 36) / 4 + 1, c = id % 4;
      table[id] = MoveInfo{static_cast<int8_t>(r * 4 + c),
                           static_cast<int8_t>((r - 1) * 4 + c), -1, false};
    }
  }
  return table;
}

/// @brief Every move, decoded. 96 entries of 4 bytes; L1-resident.
inline constexpr std::array<MoveInfo, kNumMoves> kMoveTable = makeMoveTable();

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
