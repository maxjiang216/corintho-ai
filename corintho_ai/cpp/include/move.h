#ifndef MOVE_H
#define MOVE_H

#include <ostream>

#include "util.h"

class Move {
public:
  enum class MoveType { kPlace, kMove };
  enum class Direction { kRight, kUp, kLeft, kDown };

  Move() = delete;
  Move(const Move &move) noexcept = default;
  Move(Move &&move) noexcept = default;
  Move &operator=(const Move &move) noexcept = default;
  Move &operator=(Move &&move) noexcept = default;
  ~Move() = default;
  // Construct a move from its id
  explicit Move(int32_t id);
  // Construct a place move
  Move(int32_t piece_type, int32_t row, int32_t col);
  // Construct a move move
  Move(int32_t row, int32_t col, Direction direction);

  MoveType move_type() const noexcept { return move_type_; }
  int32_t piece_type() const noexcept { return piece_type_; }
  int32_t row1() const noexcept { return row1_; }
  int32_t col1() const noexcept { return col1_; }
  int32_t row2() const noexcept { return row2_; }

  friend std::ostream &operator<<(std::ostream &os, const Move &move);

private:
  MoveType move_type_;
  int32_t piece_type_;
  int32_t row1_;
  int32_t col1_;
  int32_t row2_{0};
  int32_t col2_{0};
};

// Convert place move to its id
int32_t encodePlace(int32_t ptype, int32_t row, int32_t col);
// Convert move move to its id
int32_t encodeMove(int32_t row1, int32_t col1, int32_t row2, int32_t col2);
// Convert column index to its name (a, b, c, d)
char getColName(int32_t col);

#endif
