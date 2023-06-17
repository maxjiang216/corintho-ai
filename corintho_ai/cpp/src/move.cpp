#include "move.h"

#include <cassert>

#include <stdlib.h>

Move::Move(int32_t id) noexcept
    : move_type_{id >= 48 ? MoveType::kPlace : MoveType::kMove} {
  assert(id >= 0 && id < 96);

  // Place move
  // This is more common
  // So checking it first should be slightly more efficient
  if (move_type_ == MoveType::kPlace) {
    piece_type_ = (id - 48) / 16;
    row1_ = (id % 16) / 4;
    col1_ = id % 4;
  }
  // Move move
  else if (id < 12) {  // Right
    row1_ = id / 3;
    col1_ = id % 3;
    row2_ = row1_;
    col2_ = col1_ + 1;
  } else if (id < 24) {  // Down
    row1_ = (id - 12) / 4;
    col1_ = id % 4;
    row2_ = row1_ + 1;
    col2_ = col1_;
  } else if (id < 36) {  // Left
    row1_ = (id - 24) / 3;
    col1_ = id % 3 + 1;
    row2_ = row1_;
    col2_ = col1_ - 1;
  } else {  // Up
    row1_ = (id - 32) / 4;
    col1_ = id % 4;
    row2_ = row1_ - 1;
    col2_ = col1_;
  }
}

int32_t encodePlace(int32_t piece_type, int32_t row, int32_t col) {
  assert(piece_type >= 0 && piece_type < 3);
  assert(row >= 0 && row < 4);
  assert(col >= 0 && col < 4);
  return 48 + piece_type * 16 + row * 4 + col;
}

int32_t encodeMove(int32_t row1, int32_t col1, int32_t row2, int32_t col2) {
  assert(row1 >= 0 && row1 < 4);
  assert(col1 >= 0 && col1 < 4);
  assert(row2 >= 0 && row2 < 4);
  assert(col2 >= 0 && col2 < 4);
  if (col1 < col2) {  // Right
    assert(row1 == row2);
    assert(col1 + 1 == col2);
    return row1 * 3 + col1;
  }
  if (row1 < row2) {  // Down
    assert(col1 == col2);
    assert(row1 + 1 == row2);
    return 12 + row1 * 4 + col1;
  }
  if (col1 > col2) {  // Left
    assert(row1 == row2);
    assert(col1 - 1 == col2);
    return 24 + row1 * 3 + (col1 - 1);
  }
  // Up
  assert(col1 == col2);
  assert(row1 - 1 == row2);
  return 36 + (row1 - 1) * 4 + col1;
}

char getColName(int32_t col) {
  switch (col) {
  case 0:
    return 'a';
  case 1:
    return 'b';
  case 2:
    return 'c';
  case 3:
    return 'd';
  default:  // Should never happen
    assert(false);
    return ' ';
  }
}

std::ostream &operator<<(std::ostream &os, const Move &move) {
  if (move.move_type_ == Move::MoveType::kPlace) {
    if (move.piece_type_ == 0) {
      os << "B";
    } else if (move.piece_type_ == 1) {
      os << "C";
    } else if (move.piece_type_ == 2) {
      os << "A";
    }
    os << getColName(move.col1_) << 4 - move.row1_;
  } else {
    os << getColName(move.col1_) << 4 - move.row1_;
    if (move.col2_ < move.col1_)
      os << 'L';
    else if (move.col2_ > move.col1_)
      os << 'R';
    else if (move.row2_ < move.row1_)
      os << 'U';
    else
      os << 'D';
  }
  return os;
}
