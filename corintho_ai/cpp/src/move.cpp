#include "move.h"

#include <cassert>

#include <ostream>
#include <stdlib.h>

#include "util.h"

Move::Move(int32_t id) noexcept
    : move_type_{id >= 48 ? MoveType::kPlace : MoveType::kMove} {
  assert(id >= 0 && id < kNumMoves);

  // Place move
  // This is more common
  // So checking it first should be slightly more efficient
  if (move_type_ == MoveType::kPlace) {
    piece_type_ = (id - 48) / 16;
    spaceTo_ = {(id % 16) / 4, id % 4};
    return;
  }
  // Move move
  if (id < 12) {  // Right
    spaceFrom_ = {id / 3, id % 3};
    spaceTo_ = {id / 3, id % 3 + 1};
    return;
  }
  if (id < 24) {  // Down
    spaceFrom_ = {(id - 12) / 4, id % 4};
    spaceTo_ = {(id - 12) / 4 + 1, id % 4};
    return;
  }
  if (id < 36) {  // Left
    spaceFrom_ = {(id - 24) / 3, id % 3 + 1};
    spaceTo_ = {(id - 24) / 3, id % 3};
    return;
  }
  // Up
  spaceFrom_ = {(id - 36) / 4 + 1, id % 4};
  spaceTo_ = {(id - 36) / 4, id % 4};
}

Move::Move(Space space, PieceType piece_type) noexcept
    : move_type_{MoveType::kPlace}, piece_type_{piece_type}, spaceTo_{space} {
  assert(space.notNull());
  assert(piece_type >= 0 && piece_type < 3);
}

Move::Move(Space spaceFrom, Space spaceTo) noexcept
    : move_type_{MoveType::kMove}, spaceFrom_{spaceFrom}, spaceTo_{spaceTo} {
  assert(spaceFrom.notNull());
  assert(spaceTo.notNull());
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
    os << getColName(move.spaceFrom_.col) << 4 - move.spaceFrom_.row;
  } else {
    os << getColName(move.spaceFrom_.col) << 4 - move.spaceFrom_.row;
    if (move.spaceTo_.col < move.spaceFrom_.col)
      os << 'L';
    else if (move.spaceTo_.col > move.spaceFrom_.col)
      os << 'R';
    else if (move.spaceTo_.row < move.spaceFrom_.row)
      os << 'U';
    else
      os << 'D';
  }
  return os;
}

int32_t encodePlace(Space space, PieceType piece_type) noexcept {
  assert(piece_type >= 0 && piece_type < 3);
  assert(space.notNull());
  return 48 + piece_type * 16 + space.row * 4 + space.col;
}

int32_t encodeMove(Space spaceFrom, Space spaceTo) noexcept {
  assert(spaceFrom.notNull());
  assert(spaceTo.notNull());
  if (spaceFrom.col < spaceTo.col) {  // Right
    assert(spaceFrom.row == spaceTo.row);
    assert(spaceFrom.col + 1 == spaceTo.col);
    return spaceFrom.row * 3 + spaceFrom.col;
  }
  if (spaceFrom.row < spaceTo.row) {  // Down
    assert(spaceFrom.col == spaceTo.col);
    assert(spaceFrom.row + 1 == spaceTo.row);
    return 12 + spaceFrom.row * 4 + spaceFrom.col;
  }
  if (spaceFrom.col > spaceTo.col) {  // Left
    assert(spaceFrom.row == spaceTo.row);
    assert(spaceFrom.col - 1 == spaceTo.col);
    return 24 + spaceFrom.row * 3 + (spaceFrom.col - 1);
  }
  // Up
  assert(spaceFrom.col == spaceTo.col);
  assert(spaceFrom.row - 1 == spaceTo.row);
  return 36 + (spaceFrom.row - 1) * 4 + spaceFrom.col;
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