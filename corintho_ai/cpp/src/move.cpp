#include "move.h"

#include <cassert>
#include <cstdint>

#include <ostream>
#include <stdlib.h>

#include "util.h"

Move::Move(int32_t id) noexcept
    : move_type_{id >= 48 ? MoveType::kPlace : MoveType::kMove} {
  assert(id >= 0 && id < kNumMoves);
  // One table load instead of several integer divisions by 3 behind a
  // data-dependent branch chain. kMoveTable is generated at compile time from
  // the same formulas this used to evaluate, so it cannot drift.
  const MoveInfo &info = kMoveTable[id];
  if (info.is_place) {
    piece_type_ = info.piece;
    space_to_ = Space{info.to / 4, info.to % 4};
    return;
  }
  space_from_ = Space{info.from / 4, info.from % 4};
  space_to_ = Space{info.to / 4, info.to % 4};
}

Move::Move(Space space, PieceType piece_type) noexcept
    : move_type_{MoveType::kPlace}, piece_type_{piece_type}, space_to_{space} {
  assert(space.notNull());
  assert(piece_type >= 0 && piece_type < 3);
}

Move::Move(Space spaceFrom, Space spaceTo) noexcept
    : move_type_{MoveType::kMove}, space_from_{spaceFrom}, space_to_{spaceTo} {
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
    os << getColName(move.space_to_.col) << 4 - move.space_to_.row;
  } else {
    os << getColName(move.space_from_.col) << 4 - move.space_from_.row;
    if (move.space_to_.col < move.space_from_.col)
      os << 'L';
    else if (move.space_to_.col > move.space_from_.col)
      os << 'R';
    else if (move.space_to_.row < move.space_from_.row)
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