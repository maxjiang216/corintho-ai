#include "move.h"
#include <stdlib.h>

// Convert move_id to Move object
// We can make this a table in the future, especially if we incude rotations (?)
Move::Move(uintf move_id) : mtype{move_id >= 48} {

  // Place
  // I believe that place is more common
  if (mtype) {
    ptype = (move_id - 48) / 16;
    row1 = (move_id % 16) / 4;
    col1 = move_id % 4;
  }

  // Move

  // Right
  else if (move_id < 12) {
    row1 = move_id / 3;
    col1 = move_id % 3;
    row2 = row1;
    col2 = col1 + 1;
  }

  // Down
  else if (move_id < 24) {
    row1 = (move_id - 12) / 4;
    col1 = move_id % 4;
    row2 = row1 + 1;
    col2 = col1;
  }

  // Left
  else if (move_id < 36) {
    row1 = (move_id - 24) / 3;
    col1 = move_id % 3 + 1;
    row2 = row1;
    col2 = col1 - 1;
  }

  // Up
  else {
    row1 = (move_id - 32) / 4;
    col1 = move_id % 4;
    row2 = row1 - 1;
    col2 = col1;
  }
}

// Convert place move to int
uintf encode_place(uintf ptype, uintf row, uintf col) {

  return 48 + ptype * 16 + row * 4 + col;
}

// Convert move move to int
uintf encode_move(uintf row1, uintf col1, uintf row2, uintf col2) {

  // Right
  if (col1 < col2)
    return row1 * 3 + col1;

  // Down
  if (row1 < row2)
    return 12 + row1 * 4 + col1;

  // Left
  if (col1 > col2)
    return 24 + row1 * 3 + (col1 - 1);

  // Up
  return 36 + (row1 - 1) * 4 + col1;
}

char get_col_name(uintf col) {
  if (col == 0)
    return 'a';
  if (col == 1)
    return 'b';
  if (col == 2)
    return 'c';
  return 'd';
}

std::ostream &operator<<(std::ostream &os, const Move &move) {

  if (move.mtype) {
    if (move.ptype == 0) {
      os << "B";
    } else if (move.ptype == 1) {
      os << "C";
    } else if (move.ptype == 2) {
      os << "A";
    }
    os << get_col_name(move.col1) << 4 - move.row1;
  } else {
    os << get_col_name(move.col1) << 4 - move.row1;
    if (move.col2 < move.col1)
      os << 'L';
    else if (move.col2 > move.col1)
      os << 'R';
    else if (move.row2 < move.row1)
      os << 'U';
    else
      os << 'D';
  }

  return os;
}