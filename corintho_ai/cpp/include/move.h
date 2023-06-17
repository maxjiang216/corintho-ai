#ifndef MOVE_H
#define MOVE_H

#include "util.h"
#include <ostream>

class Move {
public:
  enum MoveType { kPlace, kMove };

  Move() = delete;
  Move(const Move &move) = default;
  Move(Move &&move) = default;
  Move &operator=(const Move &move) = default;
  Move &operator=(Move &&move) = default;
  ~Move() = default;
  explicit Move(uintf move_id);

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
  int32_t row2_;
  int32_t col2_;
};

char get_col_name(int32_t col);
int32_t encode_place(int32_t ptype, int32_t row, int32_t col);
int32_t encode_move(int32_t row1, int32_t col1, int32_t row2, int32_t col2);

#endif
