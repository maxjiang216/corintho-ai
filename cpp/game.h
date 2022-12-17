#ifndef GAME_H
#define GAME_H

// Bitset can save on memory and make resetting all bits faster
#include "util.h"
#include <array>
#include <bitset>
#include <ostream>

using std::bitset;

const uintf BOARD_SIZE = 16;

class Game {

  bitset<3 * BOARD_SIZE> board;
  bitset<BOARD_SIZE> frozen;
  // We want to add orientation later
  uint_least8_t to_play, pieces[6];

  // Finds lines and moves that break all lines
  // Returns whether there were lines
  bool get_line_breakers(std::array<bool, NUM_TOTAL_MOVES> &legal_moves) const;

  // Returns an int representing the top of a stack, -1 if empty
  intf get_top(uintf row, uintf col) const;
  // Returns an int representing the bottom piece of a stack, 3 if empty
  intf get_bottom(uintf row, uintf col) const;

  void apply_line(uintf line, std::array<bool, NUM_TOTAL_MOVES> &legal_moves) const;

  bool is_legal_move(uintf move_id) const;
  bool can_place(uintf ptype, uintf row, uintf col) const;
  bool can_move(uintf row1, uintf col1, uintf row2, uintf col2) const;
  bool is_empty(uintf row, uintf col) const;

public:
  Game();
  Game(const Game &game) = default;
  ~Game() = default;

  bool is_terminal() const;

  void do_move(uintf move_id);

  // Returns whether there are lines
  bool get_legal_moves(std::array<bool, NUM_MOVES> &legal_moves) const;

  void write_game_state(float game_state[GAME_STATE_SIZE]) const;
  void write_game_state(std::array<float, GAME_STATE_SIZE> &game_state) const;

  friend std::ostream &operator<<(std::ostream &stream, const Game &game);
};

#endif
