#ifndef GAME_H
#define GAME_H

// Bitset can save on memory and make resetting all bits faster
#include "util.h"
#include <array>
#include <bitset>
#include <ostream>

using std::bitset;

class Game {

  // Includes the pieces and if the space is frozen
  // bitsets come in multiples of 8 byte sizes
  // So it saves 8 bytes to combine these
  bitset<4 * BOARD_SIZE> board;
  uint_least8_t to_play, pieces[6];

  // Finds lines and moves that break all lines
  // Returns whether there were lines
  bool get_line_breakers(bitset<NUM_MOVES> &legal_moves) const;

  // Returns an int representing the top of a stack, -1 if empty
  intf get_top(uintf row, uintf col) const;
  // Returns an int representing the bottom piece of a stack, 3 if empty
  intf get_bottom(uintf row, uintf col) const;

  bool get_board(uintf row, uintf col, uintf ptype) const;
  bool get_frozen(uintf row, uintf col) const;
  void set_board(uintf row, uintf col, uintf ptype, bool state = true);
  void set_frozen(uintf row, uintf col, bool state = true);

  void apply_line(uintf line, bitset<NUM_MOVES> &legal_moves) const;

  bool is_legal_move(int32_t move_id) const;
  bool can_place(uintf ptype, uintf row, uintf col) const;
  bool can_move(uintf row1, uintf col1, uintf row2, uintf col2) const;
  bool is_empty(uintf row, uintf col) const;

public:
  Game();
  Game(const Game &game) = default;
  // Create a new game from arbitrary state
  // For web app
  Game(long *board, int to_play, long *pieces);
  ~Game() = default;

  void do_move(int32_t move_id);

  // Returns whether there are lines
  bool get_legal_moves(bitset<NUM_MOVES> &legal_moves) const;

  void write_game_state(float game_state[GAME_STATE_SIZE]) const;

  friend std::ostream &operator<<(std::ostream &stream, const Game &game);

  friend class Node;
};

#endif
