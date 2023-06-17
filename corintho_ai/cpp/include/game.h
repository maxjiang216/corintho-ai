#ifndef GAME_H
#define GAME_H

#include <array>
#include <bitset>
#include <ostream>

#include "util.h"

class Game {
 public:
  /// The default constructor creates a game in the starting position
  Game() noexcept;
  Game(const Game &game) noexcept = default;
  Game(Game &&game) noexcept = default;
  Game &operator=(const Game &game) noexcept = default;
  Game &operator=(Game &&game) noexcept = default;
  ~Game() = default;
  // Create a new game from an arbitrary state
  // Used in the web app to feed an arbitrary game state to the MCST
  Game(long *board, int to_play, long *pieces);

  /// @brief Mutates legal_moves to indicate which moves are legal
  /// @param legal_moves A bitset of size NUM_MOVES
  /// @return Whether there are any "lines" in the current position
  bool get_legal_moves(std::bitset<NUM_MOVES> &legal_moves) const;

  /// @brief Write a representation of the game state to a float array
  /// @param game_state A float array of size GAME_STATE_SIZE, used for input to the neural network
  void write_game_state(float game_state[GAME_STATE_SIZE]) const;

  /// @brief Applies a move to the game state
  /// @param move_id The ID of the move to apply
  void do_move(int32_t move_id);

  friend std::ostream &operator<<(std::ostream &stream, const Game &game);

 private:
  // Includes the pieces and if the space is frozen
  // bitsets come in multiples of 8 byte sizes
  // So it saves 8 bytes to combine these
  // Bitset can save on memory and make resetting all bits faster
  std::bitset<4 * BOARD_SIZE> board;
  uint_least8_t to_play, pieces[6];

  // Finds lines and moves that break all lines
  // Returns whether there were lines
  bool get_line_breakers(std::bitset<NUM_MOVES> &legal_moves) const;

  // Returns an int representing the top of a stack, -1 if empty
  intf get_top(uintf row, uintf col) const;
  // Returns an int representing the bottom piece of a stack, 3 if empty
  intf get_bottom(uintf row, uintf col) const;

  bool get_board(uintf row, uintf col, uintf ptype) const;
  bool get_frozen(uintf row, uintf col) const;
  void set_board(uintf row, uintf col, uintf ptype, bool state = true);
  void set_frozen(uintf row, uintf col, bool state = true);

  void apply_line(uintf line, std::bitset<NUM_MOVES> &legal_moves) const;

  bool is_legal_move(int32_t move_id) const;
  bool can_place(uintf ptype, uintf row, uintf col) const;
  bool can_move(uintf row1, uintf col1, uintf row2, uintf col2) const;
  bool is_empty(uintf row, uintf col) const;

  friend class Node;
};

#endif
