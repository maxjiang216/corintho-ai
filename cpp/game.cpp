#include "game.h"
#include "move.h"
#include "util.h"
#include <ostream>

// Indexing for lines
const uintf RL = 0;
const uintf RR = 1;
const uintf RB = 2;
const uintf CU = 3;
const uintf CD = 4;
const uintf CB = 5;

const uintf D0U = 0;
const uintf D0D = 1;
const uintf D0B = 2;
const uintf D1U = 3;
const uintf D1D = 4;
const uintf D1B = 5;
const uintf S0 = 6;
const uintf S1 = 7;
const uintf S2 = 8;
const uintf S3 = 9;

Game::Game()
    : to_play{0}, pieces{4, 4, 4, 4, 4, 4} {}

void Game::do_move(uintf move_id) {

  Move move{move_id};

  // Reset which space is frozen
  for (uintf row = 0; row < 4; ++row) {
    for (uintf col = 0; col < 4; ++col) {
      set_frozen(row, col, false);
    }
  }

  // Place
  if (move.mtype) {
    --pieces[to_play * 3 + move.ptype];
    set_board(move.row1, move.col1, move.ptype);
    set_frozen(move.row1, move.col1);
  }
  // Move
  else {
    for (uintf i = 0; i < 3; ++i) {
      set_board(move.row2, move.col2, i,
                board[move.row1 * 12 + move.col1 * 3 + i] ||
                    board[move.row2 * 12 + move.col2 * 3 + i]);
      set_board(move.row1, move.col1, i, false);
    }
    set_frozen(move.row2, move.col2, true);
  }
  to_play = 1 - to_play;
}

bool Game::get_legal_moves(std::array<bool, NUM_MOVES> &legal_moves) const {

  // Find line breakers using full set of moves
  bitset<NUM_LEGAL_MOVES> full_legal_moves;
  full_legal_moves.set();

  // Filter out moves that don't break lines
  bool is_lines = get_line_breakers(full_legal_moves);

  // Apply other rules
  for (uintf i = 0; i < NUM_MOVES; ++i) {
    if (full_legal_moves[i] && !is_legal_move(i)) {
      full_legal_moves[i] = false;
    }
  }

  // Shrink legal moves
  

  // Node will decide if it is a terminal state and if so the game result
  return is_lines;
}

void Game::write_game_state(
    std::array<float, GAME_STATE_SIZE> &game_state) const {
  for (uintf i = 0; i < 4 * BOARD_SIZE; ++i) {
    if (board[i]) {
      game_state[i] = 1.0;
    } else {
      game_state[i] = 0.0;
    }
  }
  // Canonize the pieces
  for (uintf i = 0; i < 6; ++i) {
    game_state[4 * BOARD_SIZE + i] =
        (float)pieces[(to_play * 3 + i) % 6] * 0.25;
  }
}

std::ostream &operator<<(std::ostream &os, const Game &game) {

  // Print board
  for (uintf i = 0; i < 4; ++i) {
    for (uint j = 0; j < 4; ++j) {
      if (game.get_board(i, j, 0))
        os << 'B';
      else
        os << ' ';
      if (game.get_board(i, j, 1))
        os << 'C';
      else
        os << ' ';
      if (game.get_board(i, j, 3))
        os << 'A';
      else
        os << ' ';
      if (game.get_frozen(i, j))
        os << '#';
      else
        os << ' ';
      if (j < 3)
        os << '|';
    }
    if (i < 3)
      os << "\n-------------------\n";
  }
  os << '\n';
  // Print pieces
  for (uintf i = 0; i < 2; ++i) {
    os << "Player " << i + 1 << ": ";
    os << "B: " << (uintf)game.pieces[i * 3 + 0] << ' ';
    os << "C: " << (uintf)game.pieces[i * 3 + 1] << ' ';
    os << "A: " << (uintf)game.pieces[i * 3 + 2] << '\n';
  }
  os << "Player " << game.to_play + 1 << " to play";

  return os;
}

bool Game::get_line_breakers(bitset<NUM_TOTAL_MOVES> &legal_moves) const {

  bool is_lines = false;
  intf space_0, space_1, space_2, space_3;

  // Rows
  for (uintf i = 0; i < 4; ++i) {
    space_1 = get_top(i, 1);
    if (space_1 != -1) { // If not empty
      space_2 = get_top(i, 2);
      if (space_1 == space_2) {
        space_0 = get_top(i, 0), space_3 = get_top(i, 3);
        if (space_0 == space_2) {
          if (space_0 == space_3) {
            is_lines = true;
            apply_line(RB * 12 + i * 3 + space_0, legal_moves);
          } else {
            is_lines = true;
            apply_line(RL * 12 + i * 3 + space_0, legal_moves);
            if (space_0 == 2) {
              // Capital must be used to extend line when moving
              // We don't need to know which row it is
              // If it is wrong, the move is illegal anyways
              if (!get_board(0, 3, 2)) {
                legal_moves[encode_move(0, 3, 1, 3)] = false;
              }
              if (!get_board(1, 3, 2)) {
                legal_moves[encode_move(1, 3, 0, 3)] = false;
                legal_moves[encode_move(1, 3, 2, 3)] = false;
              }
              if (!get_board(2, 3, 2)) {
                legal_moves[encode_move(2, 3, 1, 3)] = false;
                legal_moves[encode_move(2, 3, 3, 3)] = false;
              }
              if (!get_board(3, 3 ,2)) {
                legal_moves[encode_move(3, 3, 2, 3)] = false;
              }
            }
          }
        } else if (space_2 == space_3) {
          is_lines = true;
          apply_line(RR * 12 + i * 3 + space_3, legal_moves);
          if (space_3 == 2) {
            // Capital must be used to extend line when moving
            if (!get_board(0, 0, 2)) {
              legal_moves[encode_move(0, 0, 1, 0)] = false;
            }
            if (!get_board(1, 0, 2)) {
              legal_moves[encode_move(1, 0, 0, 0)] = false;
              legal_moves[encode_move(1, 0, 2, 0)] = false;
            }
            if (!get_board(2, 0, 2)) {
              legal_moves[encode_move(2, 0, 1, 0)] = false;
              legal_moves[encode_move(2, 0, 3, 0)] = false;
            }
            if (!get_board(3, 0, 2)) {
              legal_moves[encode_move(3, 0, 2, 0)] = false;
            }
          }
        }
      }
    }
  }

  // Columns
  for (uintf j = 0; j < 4; ++j) {
    space_1 = get_top(1, j);
    if (space_1 != -1) { // If not empty
      space_2 = get_top(2, j);
      if (space_1 == space_2) {
        space_0 = get_top(0, j), space_3 = get_top(3, j);
        if (space_0 == space_2) {
          if (space_0 == space_3) {
            is_lines = true;
            apply_line(CB * 12 + j * 3 + space_0, legal_moves);
          } else {
            is_lines = true;
            apply_line(CU * 12 + j * 3 + space_0, legal_moves);
            if (space_0 == 2) {
              // Capital must be used to extend line when moving
              // We don't need to know which column it is
              // If it is wrong, the move is illegal anyways
              if (!get_board(3, 0, 2)) {
                legal_moves[encode_move(3, 0, 3, 1)] = false;
              }
              if (!get_board(3, 1, 2)) {
                legal_moves[encode_move(3, 1, 3, 0)] = false;
                legal_moves[encode_move(3, 1, 3, 2)] = false;
              }
              if (!get_board(3, 2, 2)) {
                legal_moves[encode_move(3, 2, 3, 1)] = false;
                legal_moves[encode_move(3, 2, 3, 3)] = false;
              }
              if (!get_board(3, 3, 2)) {
                legal_moves[encode_move(3, 3, 3, 2)];
              }
            }
          }
        } else if (space_2 == space_3) {
          is_lines = true;
          apply_line(CD * 12 + j * 3 + space_3, legal_moves);
          if (space_3 == 2) {
            // Capital must be used to extend line when moving
            // We don't need to know which column it is
            // If it is wrong, the move is illegal anyways
            if (!get_board(0, 0, 2)) {
              legal_moves[encode_move(0, 0, 0, 1)] = false;
            }
            if (!get_board(0, 1, 2)) {
              legal_moves[encode_move(0, 1, 0, 0)] = false;
              legal_moves[encode_move(0, 1, 0, 2)] = false;
            }
            if (!get_board(0, 2, 2)) {
              legal_moves[encode_move(0, 2, 0, 1)] = false;
              legal_moves[encode_move(0, 2, 0, 3)] = false;
            }
            if (!get_board(0, 3, 2)) {
              legal_moves[encode_move(0, 3, 0, 2)] = false;
            }
          }
        }
      }
    }
  }

  // Upper left to lower right long diagonal
  space_1 = get_top(1, 1);
  if (space_1 != -1) { // If not empty
    space_2 = get_top(2, 2);
    if (space_1 == space_2) {
      space_0 = get_top(0, 0), space_3 = get_top(3, 3);
      if (space_0 == space_2) {
        if (space_0 == space_3) {
          is_lines = true;
          apply_line(72 + D0B * 3 + space_0, legal_moves);
        } else {
          is_lines = true;
          apply_line(72 + D0U * 3 + space_0, legal_moves);
          if (space_0 == 2) {
            // Capital must be used to extend line when moving
            if (!get_board(2, 3, 2)) {
              legal_moves[encode_move(2, 3, 3, 3)] = false;
            }
            if (!get_board(3, 2, 2)) {
              legal_moves[encode_move(3, 2, 3, 3)] = false;
            }
          }
        }
      } else if (space_2 == space_3) {
        is_lines = true;
        apply_line(72 + D0D * 3 + space_3, legal_moves);
        if (space_3 == 2) {
          // Capital must be used to extend line when moving
          if (!get_board(0, 1, 2)) {
            legal_moves[encode_move(0, 1, 0, 0)] = false;
          }
          if (!get_board(1, 0, 2)) {
            legal_moves[encode_move(1, 0, 0, 0)] = false;
          }
        }
      }
    }
  }

  // Upper right to lower left long diagonal
  space_1 = get_top(1, 2);
  if (space_1 != -1) { // If not empty
    space_2 = get_top(2, 1);
    if (space_1 == space_2) {
      space_0 = get_top(0, 3), space_3 = get_top(3, 0);
      if (space_0 == space_2) {
        if (space_0 == space_3) {
          is_lines = true;
          apply_line(72 + D1B * 3 + space_0, legal_moves);
        } else {
          is_lines = true;
          apply_line(72 + D1U * 3 + space_0, legal_moves);
          if (space_0 == 2) {
            // Capital must be used to extend line when moving
            if (!get_board(2, 0, 2)) {
              legal_moves[encode_move(2, 0, 3, 0)] = false;
            }
            if (!get_board(3, 1, 2)) {
              legal_moves[encode_move(3, 1, 3, 0)] = false;
            }
          }
        }
      } else if (space_2 == space_3) {
        is_lines = true;
        apply_line(72 + D1D * 3 + space_3, legal_moves);
        if (space_3 == 2) {
          // Capital must be used to extend line when moving
          if (!get_board(0, 2, 2)) {
            legal_moves[encode_move(0, 2, 0, 3)] = false;
          }
          if (!get_board(1, 3, 2)) {
            legal_moves[encode_move(1, 3, 0, 3)] = false;
          }
        }
      }
    }
  }

  // Top left short diagonal
  space_1 = get_top(1, 1);
  if (space_1 != -1 && space_1 == get_top(0, 2) && space_1 == get_top(2, 0)) {
    is_lines = true;
    apply_line(72 + S0 * 3 + space_1, legal_moves);
  }

  // Top right short diagonal
  space_1 = get_top(1, 2);
  if (space_1 != -1 && space_1 == get_top(0, 1) && space_1 == get_top(2, 3)) {
    is_lines = true;
    apply_line(72 + S1 * 3 + space_1, legal_moves);
  }

  // Bottom right short diagonal
  space_1 = get_top(2, 2);
  if (space_1 != -1 && space_1 == get_top(1, 3) && space_1 == get_top(3, 1)) {
    is_lines = true;
    apply_line(72 + S2 * 3 + space_1, legal_moves);
  }

  // Bottom left short diagonal
  space_1 = get_top(2, 1);
  if (space_1 != -1 && space_1 == get_top(1, 0) && space_1 == get_top(3, 2)) {
    is_lines = true;
    apply_line(72 + S3 * 3 + space_1, legal_moves);
  }

  return is_lines;
}

intf Game::get_top(uintf row, uintf col) const {
  for (uintf i = 0; i < 3; ++i) {
    if (get_board(row, col, 2-i))
      return 2-i;
  }
  // Empty
  return -1;
}

intf Game::get_bottom(uintf row, uintf col) const {
  for (uintf i = 0; i < 3; ++i) {
    if (get_board(row, col, i))
      return i;
  }
  // Empty
  return 3;
}

bool Game::get_board(uintf row, uintf col, uintf ptype) const {
  return board[row * 16 + col * 4 + ptype];
}

bool Game::get_frozen(uintf row, uintf col) const {
  return board[row * 16 + col * 4 + 3];
}

void Game::set_board(uintf row, uintf col, uintf ptype, bool state) {
  board[row * 16 + col * 4 + ptype] = state;
}

void Game::set_frozen(uintf row, uintf col, bool state) {
  board[row * 16 + col * 4 + 3] = state;
}

void Game::apply_line(uintf line, bitset<NUM_TOTAL_MOVES> &legal_moves) const { legal_moves &= line_breakers[line]; }

bool Game::is_legal_move(uintf move_id) const {
  Move move{move_id};
  // Place
  if (move.mtype)
    return can_place(move.ptype, move.row1, move.col1);
  // Move
  return can_move(move.row1, move.col1, move.row2, move.col2);
}

bool Game::can_place(uintf ptype, uintf row, uintf col) const {

  // Check if player has the piece left
  if (pieces[to_play * 3 + ptype] == 0)
    return false;
  // Check if space is empty
  // This is more common than frozen spaces
  // An empty space cannot be frozen
  if (is_empty(row, col))
    return true;
  // Check if space is frozen
  if (get_frozen(row, col))
    return false;
  // Bases can only be placed on empty spaces
  if (ptype == 0)
    return false;
  // Column
  // Check for absence of column and capital
  if (ptype == 1) {
    return !(get_board(row, col, 1) || get_board(row, col, 2));
  }
  // Capital
  // Check for absence of base without column and capital
  return !(get_board(row, col, 2) ||
           (get_board(row, col, 0) && !get_board(row, col, 1)));
}

bool Game::can_move(uintf row1, uintf col1, uintf row2, uintf col2) const {
  // Empty spaces, move moves not possible
  if (is_empty(row1, col1) || is_empty(row2, col2))
    return false;
  // Frozen spaces
  if (get_frozen(row1, col1) || get_frozen(row2, col2))
    return false;
  // The bottom of the first stack must go on the top of the second
  return get_bottom(row1, col1) - get_top(row2, col2) == 1;
}

bool Game::is_empty(uintf row, uintf col) const {
  return !(get_board(row, col, 0) || get_board(row, col, 1) ||
           get_board(row, col, 2));
}