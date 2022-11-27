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
    : board{}, frozen{}, to_play{0}, pieces{4, 4, 4, 4, 4, 4}, result{NONE} {}

bool Game::is_legal(uintf move_choice) const {
  return legal_moves[move_choice];
}

uintf Game::get_to_play() const { return to_play; }

Result Game::get_result() const { return result; }

bool Game::is_terminal() const { return result != NONE; }

void Game::do_move(uintf move_id) {

  Move move{move_id};

  // Reset which space is frozen
  frozen.reset();

  // Place
  if (move.mtype) {
    --pieces[to_play * 3 + move.ptype];
    board.set(move.row1 * 12 + move.col1 * 3 + move.ptype);
    frozen.set(move.row1 * 4 + move.col1);
  }
  // Move
  else {
    for (uintf i = 0; i < 3; ++i) {
      board.set(move.row2 * 12 + move.col2 * 3 + i,
                board[move.row1 * 12 + move.col1 * 3 + i] ||
                    board[move.row2 * 12 + move.col2 * 3 + i]);
      board.reset(move.row1 * 12 + move.col1 * 3 + i);
    }
    frozen.set(move.row2 * 4 + move.col2);
  }
  to_play = 1 - to_play;

  get_legal_moves();
}

void Game::write_game_state(float game_state[GAME_STATE_SIZE]) const {
  for (uintf i = 0; i < 3 * BOARD_SIZE; ++i) {
    if (board[i]) {
      game_state[i] = 1.0;
    } else {
      game_state[i] = 0.0;
    }
  }
  for (uintf i = 0; i < BOARD_SIZE; ++i) {
    if (frozen[i]) {
      game_state[3 * BOARD_SIZE + i] = 1.0;
    } else {
      game_state[3 * BOARD_SIZE + i] = 0.0;
    }
  }
  // Canonize the pieces
  for (uintf i = 0; i < 6; ++i) {
    game_state[4 * BOARD_SIZE + i] =
        (float)pieces[(to_play * 3 + i) % 6] * 0.25;
  }
}

void Game::write_game_state(
    std::array<float, GAME_STATE_SIZE> &game_state) const {
  for (uintf i = 0; i < 3 * BOARD_SIZE; ++i) {
    if (board[i]) {
      game_state[i] = 1.0;
    } else {
      game_state[i] = 0.0;
    }
  }
  for (uintf i = 0; i < BOARD_SIZE; ++i) {
    if (frozen[i]) {
      game_state[3 * BOARD_SIZE + i] = 1.0;
    } else {
      game_state[3 * BOARD_SIZE + i] = 0.0;
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
      if (game.board[i * 12 + j * 3 + 0])
        os << 'B';
      else
        os << ' ';
      if (game.board[i * 12 + j * 3 + 1])
        os << 'C';
      else
        os << ' ';
      if (game.board[i * 12 + j * 3 + 2])
        os << 'A';
      else
        os << ' ';
      if (game.frozen[i * 4 + j])
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

void Game::get_legal_moves() {

  // Set all bits to 1
  // Since apply_line does &=, we need to do this
  legal_moves.set();

  // Filter out moves that don't break lines
  bool is_lines = get_line_breakers();

  // Apply other rules
  for (uintf i = 0; i < NUM_MOVES; ++i) {
    if (legal_moves[i] && !is_legal_move(i)) {
      legal_moves.reset(i);
    }
  }

  // Terminal state
  if (legal_moves.none()) {
    // There is a line
    if (is_lines) {
      // Person to play is lost
      if (to_play == 0) {
        result = LOSS;
      } else {
        result = WIN;
      }
    } else {
      result = DRAW;
    }
  }
}

bool Game::get_line_breakers() {

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
            apply_line(RB * 12 + i * 3 + space_0);
          } else {
            is_lines = true;
            apply_line(RL * 12 + i * 3 + space_0);
            // add the capital case
          }
        } else if (space_2 == space_3) {
          is_lines = true;
          apply_line(RR * 12 + i * 3 + space_3);
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
            apply_line(CB * 12 + j * 3 + space_0);
          } else {
            is_lines = true;
            apply_line(CU * 12 + j * 3 + space_0);
          }
        } else if (space_2 == space_3) {
          is_lines = true;
          apply_line(CD * 12 + j * 3 + space_3);
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
          apply_line(72 + D0B * 3 + space_0);
        } else {
          is_lines = true;
          apply_line(72 + D0U * 3 + space_0);
        }
      } else if (space_2 == space_3) {
        is_lines = true;
        apply_line(72 + D0D * 3 + space_3);
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
          apply_line(72 + D1B * 3 + space_0);
        } else {
          is_lines = true;
          apply_line(72 + D1U * 3 + space_0);
        }
      } else if (space_2 == space_3) {
        is_lines = true;
        apply_line(72 + D1D * 3 + space_3);
      }
    }
  }

  // Top left short diagonal
  space_1 = get_top(1, 1);
  if (space_1 != -1 && space_1 == get_top(0, 2) && space_1 == get_top(2, 0)) {
    is_lines = true;
    apply_line(72 + S0 * 3 + space_1);
  }

  // Top right short diagonal
  space_1 = get_top(1, 2);
  if (space_1 != -1 && space_1 == get_top(0, 1) && space_1 == get_top(2, 3)) {
    is_lines = true;
    apply_line(72 + S1 * 3 + space_1);
  }

  // Bottom right short diagonal
  space_1 = get_top(2, 2);
  if (space_1 != -1 && space_1 == get_top(1, 3) && space_1 == get_top(3, 1)) {
    is_lines = true;
    apply_line(72 + S2 * 3 + space_1);
  }

  // Bottom left short diagonal
  space_1 = get_top(2, 1);
  if (space_1 != -1 && space_1 == get_top(1, 0) && space_1 == get_top(3, 2)) {
    is_lines = true;
    apply_line(72 + S3 * 3 + space_1);
  }

  return is_lines;
}

intf Game::get_top(uintf row, uintf col) const {
  if (board[row * 12 + col * 3 + 2])
    return 2;
  if (board[row * 12 + col * 3 + 1])
    return 1;
  if (board[row * 12 + col * 3 + 0])
    return 0;
  // Empty
  return -1;
}

intf Game::get_bottom(uintf row, uintf col) const {
  if (board[row * 12 + col * 3 + 0])
    return 0;
  if (board[row * 12 + col * 3 + 1])
    return 1;
  if (board[row * 12 + col * 3 + 2])
    return 2;
  // Empty
  return 3;
}

void Game::apply_line(uintf line) { legal_moves &= line_breakers[line]; }

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
  if (frozen[row * 4 + col])
    return false;
  // Bases can only be placed on empty spaces
  if (ptype == 0)
    return false;
  // Column
  // Check for absence of column and capital
  if (ptype == 1) {
    return !(board[row * 12 + col * 3 + 1] || board[row * 12 + col * 3 + 2]);
  }
  // Capital
  // Check for absence of base without column and capital
  return !(board[row * 12 + col * 3 + 2] ||
           (board[row * 12 + col * 3 + 0] && !board[row * 12 + col * 3 + 1]));
}

bool Game::can_move(uintf row1, uintf col1, uintf row2, uintf col2) const {
  // Empty spaces, move moves not possible
  if (is_empty(row1, col1) || is_empty(row2, col2))
    return false;
  // Frozen spaces
  if (frozen[row1 * 4 + col1] || frozen[row2 * 4 + col2])
    return false;
  // The bottom of the first stack must go on the top of the second
  return get_bottom(row1, col1) - get_top(row2, col2) == 1;
}

bool Game::is_empty(uintf row, uintf col) const {
  return !(board[row * 12 + col * 3 + 0] || board[row * 12 + col * 3 + 1] ||
           board[row * 12 + col * 3 + 2]);
}