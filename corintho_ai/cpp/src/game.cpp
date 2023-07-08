#include "game.h"

#include <cassert>
#include <cstdint>

#include <bitset>
#include <ostream>

#include <gsl/gsl>

#include "move.h"
#include "util.h"

Game::Game(int32_t board[4 * kBoardSize], int32_t to_play,
           int32_t pieces[6]) noexcept
    : to_play_{gsl::narrow_cast<int8_t>(to_play)} {
  assert(to_play == 0 || to_play == 1);
  for (int32_t i = 0; i < 4 * kBoardSize; ++i) {
    assert(board[i] == 0 || board[i] == 1);
    this->board_[i] = board[i] != 0;
  }
  for (int32_t i = 0; i < 6; ++i) {
    assert(pieces[i] >= 0 && pieces[i] <= 4);
    this->pieces_[i] = gsl::narrow_cast<int8_t>(pieces[i]);
  }
}

bool Game::getLegalMoves(std::bitset<kNumMoves> &legal_moves) const noexcept {
  // First set all moves to legal
  legal_moves.set();
  // Filter out moves that don't break lines
  bool is_lines = applyLines(legal_moves);
  // Apply other rules
  for (int32_t i = 0; i < kNumMoves; ++i) {
    if (legal_moves[i] && !isLegalMove(i)) {
      legal_moves[i] = false;
    }
  }
  // If there are no legal moves
  // the game is over and
  // the result is determined by if there are any lines
  return is_lines;
}

void Game::writeGameState(float game_state[kGameStateSize]) const noexcept {
  for (int32_t i = 0; i < 4 * kBoardSize; ++i) {
    if (board_[i]) {
      game_state[i] = 1.0;
    } else {
      game_state[i] = 0.0;
    }
  }
  // Canonize the pieces
  for (int32_t i = 0; i < 6; ++i) {
    game_state[4 * kBoardSize + i] =
        static_cast<float>(pieces_[(to_play_ * 3 + i) % 6]) * 0.25;
  }
}

void Game::doMove(int32_t move_id) noexcept {
  assert(move_id >= 0 && move_id < kNumMoves);
  // This is not a conclusive check (doesn't factor in lines)
  // But might be useful for debugging
  assert(isLegalMove(move_id));
  Move move{move_id};
  // Reset the frozen space
  for (int32_t row = 0; row < 4; ++row) {
    for (int32_t col = 0; col < 4; ++col) {
      set_frozen(Space{row, col}, false);
    }
  }
  // Place move
  if (move.move_type() == Move::MoveType::kPlace) {
    // Use a piece
    --pieces_[to_play_ * 3 + move.piece_type()];
    // Place the piece
    set_board(move.space_to(), move.piece_type());
    // Freeze the space
    set_frozen(move.space_to());
  }
  // Move move
  else {
    for (PieceType piece_type : kPieceTypes) {  // For each piece type
      // Add the piece to the new space
      set_board(move.space_to(), piece_type,
                board(move.space_from(), piece_type) ||
                    board(move.space_to(), piece_type));
      // Remove the piece from the old space
      set_board(move.space_from(), piece_type, false);
    }
    // Freeze the new space
    set_frozen(move.space_to(), true);
  }
  // Switch player
  to_play_ = 1 - to_play_;
}

std::ostream &operator<<(std::ostream &os, const Game &game) {
  // Print board
  for (int32_t row = 0; row < 4; ++row) {
    for (int32_t col = 0; col < 4; ++col) {
      if (game.board(Space{row, col}, kBase))
        os << 'B';
      else
        os << ' ';
      if (game.board(Space{row, col}, kColumn))
        os << 'C';
      else
        os << ' ';
      if (game.board(Space{row, col}, kCapital))
        os << 'A';
      else
        os << ' ';
      if (game.frozen(Space{row, col}))
        os << '#';
      else
        os << ' ';
      // Print column separator
      if (col < 3)
        os << '|';
    }
    // Print row separator
    if (row < 3)
      os << "\n-------------------\n";
  }
  os << '\n';
  // Print pieces left
  for (int32_t player = 0; player < 2; ++player) {
    os << "Player " << player + 1 << ": ";
    os << "B: " << static_cast<int32_t>(game.pieces_[player * 3 + kBase])
       << ' ';
    os << "C: " << static_cast<int32_t>(game.pieces_[player * 3 + kColumn])
       << ' ';
    os << "A: " << static_cast<int32_t>(game.pieces_[player * 3 + kCapital])
       << '\n';
  }
  os << "Player " << game.to_play_ + 1 << " to play";
  return os;
}

bool Game::board(Space space, PieceType piece_type) const noexcept {
  assert(space.notNull());
  assert(piece_type >= 0 && piece_type < 3);
  return board_[space.row * 16 + space.col * 4 + piece_type];
}

bool Game::frozen(Space space) const noexcept {
  assert(space.notNull());
  return board_[space.row * 16 + space.col * 4 + kFrozen];
}

bool Game::empty(Space space) const noexcept {
  assert(space.notNull());
  return !(board(space, kBase) || board(space, kColumn) ||
           board(space, kCapital));
}

int32_t Game::top(Space space) const noexcept {
  assert(space.notNull());
  // Since it matters the order we check the pieces in
  // We don't use a range based for loop
  for (PieceType piece_type = 2; piece_type >= 0; --piece_type) {
    if (board(space, piece_type))
      return piece_type;
  }
  // Empty space
  return -1;
}

int32_t Game::bottom(Space space) const noexcept {
  assert(space.notNull());
  // Since it matters the order we check the pieces in
  // We don't use a range based for loop
  for (PieceType piece_type = 0; piece_type < 3; ++piece_type) {
    if (board(space, piece_type))
      return piece_type;
  }
  // Empty space
  return 3;
}

void Game::set_board(Space space, PieceType piece_type, bool state) noexcept {
  assert(space.notNull());
  assert(piece_type >= 0 && piece_type < 3);
  board_[space.row * 16 + space.col * 4 + piece_type] = state;
}

void Game::set_frozen(Space space, bool state) noexcept {
  assert(space.notNull());
  board_[space.row * 16 + space.col * 4 + kFrozen] = state;
}

bool Game::canPlace(const Move &move) const noexcept {
  assert(move.move_type() == Move::MoveType::kPlace);
  // Check if player has the piece left
  if (pieces_[to_play_ * 3 + move.piece_type()] == 0)
    return false;
  // Check if the space is empty
  // This is more common than frozen spaces, so we check it first
  // An empty space cannot be frozen
  if (empty(move.space_to()))
    return true;
  // Check if the space is frozen
  if (frozen(move.space_to()))
    return false;
  // Bases can only be placed on empty spaces
  if (move.piece_type() == kBase)
    return false;
  // Place a column
  // Check for absence of a column or a capital
  if (move.piece_type() == kColumn) {
    return !(board(move.space_to(), kColumn) ||
             board(move.space_to(), kCapital));
  }
  // Place a capital
  // Check for absence of a base without a column or a capital
  return !(board(move.space_to(), kCapital) ||
           (board(move.space_to(), kBase) && !board(move.space_to(), kColumn)));
}

bool Game::canMove(const Move &move) const noexcept {
  assert(move.move_type() == Move::MoveType::kMove);
  // If either space is empty, move moves are not possible
  if (empty(move.space_from()) || empty(move.space_to()))
    return false;
  // If either space is frozen, move moves are not possible
  if (frozen(move.space_from()) || frozen(move.space_to()))
    return false;
  // The bottom of the first stack must go on the top of the second
  return bottom(move.space_from()) - top(move.space_to()) == 1;
}

bool Game::isLegalMove(int32_t move_id) const noexcept {
  assert(move_id >= 0 && move_id < kNumMoves);
  Move move{move_id};
  // Place move
  if (move.move_type() == Move::MoveType::kPlace)
    return canPlace(move);
  // Move move
  return canMove(move);
}

void Game::applyLine(int32_t line,
                     std::bitset<kNumMoves> &legal_moves) const noexcept {
  legal_moves &= line_breakers[line];
}

bool Game::applyRowColLines(std::bitset<kNumMoves> &legal_moves,
                            bool isCol) const noexcept {
  for (int32_t i = 0; i < 4; ++i) {
    int32_t top0 = top(Space{i, 0, isCol});
    int32_t top1 = top(Space{i, 1, isCol});
    int32_t top2 = top(Space{i, 2, isCol});
    int32_t top3 = top(Space{i, 3, isCol});
    if (top1 == -1 || top2 == -1)
      continue;  // Empty space in middle, no line possible
    // Check for a long line
    if (top0 == top1 && top1 == top2 && top2 == top3) {
      if (isCol) {
        applyLine(CB * 12 + i * 3 + top0, legal_moves);
      } else {
        applyLine(RB * 12 + i * 3 + top0, legal_moves);
      }
      return true;  // No need to check for other lines
    }
    // Check for a left/upper short line
    for (int32_t extend_coord : {3, 0}) {
      if (top1 == top2 && ((extend_coord == 3 && top0 == top1) ||
                           (extend_coord == 0 && top2 == top3))) {
        if (isCol && extend_coord == 0) {  // Lower/down column
          applyLine(CD * 12 + i * 3 + top1, legal_moves);
        } else if (isCol && extend_coord == 3) {  // Upper column
          applyLine(CU * 12 + i * 3 + top1, legal_moves);
        } else if (extend_coord == 0) {  // Right row
          applyLine(RR * 12 + i * 3 + top1, legal_moves);
        } else {  // Left row
          applyLine(RL * 12 + i * 3 + top1, legal_moves);
        }
        if (top1 == 2) {
          // If the line is a capital line
          // A capital must be used to extend line when moving
          // The applyLine function is liberal in this case
          // So we need to remove the illegal moves
          // We turn off moving to adjacent rows/columns
          // if the top is not a capital
          // It doesn't matter which row/column the line is in
          // When it is not the correct row/column, the move is illegal anyway
          if (!board(Space{0, extend_coord, isCol}, kCapital)) {
            legal_moves[encodeMove(Space{0, extend_coord, isCol},
                                   Space{1, extend_coord, isCol})] = false;
          }
          if (!board(Space{1, extend_coord, isCol}, kCapital)) {
            legal_moves[encodeMove(Space{1, extend_coord, isCol},
                                   Space{0, extend_coord, isCol})] = false;
            legal_moves[encodeMove(Space{1, extend_coord, isCol},
                                   Space{2, extend_coord, isCol})] = false;
          }
          if (!board(Space{2, extend_coord, isCol}, kCapital)) {
            legal_moves[encodeMove(Space{2, extend_coord, isCol},
                                   Space{1, extend_coord, isCol})] = false;
            legal_moves[encodeMove(Space{2, extend_coord, isCol},
                                   Space{3, extend_coord, isCol})] = false;
          }
          if (!board(Space{3, extend_coord, isCol}, kCapital)) {
            legal_moves[encodeMove(Space{3, extend_coord, isCol},
                                   Space{2, extend_coord, isCol})] = false;
          }
        }
        return true;  // No need to check for other lines
      }
    }
  }
  return false;  // No lines
}

bool Game::applyLongDiagLines(
    std::bitset<kNumMoves> &legal_moves) const noexcept {
  // Checking the upper left to lower right long diagonal
  // is the same as checking the upper right to lower left long diagonal
  // except we flip over the y-axis
  // and use different line numbers
  for (bool flip : {false, true}) {
    int32_t top0 = top(Space{0, flip ? 3 : 0});
    int32_t top1 = top(Space{1, flip ? 2 : 1});
    int32_t top2 = top(Space{2, flip ? 1 : 2});
    int32_t top3 = top(Space{3, flip ? 0 : 3});
    if (top1 == -1 || top2 == -1) {
      continue;  // Empty space in middle, no line possible
    }
    // Long line
    if (top0 == top1 && top1 == top2 && top2 == top3) {
      if (flip) {
        applyLine(72 + D1B * 3 + top1, legal_moves);
      } else {
        applyLine(72 + D0B * 3 + top1, legal_moves);
      }
      return true;  // No need to check for other lines
    }
    // Upper line
    if (top0 == top1 && top1 == top2) {
      if (flip) {
        applyLine(72 + D1U * 3 + top1, legal_moves);
      } else {
        applyLine(72 + D0U * 3 + top1, legal_moves);
      }
      return true;  // No need to check for other lines
    }
    // Lower/down line
    if (top1 == top2 && top2 == top3) {
      if (flip) {
        applyLine(72 + D1D * 3 + top1, legal_moves);
      } else {
        applyLine(72 + D0D * 3 + top1, legal_moves);
      }
      return true;  // No need to check for other lines
    }
  }
  return false;  // No lines
}

bool Game::applyShortDiagLines(
    std::bitset<kNumMoves> &legal_moves) const noexcept {
  // Top left short diagonal
  int32_t top1 = top(Space{1, 1});
  if (top1 != -1 && top1 == top(Space{0, 2}) && top1 == top(Space{2, 0})) {
    applyLine(72 + S0 * 3 + top1, legal_moves);
    return true;  // There can only be up to 1 short diagonal line
  }
  // Top right short diagonal
  top1 = top(Space{1, 2});
  if (top1 != -1 && top1 == top(Space{0, 1}) && top1 == top(Space{2, 3})) {
    applyLine(72 + S1 * 3 + top1, legal_moves);
    return true;  // There can only be up to 1 short diagonal line
  }

  // Bottom right short diagonal
  top1 = top(Space{2, 2});
  if (top1 != -1 && top1 == top(Space{1, 3}) && top1 == top(Space{3, 1})) {
    applyLine(72 + S2 * 3 + top1, legal_moves);
    return true;  // There can only be up to 1 short diagonal line
  }

  // Bottom left short diagonal
  top1 = top(Space{2, 1});
  if (top1 != -1 && top1 == top(Space{1, 0}) && top1 == top(Space{3, 2})) {
    applyLine(72 + S3 * 3 + top1, legal_moves);
    return true;  // There can only be up to 1 short diagonal line
  }
  return false;  // No lines
}

bool Game::applyLines(std::bitset<kNumMoves> &legal_moves) const noexcept {
  // Flag for if there are any lines
  bool is_any_lines = false;
  // Row lines
  is_any_lines |= applyRowColLines(legal_moves, false);
  // Column lines
  is_any_lines |= applyRowColLines(legal_moves, true);
  // Long diagonal lines
  is_any_lines |= applyLongDiagLines(legal_moves);
  // Short diagonal lines
  is_any_lines |= applyShortDiagLines(legal_moves);
  return is_any_lines;
}