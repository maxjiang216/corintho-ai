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
    board_[i] = board[i] != 0;
  }
  for (int32_t i = 0; i < 6; ++i) {
    assert(pieces[i] >= 0 && pieces[i] <= 4);
    pieces_[i] = gsl::narrow_cast<int8_t>(pieces[i]);
  }
}

bool Game::getLegalMoves(MoveMask &legal_moves) const noexcept {
  // Compute top/bottom/empty/frozen for every space once. The board cannot
  // change during this call.
  SpaceInfo info;
  computeSpaceInfo(info);
  // Find the lines on the board. 85% of positions have none, and those skip
  // the breaking check entirely.
  PresentLine lines[kNumLineShapes];
  const int32_t num_lines = findLines(info, lines);
  // First set all moves to legal
  legal_moves.setAll();
  for (int32_t i = 0; i < kNumMoves; ++i) {
    if (!isLegalMove(i, info))
      legal_moves.reset(i);
  }
  // One AND per line, rather than testing every move against every line.
  if (num_lines > 0)
    legal_moves &= lineBreakers(info, lines, num_lines);
  // If there are no legal moves
  // the game is over and
  // the result is determined by if there are any lines
  return num_lines > 0;
}

bool Game::getLegalMoves(std::bitset<kNumMoves> &legal_moves) const noexcept {
  MoveMask mask;
  const bool is_lines = getLegalMoves(mask);
  legal_moves.reset();
  forEachMove(mask, [&legal_moves](int32_t id) { legal_moves[id] = true; });
  return is_lines;
}

int32_t Game::findLines(const SpaceInfo &info,
                        PresentLine *out) const noexcept {
  int32_t count = 0;
  for (int32_t s = 0; s < kNumLineShapes; ++s) {
    const LineShape &shape = kLineShapes[s];
    const int32_t type = info.top[shape.cells[0]];
    if (type < 0)
      continue;  // an empty space cannot be part of a line
    bool all = true;
    for (int32_t k = 1; k < shape.count; ++k) {
      if (info.top[shape.cells[k]] != type) {
        all = false;
        break;
      }
    }
    if (all) {
      out[count].shape = gsl::narrow_cast<int8_t>(s);
      out[count].type = gsl::narrow_cast<int8_t>(type);
      ++count;
    }
  }
  return count;
}

MoveMask Game::lineBreakers(const SpaceInfo &info, const PresentLine *lines,
                            int32_t num_lines) const noexcept {
  MoveMask breakers;
  breakers.setAll();
  for (int32_t i = 0; i < num_lines; ++i) {
    const int32_t shape = lines[i].shape;
    const int32_t type = lines[i].type;
    MoveMask mask = kLineBreakTable[shape][type];
    // The one board-dependent case: a stack moved onto the extending space
    // completes a four only if its own top matches the line's type. At most
    // four moves land on any given space, so this walks them directly rather
    // than building a set of every move starting from a space of each type.
    forEachMove(kLineExtendMoves[shape], [&](int32_t id) {
      if (info.top[kMoveTable[id].from] == type)
        mask.set(id);
    });
    breakers &= mask;
  }
  return breakers;
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
  // This is not a conclusive check (doesn't factor in lines) but has some use
  // for debugging
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

void Game::computeSpaceInfo(SpaceInfo &info) const noexcept {
  info.empty = 0;
  info.frozen = 0;
  for (int32_t space_index = 0; space_index < kBoardSize; ++space_index) {
    const int32_t base = space_index * 4;
    const bool has_base = board_[base + kBase];
    const bool has_column = board_[base + kColumn];
    const bool has_capital = board_[base + kCapital];
    // Top is the highest piece present, bottom the lowest. The sentinels
    // match the originals: -1 for an empty top, 3 for an empty bottom.
    info.top[space_index] = has_capital  ? kCapital
                            : has_column ? kColumn
                            : has_base   ? kBase
                                         : -1;
    info.bottom[space_index] = has_base     ? kBase
                               : has_column ? kColumn
                               : has_capital ? kCapital
                                             : 3;
    if (!(has_base || has_column || has_capital))
      info.empty |= static_cast<uint16_t>(1u << space_index);
    if (board_[base + kFrozen])
      info.frozen |= static_cast<uint16_t>(1u << space_index);
  }
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

bool Game::canPlace(const MoveInfo &move, const SpaceInfo &info) const noexcept {
  assert(move.is_place);
  const int32_t to = move.to;
  // Check if player has the piece left
  if (pieces_[to_play_ * 3 + move.piece] == 0)
    return false;
  // Check if the space is empty
  // This is more common than frozen spaces, so we check it first
  // An empty space cannot be frozen
  if ((info.empty >> to) & 1u)
    return true;
  // Check if the space is frozen
  if ((info.frozen >> to) & 1u)
    return false;
  // Bases can only be placed on empty spaces
  if (move.piece == kBase)
    return false;
  // Place a column
  // Check for absence of a column or a capital
  if (move.piece == kColumn)
    return !(board_[to * 4 + kColumn] || board_[to * 4 + kCapital]);
  // Place a capital
  // Check for absence of a base without a column or a capital
  return !(board_[to * 4 + kCapital] ||
           (board_[to * 4 + kBase] && !board_[to * 4 + kColumn]));
}

bool Game::canMove(const MoveInfo &move, const SpaceInfo &info) const noexcept {
  assert(!move.is_place);
  const int32_t from = move.from;
  const int32_t to = move.to;
  // If either space is empty, move moves are not possible
  if (((info.empty >> from) | (info.empty >> to)) & 1u)
    return false;
  // If either space is frozen, move moves are not possible
  if (((info.frozen >> from) | (info.frozen >> to)) & 1u)
    return false;
  // The bottom of the first stack must go on the top of the second
  return info.bottom[from] - info.top[to] == 1;
}

bool Game::isLegalMove(int32_t move_id, const SpaceInfo &info) const noexcept {
  assert(move_id >= 0 && move_id < kNumMoves);
  // Read the decoded move straight from the table; constructing a Move here
  // was 10.7% of all instructions once it stopped being inlined.
  const MoveInfo &move = kMoveTable[move_id];
  // Place move
  if (move.is_place)
    return canPlace(move, info);
  // Move move
  return canMove(move, info);
}

bool Game::isLegalMove(int32_t move_id) const noexcept {
  SpaceInfo info;
  computeSpaceInfo(info);
  return isLegalMove(move_id, info);
}
