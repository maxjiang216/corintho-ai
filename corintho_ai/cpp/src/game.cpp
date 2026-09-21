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
  legal_moves = basicLegalMoves(info);
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
  for (int32_t type = 0; type < 3; ++type) {
    const uint32_t p = info.top_plane[type];
    if (p == 0)
      continue;
    // One shifted AND per direction finds every run of that length on the
    // whole board at once. The start masks stop runs wrapping off an edge.
    uint32_t run3[4];
    uint32_t run4[4];
    uint32_t any = 0;
    for (int32_t d = 0; d < 4; ++d) {
      const uint32_t stride = static_cast<uint32_t>(kLineStride[d]);
      const uint32_t three = p & (p >> stride) & (p >> (2 * stride));
      run3[d] = three & kRunStart[d][0];
      run4[d] = three & (p >> (3 * stride)) & kRunStart[d][1];
      any |= run3[d];
    }
    if (any == 0)
      continue;  // no line of this type; the common case
    for (int32_t d = 0; d < 4; ++d) {
      uint32_t w = run3[d];
      while (w != 0) {
        const int32_t start = __builtin_ctz(w);
        w &= w - 1;
        const int8_t shape = kRunToShape[d][start][0];
        if (shape >= 0) {
          out[count].shape = shape;
          out[count].type = gsl::narrow_cast<int8_t>(type);
          ++count;
        }
      }
      w = run4[d];
      while (w != 0) {
        const int32_t start = __builtin_ctz(w);
        w &= w - 1;
        const int8_t shape = kRunToShape[d][start][1];
        if (shape >= 0) {
          out[count].shape = shape;
          out[count].type = gsl::narrow_cast<int8_t>(type);
          ++count;
        }
      }
    }
  }
  return count;
}

MoveMask Game::basicLegalMoves(const SpaceInfo &info) const noexcept {
  const uint32_t all = 0xFFFFU;
  const uint32_t unfrozen = ~static_cast<uint32_t>(info.frozen) & all;

  // --- Placements. One expression per piece type, covering all 16 spaces. ---
  // A base needs an empty space. A column needs a space with no column and no
  // capital; empty spaces satisfy that too, since an empty space is never
  // frozen. A capital needs no capital, and not a lone base.
  uint32_t place[3];
  place[kBase] = info.empty;
  place[kColumn] = unfrozen & ~info.has[kColumn] & ~info.has[kCapital] & all;
  place[kCapital] = unfrozen & ~info.has[kCapital] &
                    (~static_cast<uint32_t>(info.has[kBase]) |
                     info.has[kColumn]) & all;
  for (int32_t p = 0; p < 3; ++p) {
    if (pieces_[to_play_ * 3 + p] == 0)
      place[p] = 0;
  }

  // --- Moves. canMove needs bottom(from) - top(to) == 1, and with both
  // spaces occupied each is in {0,1,2}, so only two pairings are possible:
  // a column-bottomed stack onto a base-topped space, or a capital-bottomed
  // stack onto a column-topped one. ---
  const uint32_t src_col = info.has[kColumn] &
                           ~static_cast<uint32_t>(info.has[kBase]) & unfrozen;
  const uint32_t dst_base = info.top_plane[kBase] & unfrozen;
  const uint32_t src_cap = info.has[kCapital] &
                           ~static_cast<uint32_t>(info.has[kBase]) &
                           ~static_cast<uint32_t>(info.has[kColumn]) & unfrozen;
  const uint32_t dst_col = info.top_plane[kColumn] & unfrozen;

  MoveMask legal;
  // Place IDs are 48 + piece * 16 + space, so each piece's sixteen placements
  // are contiguous and drop straight in as a shift.
  legal.lo = static_cast<uint64_t>(place[kBase]) << 48;
  legal.hi = static_cast<uint64_t>(place[kColumn]) |
             (static_cast<uint64_t>(place[kCapital]) << 16);

  // For each direction, a source is playable when the matching destination
  // sits one step away. Shifting the destination set back onto the source set
  // tests all sixteen spaces at once; the file masks stop a row wrapping.
  const uint32_t kNotFileD = 0x7777U;  // source may step right
  const uint32_t kNotFileA = 0xEEEEU;  // source may step left
  const uint32_t right = (((dst_base >> 1) & src_col) |
                        ((dst_col >> 1) & src_cap)) & kNotFileD;
  const uint32_t left = (((dst_base << 1) & src_col) |
                        ((dst_col << 1) & src_cap)) & kNotFileA;
  const uint32_t down = (((dst_base >> 4) & src_col) |
                        ((dst_col >> 4) & src_cap)) & all;
  const uint32_t up = (((dst_base << 4) & src_col) |
                       ((dst_col << 4) & src_cap)) & all;

  // Down and up land on contiguous ID ranges, so they shift in directly.
  legal.lo |= static_cast<uint64_t>(down & 0x0FFFU) << 12;
  legal.lo |= static_cast<uint64_t>((up >> 4) & 0x0FFFU) << 36;
  // Right and left do not, since each row contributes three IDs rather than
  // four, so those two walk their set bits.
  uint32_t w = right;
  while (w != 0) {
    const int32_t c = __builtin_ctz(w);
    w &= w - 1;
    legal.lo |= 1ULL << ((c >> 2) * 3 + (c & 3));
  }
  w = left;
  while (w != 0) {
    const int32_t c = __builtin_ctz(w);
    w &= w - 1;
    legal.lo |= 1ULL << (24 + (c >> 2) * 3 + (c & 3) - 1);
  }
  return legal;
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
  // Read the decoded move straight out of the table rather than constructing a
  // Move. Move's constructor lives in move.cpp, so without LTO the old code
  // reached it through a PLT call -- visible as `call _ZN4MoveC1Ei@PLT` in the
  // disassembly of this function -- for what is one array read.
  const MoveInfo &move = kMoveTable[move_id];

  // The board is four bits per space: three piece bits then a frozen bit, so
  // space s occupies bits [4s, 4s+4). Everything below is one 64-bit word.
  uint64_t b = board_.to_ullong();
  // Clear every frozen bit. (The old sixteen-iteration loop compiled to this
  // same single AND -- GCC had already reduced it -- so this costs nothing
  // extra and only removes the source-level noise.)
  b &= kUnfrozenMask;

  const int32_t to_shift = move.to * 4;
  if (move.is_place) {
    // Use a piece
    --pieces_[to_play_ * 3 + move.piece];
    b |= UINT64_C(1) << (to_shift + move.piece);
  } else {
    const int32_t from_shift = move.from * 4;
    // Move the whole stack at once. The old code looped over the three piece
    // types, reading source and destination and OR-ing them per type, which
    // the compiler unrolled into a chain of test/or/andn/cmove. The stack is
    // three adjacent bits, so lifting and depositing it is three operations.
    const uint64_t stack = (b >> from_shift) & kStackMask;
    b &= ~(kStackMask << from_shift);
    b |= stack << to_shift;
  }
  // Freeze the destination. Both branches do this, so it is hoisted out.
  b |= UINT64_C(1) << (to_shift + kFrozen);

  board_ = std::bitset<4 * kBoardSize>{b};
  // Switch player
  to_play_ = 1 - to_play_;
}

void Game::computeSpaceInfo(SpaceInfo &info) const noexcept {
  info.empty = 0;
  info.frozen = 0;
  info.top_plane[0] = 0;
  info.top_plane[1] = 0;
  info.top_plane[2] = 0;
  info.has[0] = 0;
  info.has[1] = 0;
  info.has[2] = 0;
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
    const uint16_t bit = static_cast<uint16_t>(1u << space_index);
    if (has_base)
      info.has[kBase] |= bit;
    if (has_column)
      info.has[kColumn] |= bit;
    if (has_capital)
      info.has[kCapital] |= bit;
    if (!(has_base || has_column || has_capital))
      info.empty |= static_cast<uint16_t>(1u << space_index);
    else
      info.top_plane[info.top[space_index]] |=
          static_cast<uint16_t>(1u << space_index);
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
