#include "game.h"

#include <immintrin.h>

#include <cassert>
#include <cstdint>
#include <cstring>

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

void Game::getLegalMovesNoLines(MoveMask &legal_moves) const noexcept {
  assert(!hasLine());
  SpaceInfo info;
  computeSpaceInfo(info);
  legal_moves = basicLegalMoves(info);
}

bool Game::getLegalMoves(std::bitset<kNumMoves> &legal_moves) const noexcept {
  MoveMask mask;
  const bool is_lines = getLegalMoves(mask);
  legal_moves.reset();
  forEachMove(mask, [&legal_moves](int32_t id) { legal_moves[id] = true; });
  return is_lines;
}

int32_t Game::horizon() const noexcept {
  const uint64_t b = board_.to_ullong();
  int32_t reserves = 0;
  for (int32_t i = 0; i < 6; ++i)
    reserves += pieces_[i];
  return 2 * reserves +
         __builtin_popcountll((b | b >> 1 | b >> 2) & 0x1111111111111111ULL);
}

void Game::key(uint64_t &board, uint64_t &rest) const noexcept {
  board = board_.to_ullong();
  rest = static_cast<uint64_t>(to_play_);
  for (int32_t i = 0; i < 6; ++i)
    rest |= static_cast<uint64_t>(pieces_[i]) << (4 + 4 * i);
}

void Game::lineSpaces(uint16_t by_type[3]) const noexcept {
  SpaceInfo info;
  computeSpaceInfo(info);
  PresentLine lines[kNumLineShapes];
  const int32_t num_lines = findLines(info, lines);
  by_type[0] = by_type[1] = by_type[2] = 0;
  for (int32_t i = 0; i < num_lines; ++i) {
    const LineShape &shape = kLineShapes[lines[i].shape];
    for (int32_t c = 0; c < shape.count; ++c)
      by_type[lines[i].type] |=
          static_cast<uint16_t>(1U << static_cast<uint32_t>(shape.cells[c]));
  }
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
  place[kCapital] =
      unfrozen & ~info.has[kCapital] &
      (~static_cast<uint32_t>(info.has[kBase]) | info.has[kColumn]) & all;
  for (int32_t p = 0; p < 3; ++p) {
    if (pieces_[to_play_ * 3 + p] == 0)
      place[p] = 0;
  }

  // --- Moves. canMove needs bottom(from) - top(to) == 1, and with both
  // spaces occupied each is in {0,1,2}, so only two pairings are possible:
  // a column-bottomed stack onto a base-topped space, or a capital-bottomed
  // stack onto a column-topped one. ---
  const uint32_t src_col =
      info.has[kColumn] & ~static_cast<uint32_t>(info.has[kBase]) & unfrozen;
  const uint32_t dst_base = info.top_plane[kBase] & unfrozen;
  const uint32_t src_cap =
      info.has[kCapital] & ~static_cast<uint32_t>(info.has[kBase]) &
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
  const uint32_t right =
      (((dst_base >> 1) & src_col) | ((dst_col >> 1) & src_cap)) & kNotFileD;
  const uint32_t left =
      (((dst_base << 1) & src_col) | ((dst_col << 1) & src_cap)) & kNotFileA;
  const uint32_t down =
      (((dst_base >> 4) & src_col) | ((dst_col >> 4) & src_cap)) & all;
  const uint32_t up =
      (((dst_base << 4) & src_col) | ((dst_col << 4) & src_cap)) & all;

  // Down and up land on contiguous ID ranges, so they shift in directly.
  legal.lo |= static_cast<uint64_t>(down & 0x0FFFU) << 12;
  legal.lo |= static_cast<uint64_t>((up >> 4) & 0x0FFFU) << 36;
  // Right and left do not, since each row contributes three IDs rather than
  // four (ID row * 3 + column for right, row * 3 + column - 1 for left).
  // Compressing out the one impossible column of each row gives exactly
  // those IDs: one pext each (entry 11); otherwise the set bits are walked.
#if defined(__BMI2__)
  legal.lo |= _pext_u64(right, kNotFileD);
  legal.lo |= _pext_u64(left, kNotFileA) << 24;
#else
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
#endif
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
    // A stack's top is `type` exactly when its bit is set in top_plane[type],
    // so each candidate move is one branch-free bit test (entry 11). The
    // extending moves are all move-moves, whose IDs lie in the low word.
    const uint32_t tops = info.top_plane[type];
    uint64_t w = kLineExtendMoves[shape].lo;
    assert(kLineExtendMoves[shape].hi == 0);
    while (w != 0) {
      const int32_t id = __builtin_ctzll(w);
      w &= w - 1;
      mask.lo |= static_cast<uint64_t>((tops >> kMoveTable[id].from) & 1U)
                 << id;
    }
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

namespace {

/// @brief The four floats a single space expands to, for each nibble value
/// @details The board stores four bits per space -- base, column, capital,
/// frozen -- so one space's nibble is exactly one group of four network
/// inputs. Sixteen possible nibbles, so the whole table is 256 bytes and
/// stays resident in L1 alongside the rest of the working set. An 8 KB
/// byte-indexed table is faster under -march=native and slower without it,
/// and this engine's speed rests on a small cache footprint; see worklog
/// entry 07.
/// @note Generated, not transcribed.
struct NibbleFloats {
  float value[16][4];
};

constexpr NibbleFloats makeNibbleFloats() {
  NibbleFloats table{};
  for (int32_t nibble = 0; nibble < 16; ++nibble) {
    for (int32_t bit = 0; bit < 4; ++bit) {
      table.value[nibble][bit] = static_cast<float>((nibble >> bit) & 1);
    }
  }
  return table;
}

constexpr NibbleFloats kNibbleFloats = makeNibbleFloats();

}  // namespace

void Game::writeGameState(float game_state[kGameStateSize]) const noexcept {
  // The old loop tested one bit and stored one float, sixty-four times. The
  // branch is on board contents, so it mispredicts constantly. Expanding a
  // nibble at a time is branchless and copies sixteen bytes per step.
  const uint64_t board = board_.to_ullong();
  for (int32_t space = 0; space < kBoardSize; ++space) {
    std::memcpy(game_state + space * 4,
                kNibbleFloats.value[(board >> (space * 4)) & 0xF],
                4 * sizeof(float));
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

namespace {

/// @brief Gather bit 0 of every nibble of x into a 16-bit mask: bit 4i of x
/// becomes bit i of the result
/// @details The board keeps four bits per space, so shifting the board right
/// by a piece type and gathering gives that type's plane over all sixteen
/// spaces. Each step halves the number of groups and doubles their width:
/// pairs of bits, then nibbles, bytes and finally one 16-bit group. Plain
/// shifts rather than BMI2 pext, so it needs no particular ISA.
constexpr uint32_t gatherNibbleBits(uint64_t x) noexcept {
  x &= 0x1111111111111111ULL;
  x = (x | (x >> 3)) & 0x0303030303030303ULL;
  x = (x | (x >> 6)) & 0x000F000F000F000FULL;
  x = (x | (x >> 12)) & 0x000000FF000000FFULL;
  x = (x | (x >> 24)) & 0x000000000000FFFFULL;
  return static_cast<uint32_t>(x);
}

static_assert(gatherNibbleBits(0x1ULL) == 0x0001U);
static_assert(gatherNibbleBits(0x10ULL) == 0x0002U);
static_assert(gatherNibbleBits(0x1000000000000000ULL) == 0x8000U);
static_assert(gatherNibbleBits(0x1111111111111111ULL) == 0xFFFFU);
static_assert(gatherNibbleBits(0xEEEEEEEEEEEEEEEEULL) == 0x0000U);
static_assert(gatherNibbleBits(0x0101010101010101ULL) == 0x5555U);

}  // namespace

void Game::computeSpaceInfo(SpaceInfo &info) const noexcept {
  const uint64_t b = board_.to_ullong();
#if defined(__BMI2__)
  // pext gathers one bit of every nibble in one instruction; the portable
  // gathering below was ~6% of the solver's instructions (entry 11)
  const uint64_t nibble = 0x1111111111111111ULL;
  const uint32_t base = static_cast<uint32_t>(_pext_u64(b, nibble << kBase));
  const uint32_t column =
      static_cast<uint32_t>(_pext_u64(b, nibble << kColumn));
  const uint32_t capital =
      static_cast<uint32_t>(_pext_u64(b, nibble << kCapital));
  const uint32_t frozen =
      static_cast<uint32_t>(_pext_u64(b, nibble << kFrozen));
#else
  const uint32_t base = gatherNibbleBits(b >> kBase);
  const uint32_t column = gatherNibbleBits(b >> kColumn);
  const uint32_t capital = gatherNibbleBits(b >> kCapital);
  const uint32_t frozen = gatherNibbleBits(b >> kFrozen);
#endif
  info.has[kBase] = static_cast<uint16_t>(base);
  info.has[kColumn] = static_cast<uint16_t>(column);
  info.has[kCapital] = static_cast<uint16_t>(capital);
  info.frozen = static_cast<uint16_t>(frozen);
  info.empty = static_cast<uint16_t>(~(base | column | capital) & 0xFFFFU);
  // The top is the highest piece present: capital over column over base
  info.top_plane[kCapital] = static_cast<uint16_t>(capital);
  info.top_plane[kColumn] = static_cast<uint16_t>(column & ~capital);
  info.top_plane[kBase] = static_cast<uint16_t>(base & ~column & ~capital);
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

bool Game::canPlace(const MoveInfo &move,
                    const SpaceInfo &info) const noexcept {
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

bool Game::canMove(const MoveInfo &move,
                   const SpaceInfo &info) const noexcept {
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
  return info.bottom(from) - info.top(to) == 1;
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

// Every run of three that kRunStart admits is a line shape, so a line stands
// exactly when some top plane has a run of three in some direction
constexpr bool everyRunIsAShape() {
  for (int32_t d = 0; d < 4; ++d)
    for (int32_t c = 0; c < kBoardSize; ++c)
      if (((kRunStart[d][0] >> c) & 1) != 0 && kRunToShape[d][c][0] < 0)
        return false;
  return true;
}
static_assert(everyRunIsAShape());

bool Game::boardHasLine(uint64_t b) noexcept {
  // Branch-free: the three top planes, packed into 16-bit lanes of one word,
  // then one shifted AND per direction for all three types at once. A lane
  // shifted right spills its low bits into the lane below, but only onto
  // positions the run masks exclude: every admitted run lies within its lane
  // (entry 08). pext gathers a piece's bit from all 16 nibbles at once.
#if defined(__BMI2__)
  const uint64_t nibble = 0x1111111111111111ULL;
  const uint64_t base = _pext_u64(b, nibble << kBase);
  const uint64_t column = _pext_u64(b, nibble << kColumn);
  const uint64_t capital = _pext_u64(b, nibble << kCapital);
#else
  const uint64_t base = gatherNibbleBits(b >> kBase);
  const uint64_t column = gatherNibbleBits(b >> kColumn);
  const uint64_t capital = gatherNibbleBits(b >> kCapital);
#endif
  const uint64_t planes =
      (base & ~column & ~capital) | (column & ~capital) << 16 | capital << 32;
  constexpr auto lanes = [](uint16_t m) {
    return static_cast<uint64_t>(m) * 0x0000000100010001ULL;
  };
  uint64_t any = 0;
  for (int32_t d = 0; d < 4; ++d) {
    const uint32_t s = static_cast<uint32_t>(kLineStride[d]);
    any |=
        planes & (planes >> s) & (planes >> (2 * s)) & lanes(kRunStart[d][0]);
  }
  return any != 0;
}
