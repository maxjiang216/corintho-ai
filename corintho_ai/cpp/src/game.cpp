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

namespace {

/// @brief Bit k of each space's nibble set when the space's top piece is
/// type k (base, column, capital); bit 3 is garbage (the frozen bit)
/// @details The top is the highest piece present, so a piece is the top
/// when no piece above it is. c brings each nibble's column and capital
/// bits down one place, onto base and column; c >> 1 brings the capital
/// down onto base. Each clears what it lands on. Bits that slide across a
/// space boundary only ever land on bit 3, which every caller masks off.
/// Found by bench/superopt/synth.py, which also proved that no program of
/// four instructions (and, or, xor, andn, not, shifts, constants) computes
/// it; it replaces four pext and their combination (entry 26).
constexpr uint64_t topNibbles(uint64_t b) noexcept {
  const uint64_t c = (b & 0x6666666666666666ULL) >> 1;
  return b & ~c & ~(c >> 1);
}

/// @brief A 16-bit space mask spread to the board's layout: the three piece
/// bits of each chosen space's nibble
constexpr uint64_t spreadNibbles(uint16_t spaces) noexcept {
  uint64_t out = 0;
  for (int32_t s = 0; s < kBoardSize; ++s)
    if ((spaces >> s) & 1U)
      out |= UINT64_C(7) << (4 * s);
  return out;
}

/// @brief kRunStart in the board's layout, for runs of all types at once
constexpr uint64_t kRunStartNibbles[4][2] = {
    {spreadNibbles(kRunStart[0][0]), spreadNibbles(kRunStart[0][1])},
    {spreadNibbles(kRunStart[1][0]), spreadNibbles(kRunStart[1][1])},
    {spreadNibbles(kRunStart[2][0]), spreadNibbles(kRunStart[2][1])},
    {spreadNibbles(kRunStart[3][0]), spreadNibbles(kRunStart[3][1])},
};

constexpr uint64_t kPieceBits = 0x7777777777777777ULL;
static_assert((topNibbles(0x0ULL) & kPieceBits) == 0x0ULL);
static_assert((topNibbles(0x7ULL) & kPieceBits) == 0x4ULL);  // capital on top
static_assert((topNibbles(0x3ULL) & kPieceBits) == 0x2ULL);  // base, column
static_assert((topNibbles(0x9ULL) & kPieceBits) == 0x1ULL);  // frozen base
static_assert((topNibbles(0x50ULL) & kPieceBits) == 0x40ULL);  // space 1

/// @brief Every stack move whose source space is in `spaces`, as move-ID
/// bits of a MoveMask's low word
/// @details The four directions' IDs are contiguous blocks (see
/// basicLegalMoves): right and left drop the one column that cannot step
/// that way, down and up are the board shifted into place. Proven equal to
/// testing each move's source by bench/superopt/prove.py (entry 26).
uint64_t movesFrom(uint32_t spaces) noexcept {
#if defined(__BMI2__)
  const uint64_t right = _pext_u64(spaces, 0x7777U);
  const uint64_t left = _pext_u64(spaces, 0xEEEEU);
#else
  // Row r's three admitted columns move down by r (right) or r + 1 (left)
  const uint64_t right = (spaces & 0x7U) | ((spaces >> 1) & 0x38U) |
                         ((spaces >> 2) & 0x1C0U) | ((spaces >> 3) & 0xE00U);
  const uint64_t left = ((spaces >> 1) & 0x7U) | ((spaces >> 2) & 0x38U) |
                        ((spaces >> 3) & 0x1C0U) | ((spaces >> 4) & 0xE00U);
#endif
  return right | static_cast<uint64_t>(spaces & 0x0FFFU) << 12 | left << 24 |
         static_cast<uint64_t>(spaces >> 4) << 36;
}

}  // namespace

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
  // First set all moves to legal
  legal_moves = basicLegalMoves(info);
  // Then one AND per line standing, rather than testing every move against
  // every line. 85% of positions have none, and those skip it entirely.
  bool lines;
  const MoveMask breakers = breakersOfLines(info, lines);
  if (lines)
    legal_moves &= breakers;
  // If there are no legal moves
  // the game is over and
  // the result is determined by if there are any lines
  return lines;
}

MoveMask Game::breakersOfLines(const SpaceInfo &info,
                               bool &lines) const noexcept {
  // findLines and lineBreakers in one pass: each run found is ANDed in at
  // once, with no list of lines in between (entry 26). The same runs, in
  // the same layout, as findLines.
  const uint64_t t = topNibbles(board_.to_ullong());
  uint64_t run3[4];
  uint64_t run4[4];
  uint64_t any = 0;
  for (int32_t d = 0; d < 4; ++d) {
    const uint32_t stride = 4 * static_cast<uint32_t>(kLineStride[d]);
    const uint64_t three = t & (t >> stride) & (t >> (2 * stride));
    run3[d] = three & kRunStartNibbles[d][0];
    run4[d] = three & (t >> (3 * stride)) & kRunStartNibbles[d][1];
    any |= run3[d];
  }
  MoveMask breakers;
  breakers.setAll();
  lines = any != 0;
  if (!lines)
    return breakers;
  uint64_t from_type[3];
  uint32_t built = 0;
  for (int32_t d = 0; d < 4; ++d) {
    for (int32_t n = 0; n < 2; ++n) {
      for (uint64_t w = n == 0 ? run3[d] : run4[d]; w != 0; w &= w - 1) {
        const int32_t bit = __builtin_ctzll(w);
        const int32_t shape = kRunToShape[d][bit >> 2][n];
        if (shape < 0)
          continue;
        const int32_t type = bit & 3;
        MoveMask mask = kLineBreakTable[shape][type];
        // Four of the five ways to break a line are constant per shape and
        // type (kLineBreakTable). The fifth depends on the board: a stack
        // moved onto the extending space completes a four only if its own
        // top matches the line's type, so it breaks the three exactly then.
        // One AND with every move from a space topped by the type (entry
        // 24); the extending moves are all move-moves, in the low word.
        if (((built >> type) & 1U) == 0) {
          from_type[type] = movesFrom(info.top_plane[type]);
          built |= 1U << type;
        }
        mask.lo |= from_type[type] & kLineExtendMoves[shape].lo;
        breakers &= mask;
      }
    }
  }
  return breakers;
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

static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
              "horizon and key read the reserves as a little-endian word");

int32_t Game::horizon() const noexcept {
  const uint64_t b = board_.to_ullong();
  // The six reserves in one load; the multiply adds every byte into byte 5
  // (at most 24, so no byte carries). Proven by bench/superopt/prove.py.
  uint64_t bytes = 0;
  std::memcpy(&bytes, pieces_, sizeof(pieces_));
  const int32_t reserves =
      static_cast<int32_t>(((bytes * 0x010101010101ULL) >> 40) & 0xFF);
  return 2 * reserves +
         __builtin_popcountll((b | b >> 1 | b >> 2) & 0x1111111111111111ULL);
}

void Game::key(uint64_t &board, uint64_t &rest) const noexcept {
  board = board_.to_ullong();
  // The six reserves in one load, a byte each (little-endian), then each
  // byte's low nibble packed together: the same word the old six-step loop
  // built. Proven equal for every reserve count 0-4 by
  // bench/superopt/prove.py (entry 26).
  uint64_t bytes = 0;
  std::memcpy(&bytes, pieces_, sizeof(pieces_));
#if defined(__BMI2__)
  const uint64_t reserves = _pext_u64(bytes, 0x0F0F0F0F0F0FULL);
#else
  bytes = (bytes | bytes >> 4) & 0x00FF00FF00FFULL;
  bytes = (bytes | bytes >> 8) & 0x0000FFFF0000FFFFULL;
  const uint64_t reserves = (bytes | bytes >> 16) & 0xFFFFFFULL;
#endif
  rest = reserves << 4 | static_cast<uint64_t>(to_play_);
}

void Game::lineSpaces(uint16_t by_type[3]) const noexcept {
  SpaceInfo info;
  computeSpaceInfo(info);
  PresentLine lines[kNumLineShapes];
  const int32_t num_lines = findLines(lines);
  by_type[0] = by_type[1] = by_type[2] = 0;
  for (int32_t i = 0; i < num_lines; ++i) {
    const LineShape &shape = kLineShapes[lines[i].shape];
    for (int32_t c = 0; c < shape.count; ++c)
      by_type[lines[i].type] |=
          static_cast<uint16_t>(1U << static_cast<uint32_t>(shape.cells[c]));
  }
}

int32_t Game::findLines(PresentLine *out) const noexcept {
  // All three types at once, in the board's own layout: bit k of a space's
  // nibble is set when its top is type k, and a run of equal tops is a
  // shifted AND, four bits per space (entry 26)
  const uint64_t t = topNibbles(board_.to_ullong());
  uint64_t run3[4];
  uint64_t run4[4];
  uint64_t any = 0;
  for (int32_t d = 0; d < 4; ++d) {
    const uint32_t stride = 4 * static_cast<uint32_t>(kLineStride[d]);
    const uint64_t three = t & (t >> stride) & (t >> (2 * stride));
    run3[d] = three & kRunStartNibbles[d][0];
    run4[d] = three & (t >> (3 * stride)) & kRunStartNibbles[d][1];
    any |= run3[d];
  }
  if (any == 0)
    return 0;  // no line; the common case
  int32_t count = 0;
  for (int32_t d = 0; d < 4; ++d) {
    for (int32_t n = 0; n < 2; ++n) {
      uint64_t w = n == 0 ? run3[d] : run4[d];
      while (w != 0) {
        const int32_t bit = __builtin_ctzll(w);
        w &= w - 1;
        const int8_t shape = kRunToShape[d][bit >> 2][n];
        if (shape >= 0) {
          out[count].shape = shape;
          out[count].type = gsl::narrow_cast<int8_t>(bit & 3);
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

namespace {

/// @brief A board-word mask: `bits` in the nibble of each space in `spaces`
constexpr uint64_t spreadBits(uint16_t spaces, uint64_t bits) noexcept {
  uint64_t out = 0;
  for (int32_t s = 0; s < kBoardSize; ++s)
    if ((spaces >> s) & 1U)
      out |= bits << (4 * s);
  return out;
}

/// @brief Bit 3 of each nibble gathered into a 16-bit plane, then the
/// spaces of `keep` compressed in order (a pext of both steps at once)
template <uint16_t keep> uint64_t gatherHits(uint64_t x) noexcept {
#if defined(__BMI2__)
  constexpr uint64_t kMask = spreadBits(keep, 8);  // folded, not looped
  return _pext_u64(x, kMask);
#else
  const uint32_t plane = gatherNibbleBits(x >> 3);
  if constexpr (keep == 0xEEEE)  // rows' columns 1-3, three per row
    return ((plane >> 1) & 0x7U) | ((plane >> 2) & 0x38U) |
           ((plane >> 3) & 0x1C0U) | ((plane >> 4) & 0xE00U);
  else if constexpr (keep == 0x7777)  // rows' columns 0-2
    return (plane & 0x7U) | ((plane >> 1) & 0x38U) | ((plane >> 2) & 0x1C0U) |
           ((plane >> 3) & 0xE00U);
  else if constexpr (keep == 0xFFF0)  // rows 1-3
    return plane >> 4;
  else  // 0x0FFF, rows 0-2
    return plane & 0x0FFFU;
#endif
}

/// @brief Bit `piece` of every nibble gathered into a 16-bit plane
uint64_t gatherPiece(uint64_t x, int32_t piece) noexcept {
#if defined(__BMI2__)
  return _pext_u64(x, 0x1111111111111111ULL << piece);
#else
  return gatherNibbleBits(x >> piece);
#endif
}

}  // namespace

MoveMask Game::lineMakingMoves() const noexcept {
  assert(!hasLine());
  // With no line standing, a line after a move runs through the space the
  // move lands on, since that is the only space whose top changes (the
  // source of a stack move empties, and an empty space is in no line). So
  // for each space and type: would topping this space with this type
  // complete a run of three? The space is a run's first, middle or last
  // cell, and the other two must already carry that top. The nibble layout
  // tests every type at once, as in boardHasLine.
  const uint64_t t = topNibbles(board_.to_ullong()) & kPieceBits;
  uint64_t first[4], mid[4], last[4], full[4];
  for (int32_t d = 0; d < 4; ++d) {
    const uint32_t s = 4 * static_cast<uint32_t>(kLineStride[d]);
    const uint64_t start = kRunStartNibbles[d][0];
    const uint64_t up1 = t >> s, dn1 = t << s;
    first[d] = up1 & (t >> (2 * s)) & start;
    mid[d] = dn1 & up1 & (start << s);
    last[d] = dn1 & (t << (2 * s)) & (start << (2 * s));
    full[d] = first[d] | mid[d] | last[d];
  }
  const uint64_t diagonal = full[kDirDiagA] | full[kDirDiagB];
  const uint64_t any = full[kDirRow] | full[kDirCol] | diagonal;

  // A stack keeps its top, and its source empties: in the move's own
  // direction only the runs through the destination that avoid the source
  // count. A hit is the destination's completing types ANDed with the
  // arriving top (at most one bit per nibble); adding 7 carries it into
  // bit 3, where gatherHits reads it.
  const auto hits = [](uint64_t w, uint64_t arriving) {
    return (w & arriving) + kPieceBits;
  };
  const uint64_t right =
      hits(first[kDirRow] | full[kDirCol] | diagonal, t << 4);
  const uint64_t left = hits(last[kDirRow] | full[kDirCol] | diagonal, t >> 4);
  const uint64_t down =
      hits(full[kDirRow] | first[kDirCol] | diagonal, t << 16);
  const uint64_t up = hits(full[kDirRow] | last[kDirCol] | diagonal, t >> 16);

  // Move IDs as in basicLegalMoves, indexed here by destination: right
  // lands on columns 1-3, left on 0-2, down on rows 1-3, up on rows 0-2
  MoveMask making;
  making.lo = gatherHits<0xEEEE>(right) | gatherHits<0xFFF0>(down) << 12 |
              gatherHits<0x7777>(left) << 24 | gatherHits<0x0FFF>(up) << 36 |
              gatherPiece(any, kBase) << 48;
  making.hi = gatherPiece(any, kColumn) | gatherPiece(any, kCapital) << 16;
  return making;
}

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
  // Branch-free, and no pext: the three top planes stay interleaved in the
  // board's nibbles, so one shifted AND per direction covers every type.
  // Shifts are whole spaces, so bit k of a nibble only ever meets bit k of
  // another and the types never mix; the run masks keep runs on the board.
  // Proven equal to the 16-bit-lane version on all 2^64 words by
  // bench/superopt/prove.py (entry 26).
  const uint64_t t = topNibbles(b);
  uint64_t any = 0;
  for (int32_t d = 0; d < 4; ++d) {
    const uint32_t s = 4 * static_cast<uint32_t>(kLineStride[d]);
    any |= t & (t >> s) & (t >> (2 * s)) & kRunStartNibbles[d][0];
  }
  return any != 0;
}
