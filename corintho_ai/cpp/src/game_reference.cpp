// Reference implementation of legal move generation.
//
// Kept compiled so that every optimization can be checked against it position
// by position. See bench/verify.cpp.
//
// Re-frozen after the line-breaking fix: it now mirrors the corrected rule
// rather than the pre-fix behaviour it was first cut from. The earlier version
// encoded the defects catalogued in worklog/RULES-CHECKLIST.md item 4, so
// keeping it would have meant every later change being verified against a
// known-wrong oracle.
//
// IMPORTANT: this file deliberately duplicates everything it needs -- the
// space scan, the placement and movement rules, the line shapes, and the
// breaking rule -- instead of calling Game's own helpers. If the reference
// shared code with the implementation under test, a bug introduced in shared
// code would corrupt both sides identically and the comparison would pass. The
// duplication is the point.
//
// This file is built only by bench/Makefile. It is not listed in
// corintho_ai/python/setup.py, so it does not ship in the training module.
//
// Delete it once the bitboard stages are complete and verified.

#include "game.h"

#include <cstdint>

#include <bitset>

#include "move.h"
#include "util.h"

namespace {

// --- Private copy of the line shapes. ---

struct RefShape {
  int8_t cells[4];
  int8_t count;
  int8_t extend;
};

constexpr RefShape kRefShapes[34] = {
    {{0, 1, 2, 3}, 4, -1},     {{0, 1, 2}, 3, 3},     {{1, 2, 3}, 3, 0},
    {{4, 5, 6, 7}, 4, -1},     {{4, 5, 6}, 3, 7},     {{5, 6, 7}, 3, 4},
    {{8, 9, 10, 11}, 4, -1},   {{8, 9, 10}, 3, 11},   {{9, 10, 11}, 3, 8},
    {{12, 13, 14, 15}, 4, -1}, {{12, 13, 14}, 3, 15}, {{13, 14, 15}, 3, 12},
    {{0, 4, 8, 12}, 4, -1},    {{0, 4, 8}, 3, 12},    {{4, 8, 12}, 3, 0},
    {{1, 5, 9, 13}, 4, -1},    {{1, 5, 9}, 3, 13},    {{5, 9, 13}, 3, 1},
    {{2, 6, 10, 14}, 4, -1},   {{2, 6, 10}, 3, 14},   {{6, 10, 14}, 3, 2},
    {{3, 7, 11, 15}, 4, -1},   {{3, 7, 11}, 3, 15},   {{7, 11, 15}, 3, 3},
    {{0, 5, 10, 15}, 4, -1},   {{0, 5, 10}, 3, 15},   {{5, 10, 15}, 3, 0},
    {{3, 6, 9, 12}, 4, -1},    {{3, 6, 9}, 3, 12},    {{6, 9, 12}, 3, 3},
    {{5, 2, 8}, 3, -1},        {{6, 1, 11}, 3, -1},
    {{10, 7, 13}, 3, -1},      {{9, 4, 14}, 3, -1},
};

// --- Private copies of the board accessors. ---

bool refBoard(const std::bitset<4 * kBoardSize> &board, int32_t cell,
              PieceType piece_type) noexcept {
  return board[cell * 4 + piece_type];
}

bool refFrozen(const std::bitset<4 * kBoardSize> &board,
               int32_t cell) noexcept {
  return board[cell * 4 + kFrozen];
}

bool refEmpty(const std::bitset<4 * kBoardSize> &board, int32_t cell) noexcept {
  return !(refBoard(board, cell, kBase) || refBoard(board, cell, kColumn) ||
           refBoard(board, cell, kCapital));
}

int32_t refTop(const std::bitset<4 * kBoardSize> &board,
               int32_t cell) noexcept {
  for (PieceType piece_type = 2; piece_type >= 0; --piece_type) {
    if (refBoard(board, cell, piece_type))
      return piece_type;
  }
  return -1;
}

int32_t refBottom(const std::bitset<4 * kBoardSize> &board,
                  int32_t cell) noexcept {
  for (PieceType piece_type = 0; piece_type < 3; ++piece_type) {
    if (refBoard(board, cell, piece_type))
      return piece_type;
  }
  return 3;
}

// --- Private copies of the placement and movement rules. ---

bool refCanPlace(const std::bitset<4 * kBoardSize> &board,
                 const int8_t pieces[6], int8_t to_play,
                 const Move &move) noexcept {
  const int32_t to = move.space_to().row * 4 + move.space_to().col;
  if (pieces[to_play * 3 + move.piece_type()] == 0)
    return false;
  if (refEmpty(board, to))
    return true;
  if (refFrozen(board, to))
    return false;
  if (move.piece_type() == kBase)
    return false;
  if (move.piece_type() == kColumn)
    return !(refBoard(board, to, kColumn) || refBoard(board, to, kCapital));
  return !(refBoard(board, to, kCapital) ||
           (refBoard(board, to, kBase) && !refBoard(board, to, kColumn)));
}

bool refCanMove(const std::bitset<4 * kBoardSize> &board,
                const Move &move) noexcept {
  const int32_t from = move.space_from().row * 4 + move.space_from().col;
  const int32_t to = move.space_to().row * 4 + move.space_to().col;
  if (refEmpty(board, from) || refEmpty(board, to))
    return false;
  if (refFrozen(board, from) || refFrozen(board, to))
    return false;
  return refBottom(board, from) - refTop(board, to) == 1;
}

bool refIsLegalMove(const std::bitset<4 * kBoardSize> &board,
                    const int8_t pieces[6], int8_t to_play,
                    int32_t move_id) noexcept {
  Move move{move_id};
  if (move.move_type() == Move::MoveType::kPlace)
    return refCanPlace(board, pieces, to_play, move);
  return refCanMove(board, move);
}

// --- Private copy of the line-breaking rule. ---

int32_t refFindLines(const std::bitset<4 * kBoardSize> &board, int8_t *shapes,
                     int8_t *types) noexcept {
  int32_t count = 0;
  for (int32_t s = 0; s < 34; ++s) {
    const int32_t type = refTop(board, kRefShapes[s].cells[0]);
    if (type < 0)
      continue;
    bool all = true;
    for (int32_t k = 1; k < kRefShapes[s].count; ++k) {
      if (refTop(board, kRefShapes[s].cells[k]) != type) {
        all = false;
        break;
      }
    }
    if (all) {
      shapes[count] = static_cast<int8_t>(s);
      types[count] = static_cast<int8_t>(type);
      ++count;
    }
  }
  return count;
}

bool refBreaksAll(const std::bitset<4 * kBoardSize> &board,
                  const int8_t *shapes, const int8_t *types, int32_t num,
                  int32_t move_id) noexcept {
  Move move{move_id};
  int32_t changed[2];
  int32_t new_top[2];
  int32_t num_changed;
  const int32_t to = move.space_to().row * 4 + move.space_to().col;
  if (move.move_type() == Move::MoveType::kPlace) {
    changed[0] = to;
    new_top[0] = move.piece_type();
    num_changed = 1;
  } else {
    const int32_t from = move.space_from().row * 4 + move.space_from().col;
    changed[0] = from;
    new_top[0] = -1;
    changed[1] = to;
    new_top[1] = refTop(board, from);
    num_changed = 2;
  }
  for (int32_t i = 0; i < num; ++i) {
    const RefShape &shape = kRefShapes[shapes[i]];
    const int32_t type = types[i];
    bool survives = true;
    for (int32_t k = 0; k < num_changed && survives; ++k) {
      if (new_top[k] == type)
        continue;
      for (int32_t c = 0; c < shape.count; ++c) {
        if (shape.cells[c] == changed[k]) {
          survives = false;
          break;
        }
      }
    }
    if (!survives)
      continue;
    if (shape.extend >= 0) {
      int32_t extend_top = refTop(board, shape.extend);
      for (int32_t k = 0; k < num_changed; ++k) {
        if (changed[k] == shape.extend)
          extend_top = new_top[k];
      }
      if (extend_top == type)
        continue;
    }
    return false;
  }
  return true;
}

}  // namespace

bool Game::getLegalMovesReference(
    std::bitset<kNumMoves> &legal_moves) const noexcept {
  int8_t shapes[34];
  int8_t types[34];
  const int32_t num_lines = refFindLines(board_, shapes, types);
  legal_moves.set();
  for (int32_t i = 0; i < kNumMoves; ++i) {
    if (!refIsLegalMove(board_, pieces_, to_play_, i)) {
      legal_moves[i] = false;
    } else if (num_lines > 0 &&
               !refBreaksAll(board_, shapes, types, num_lines, i)) {
      legal_moves[i] = false;
    }
  }
  return num_lines > 0;
}
