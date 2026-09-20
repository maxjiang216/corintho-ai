// Reference implementation of legal move generation.
//
// A verbatim copy of Game::getLegalMoves as it stood before the bitboard work
// (commit 75e7d4d), kept compiled so that every optimization can be checked
// against it position by position. See bench/verify.cpp.
//
// IMPORTANT: this file deliberately duplicates every helper it needs --
// refTop, refBottom, refEmpty, refFrozen, refBoard -- instead of calling
// Game::top(), Game::empty() and so on. If the reference shared helpers with
// the implementation under test, a bug introduced in a shared helper would
// corrupt both sides identically and the comparison would pass. The
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

// --- Private copies of the board accessors, as they were. ---

bool refBoard(const std::bitset<4 * kBoardSize> &board, Space space,
              PieceType piece_type) noexcept {
  return board[space.row * 16 + space.col * 4 + piece_type];
}

bool refFrozen(const std::bitset<4 * kBoardSize> &board, Space space) noexcept {
  return board[space.row * 16 + space.col * 4 + kFrozen];
}

bool refEmpty(const std::bitset<4 * kBoardSize> &board, Space space) noexcept {
  return !(refBoard(board, space, kBase) || refBoard(board, space, kColumn) ||
           refBoard(board, space, kCapital));
}

int32_t refTop(const std::bitset<4 * kBoardSize> &board, Space space) noexcept {
  for (PieceType piece_type = 2; piece_type >= 0; --piece_type) {
    if (refBoard(board, space, piece_type))
      return piece_type;
  }
  return -1;
}

int32_t refBottom(const std::bitset<4 * kBoardSize> &board,
                  Space space) noexcept {
  for (PieceType piece_type = 0; piece_type < 3; ++piece_type) {
    if (refBoard(board, space, piece_type))
      return piece_type;
  }
  return 3;
}

// --- Private copies of the legality rules, as they were. ---

bool refCanPlace(const std::bitset<4 * kBoardSize> &board,
                 const int8_t pieces[6], int8_t to_play,
                 const Move &move) noexcept {
  if (pieces[to_play * 3 + move.piece_type()] == 0)
    return false;
  if (refEmpty(board, move.space_to()))
    return true;
  if (refFrozen(board, move.space_to()))
    return false;
  if (move.piece_type() == kBase)
    return false;
  if (move.piece_type() == kColumn) {
    return !(refBoard(board, move.space_to(), kColumn) ||
             refBoard(board, move.space_to(), kCapital));
  }
  return !(refBoard(board, move.space_to(), kCapital) ||
           (refBoard(board, move.space_to(), kBase) &&
            !refBoard(board, move.space_to(), kColumn)));
}

bool refCanMove(const std::bitset<4 * kBoardSize> &board,
                const Move &move) noexcept {
  if (refEmpty(board, move.space_from()) || refEmpty(board, move.space_to()))
    return false;
  if (refFrozen(board, move.space_from()) || refFrozen(board, move.space_to()))
    return false;
  return refBottom(board, move.space_from()) - refTop(board, move.space_to()) ==
         1;
}

bool refIsLegalMove(const std::bitset<4 * kBoardSize> &board,
                    const int8_t pieces[6], int8_t to_play,
                    int32_t move_id) noexcept {
  Move move{move_id};
  if (move.move_type() == Move::MoveType::kPlace)
    return refCanPlace(board, pieces, to_play, move);
  return refCanMove(board, move);
}

// --- Private copies of the line detectors, as they were. ---

void refApplyLine(int32_t line, std::bitset<kNumMoves> &legal_moves) noexcept {
  legal_moves &= line_breakers[line];
}

bool refApplyRowColLines(const std::bitset<4 * kBoardSize> &board,
                         std::bitset<kNumMoves> &legal_moves,
                         bool isCol) noexcept {
  for (int32_t i = 0; i < 4; ++i) {
    int32_t top0 = refTop(board, Space{i, 0, isCol});
    int32_t top1 = refTop(board, Space{i, 1, isCol});
    int32_t top2 = refTop(board, Space{i, 2, isCol});
    int32_t top3 = refTop(board, Space{i, 3, isCol});
    if (top1 == -1 || top2 == -1)
      continue;
    if (top0 == top1 && top1 == top2 && top2 == top3) {
      if (isCol) {
        refApplyLine(CB * 12 + i * 3 + top0, legal_moves);
      } else {
        refApplyLine(RB * 12 + i * 3 + top0, legal_moves);
      }
      return true;
    }
    for (int32_t extend_coord : {3, 0}) {
      if (top1 == top2 && ((extend_coord == 3 && top0 == top1) ||
                           (extend_coord == 0 && top2 == top3))) {
        if (isCol && extend_coord == 0) {
          refApplyLine(CD * 12 + i * 3 + top1, legal_moves);
        } else if (isCol && extend_coord == 3) {
          refApplyLine(CU * 12 + i * 3 + top1, legal_moves);
        } else if (extend_coord == 0) {
          refApplyLine(RR * 12 + i * 3 + top1, legal_moves);
        } else {
          refApplyLine(RL * 12 + i * 3 + top1, legal_moves);
        }
        if (top1 == 2) {
          if (!refBoard(board, Space{0, extend_coord, isCol}, kCapital)) {
            legal_moves[encodeMove(Space{0, extend_coord, isCol},
                                   Space{1, extend_coord, isCol})] = false;
          }
          if (!refBoard(board, Space{1, extend_coord, isCol}, kCapital)) {
            legal_moves[encodeMove(Space{1, extend_coord, isCol},
                                   Space{0, extend_coord, isCol})] = false;
            legal_moves[encodeMove(Space{1, extend_coord, isCol},
                                   Space{2, extend_coord, isCol})] = false;
          }
          if (!refBoard(board, Space{2, extend_coord, isCol}, kCapital)) {
            legal_moves[encodeMove(Space{2, extend_coord, isCol},
                                   Space{1, extend_coord, isCol})] = false;
            legal_moves[encodeMove(Space{2, extend_coord, isCol},
                                   Space{3, extend_coord, isCol})] = false;
          }
          if (!refBoard(board, Space{3, extend_coord, isCol}, kCapital)) {
            legal_moves[encodeMove(Space{3, extend_coord, isCol},
                                   Space{2, extend_coord, isCol})] = false;
          }
        }
        return true;
      }
    }
  }
  return false;
}

bool refApplyLongDiagLines(const std::bitset<4 * kBoardSize> &board,
                           std::bitset<kNumMoves> &legal_moves) noexcept {
  for (bool flip : {false, true}) {
    int32_t top0 = refTop(board, Space{0, flip ? 3 : 0});
    int32_t top1 = refTop(board, Space{1, flip ? 2 : 1});
    int32_t top2 = refTop(board, Space{2, flip ? 1 : 2});
    int32_t top3 = refTop(board, Space{3, flip ? 0 : 3});
    if (top1 == -1 || top2 == -1) {
      continue;
    }
    if (top0 == top1 && top1 == top2 && top2 == top3) {
      if (flip) {
        refApplyLine(72 + D1B * 3 + top1, legal_moves);
      } else {
        refApplyLine(72 + D0B * 3 + top1, legal_moves);
      }
      return true;
    }
    if (top0 == top1 && top1 == top2) {
      if (flip) {
        refApplyLine(72 + D1U * 3 + top1, legal_moves);
      } else {
        refApplyLine(72 + D0U * 3 + top1, legal_moves);
      }
      return true;
    }
    if (top1 == top2 && top2 == top3) {
      if (flip) {
        refApplyLine(72 + D1D * 3 + top1, legal_moves);
      } else {
        refApplyLine(72 + D0D * 3 + top1, legal_moves);
      }
      return true;
    }
  }
  return false;
}

bool refApplyShortDiagLines(const std::bitset<4 * kBoardSize> &board,
                            std::bitset<kNumMoves> &legal_moves) noexcept {
  int32_t top1 = refTop(board, Space{1, 1});
  if (top1 != -1 && top1 == refTop(board, Space{0, 2}) &&
      top1 == refTop(board, Space{2, 0})) {
    refApplyLine(72 + S0 * 3 + top1, legal_moves);
    return true;
  }
  top1 = refTop(board, Space{1, 2});
  if (top1 != -1 && top1 == refTop(board, Space{0, 1}) &&
      top1 == refTop(board, Space{2, 3})) {
    refApplyLine(72 + S1 * 3 + top1, legal_moves);
    return true;
  }
  top1 = refTop(board, Space{2, 2});
  if (top1 != -1 && top1 == refTop(board, Space{1, 3}) &&
      top1 == refTop(board, Space{3, 1})) {
    refApplyLine(72 + S2 * 3 + top1, legal_moves);
    return true;
  }
  top1 = refTop(board, Space{2, 1});
  if (top1 != -1 && top1 == refTop(board, Space{1, 0}) &&
      top1 == refTop(board, Space{3, 2})) {
    refApplyLine(72 + S3 * 3 + top1, legal_moves);
    return true;
  }
  return false;
}

bool refApplyLines(const std::bitset<4 * kBoardSize> &board,
                   std::bitset<kNumMoves> &legal_moves) noexcept {
  bool is_any_lines = false;
  is_any_lines |= refApplyRowColLines(board, legal_moves, false);
  is_any_lines |= refApplyRowColLines(board, legal_moves, true);
  is_any_lines |= refApplyLongDiagLines(board, legal_moves);
  is_any_lines |= refApplyShortDiagLines(board, legal_moves);
  return is_any_lines;
}

}  // namespace

bool Game::getLegalMovesReference(
    std::bitset<kNumMoves> &legal_moves) const noexcept {
  legal_moves.set();
  bool is_lines = refApplyLines(board_, legal_moves);
  for (int32_t i = 0; i < kNumMoves; ++i) {
    if (legal_moves[i] && !refIsLegalMove(board_, pieces_, to_play_, i)) {
      legal_moves[i] = false;
    }
  }
  return is_lines;
}
