#include "solver.h"

#include <algorithm>

#include "move.h"
#include "util.h"

namespace {

uint64_t mix(uint64_t board, uint64_t rest) {
  uint64_t x = board * 0x9E3779B97F4A7C15ULL ^ (rest + 0x632BE59BD9B4E019ULL);
  x ^= x >> 29;
  x *= 0xBF58476D1CE4E5B9ULL;
  x ^= x >> 32;
  return x;
}

}  // namespace

Solver::Solver(int32_t log2_entries)
    : table_(size_t{1} << log2_entries),
      mask_((uint64_t{1} << log2_entries) - 1) {}

void Solver::clear() noexcept {
  if (++epoch_ == 0) {  // wrapped: really clear, once every 2^32 clears
    std::fill(table_.begin(), table_.end(), Entry{});
    epoch_ = 1;
  }
}

int32_t Solver::solve(const Game &game, uint64_t max_nodes) {
  nodes_ = 0;
  max_nodes_ = max_nodes;
  aborted_ = false;
  const int32_t result = search(game, -1, 1);
  return aborted_ ? kUnknown : result;
}

// Negamax alpha-beta over exact results. The table's move is tried first,
// then the rest in move-ID order (the baseline; see entry 08).
int32_t Solver::search(const Game &game, int32_t alpha, int32_t beta) {
  if (++nodes_ > max_nodes_) {
    aborted_ = true;
    return 0;
  }
  MoveMask legal;
  const bool lines = game.getLegalMoves(legal);
  if (!legal.any())
    return lines ? -1 : 0;  // no moves: lost if a line stands, else drawn
  uint64_t board, rest;
  game.key(board, rest);
  Entry &slot = table_[mix(board, rest) & mask_];
  const int32_t alpha0 = alpha;
  int32_t table_move = -1;
  if (slot.epoch == epoch_ && slot.board == board && slot.rest == rest) {
    const int32_t s = slot.score;
    if (slot.bound == kExact || (slot.bound == kLower && s >= beta) ||
        (slot.bound == kUpper && s <= alpha))
      return s;
    if (slot.bound == kLower)
      alpha = std::max(alpha, s);
    else
      beta = std::min(beta, s);
    table_move = slot.move;
  }
  int32_t moves[kNumMoves];
  int32_t n = 0;
  forEachMove(legal, [&](int32_t m) { moves[n++] = m; });
  if (table_move >= 0)
    for (int32_t i = 0; i < n; ++i)
      if (moves[i] == table_move) {
        std::rotate(moves, moves + i, moves + i + 1);
        break;
      }
  int32_t best = -2, best_move = moves[0];
  for (int32_t i = 0; i < n; ++i) {
    Game child = game;
    child.doMove(moves[i]);
    const int32_t s = -search(child, -beta, -alpha);
    if (aborted_)
      return 0;
    if (s > best) {
      best = s;
      best_move = moves[i];
    }
    alpha = std::max(alpha, s);
    if (alpha >= beta)
      break;
  }
  // The search above may have overwritten this slot; store afresh
  Entry &out = table_[mix(board, rest) & mask_];
  out = Entry{board,
              rest,
              epoch_,
              static_cast<int8_t>(best),
              static_cast<int8_t>(best <= alpha0 ? kUpper
                                  : best >= beta ? kLower
                                                 : kExact),
              static_cast<int8_t>(best_move)};
  return best;
}
