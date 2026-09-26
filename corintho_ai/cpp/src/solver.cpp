#include "solver.h"

#include <algorithm>
#include <new>
#include <type_traits>

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
  MoveMask legal;
  const bool lines = game.getLegalMoves(legal);
  const int32_t result = search(game, legal, lines, -1, 1);
  return aborted_ ? kUnknown : result;
}

// Negamax alpha-beta over exact results. The table's move is searched
// first; if it does not cut off, every other child is generated: a move that
// leaves the opponent no moves while a line stands wins at once, and the
// others are searched fewest opponent replies first (entry 08).
int32_t Solver::search(const Game &game, const MoveMask &legal, bool lines,
                       int32_t alpha, int32_t beta) {
  if (++nodes_ > max_nodes_) {
    aborted_ = true;
    return 0;
  }
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
  int32_t best = -2, best_move = -1;
  // The table's move alone first: when it refutes, the other children are
  // never generated (entry 08)
  if (table_move >= 0) {
    Game child = game;
    child.doMove(table_move);
    MoveMask child_legal;
    const bool child_lines = child.getLegalMoves(child_legal);
    const int32_t s =
        !child_legal.any() && child_lines
            ? 1
            : -search(child, child_legal, child_lines, -beta, -alpha);
    if (aborted_)
      return 0;
    best = s;
    best_move = table_move;
    alpha = std::max(alpha, s);
    if (alpha >= beta)
      return store(board, rest, best, alpha0, beta, best_move);
  }
  struct Child {
    Game game;
    MoveMask legal;
    int32_t move;
    int32_t replies;
    bool lines;
  };
  // Uninitialized: default-constructing 96 Games (each set to the starting
  // position) was ~14% of the solver's instructions (entry 08)
  static_assert(std::is_trivially_copyable_v<Child>);
  alignas(Child) unsigned char storage[sizeof(Child) * kNumMoves];
  Child *children = reinterpret_cast<Child *>(storage);
  int32_t n = 0;
  bool win = false;
  int32_t win_move = -1;
  forEachMove(legal, [&](int32_t m) {
    if (win || m == table_move)
      return;
    Child &c = *new (&children[n]) Child{game, {}, m, 0, false};
    c.game.doMove(m);
    c.lines = c.game.getLegalMoves(c.legal);
    c.replies = c.legal.count();
    if (c.replies == 0 && c.lines) {
      win = true;  // the opponent is stuck with a line on the board
      win_move = m;
      return;
    }
    ++n;
  });
  if (win) {
    best = 1;
    best_move = win_move;
  } else {
    // Fewest replies first; an insertion sort, stable, with no allocation
    // (std::stable_sort's merge sort was ~6%)
    int32_t order[kNumMoves];
    for (int32_t i = 0; i < n; ++i) {
      const int32_t x = i;
      int32_t j = i;
      for (; j > 0 && children[x].replies < children[order[j - 1]].replies;
           --j)
        order[j] = order[j - 1];
      order[j] = x;
    }
    for (int32_t k = 0; k < n; ++k) {
      const Child &c = children[order[k]];
      const int32_t s = -search(c.game, c.legal, c.lines, -beta, -alpha);
      if (aborted_)
        return 0;
      if (s > best) {
        best = s;
        best_move = c.move;
      }
      alpha = std::max(alpha, s);
      if (alpha >= beta)
        break;
    }
  }
  return store(board, rest, best, alpha0, beta, best_move);
}

int32_t Solver::store(uint64_t board, uint64_t rest, int32_t best,
                      int32_t alpha0, int32_t beta, int32_t best_move) {
  // The search may have overwritten this position's slot; store afresh
  table_[mix(board, rest) & mask_] =
      Entry{board,
            rest,
            epoch_,
            static_cast<int8_t>(best),
            static_cast<int8_t>(best <= alpha0 ? kUpper
                                : best >= beta ? kLower
                                               : kExact),
            static_cast<int8_t>(best_move)};
  return best;
}
