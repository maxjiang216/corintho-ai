// Alpha-beta search with a small value network (worklog
// 2026-09-25-nn-architectures, entries 22-23), shared by ab_match and
// ab_selfplay (entry 25).
#ifndef PIPELINE_ALPHABETA_H
#define PIPELINE_ALPHABETA_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "game.h"
#include "move.h"
#include "small_net.h"
#include "util.h"

#ifdef AB_PROFILE
#include <x86intrin.h>
// Cycles by part of an alpha-beta node (worklog entry 22): move generation,
// network evaluation, move ordering, child setup; and counts
inline thread_local uint64_t g_prof[8];
#define PROF_START(v) const uint64_t v = __rdtsc()
#define PROF_ADD(i, v) (g_prof[i] += __rdtsc() - (v))
#else
#define PROF_START(v) ((void)0)
#define PROF_ADD(i, v) ((void)0)
#endif

namespace ab {

using Clock = std::chrono::steady_clock;
inline double since(Clock::time_point t) {
  return std::chrono::duration<double>(Clock::now() - t).count();
}

/// The small network's search score on a position: int8 (AB_FLOAT=1: float)
inline float evaluate(const SmallNet &net, const Game &game) {
  static const bool use_float =
      std::getenv("AB_FLOAT") != nullptr && std::atoi(std::getenv("AB_FLOAT"));
  float x[kGameStateSize];
  game.writeGameState(x);
  uint8_t state[kGameStateSize];
  for (int32_t i = 0; i < kGameStateSize; ++i)
    state[i] = static_cast<uint8_t>(x[i] * 4.0F + 0.5F);
  return use_float ? net.scoreFloat(state) : net.scoreInt8(state);
}

/// Iterative-deepening negamax with a transposition table
class AlphaBeta {
 public:
  AlphaBeta(const SmallNet &net, int32_t log2_table)
      : net_{net},
        table_(size_t{1} << log2_table), mask_{(uint64_t{1} << log2_table) -
                                               1} {}

  struct Result {
    int32_t move, depth;
    float score;
    uint64_t nodes;
  };

  /// The best move by iterative deepening, within `seconds` and up to
  /// `max_depth` plies (fixed-depth search: seconds very large)
  Result choose(const Game &root, double seconds, int32_t max_depth = 64) {
    start_ = Clock::now();
    seconds_ = seconds;
    aborted_ = false;
    nodes_ = 0;
    std::memset(history_, 0, sizeof(history_));
    MoveMask legal;
    root.getLegalMoves(legal);
    std::vector<int32_t> moves;
    forEachMove(legal, [&](int32_t m) { moves.push_back(m); });
    Result best{moves[0], 0, 0.0F, 0};
    alignas(32) SmallNet::Acc root_acc[2 * SmallNet::kMaxHidden];
    uint64_t root_board, root_rest;
    root.key(root_board, root_rest);
    net_.rootAccumulators(root_board, root_rest, root_acc);
    for (int32_t depth = 1; depth <= max_depth; ++depth) {
      float alpha = -kInf, best_score = -kInf;
      int32_t best_move = moves[0], done = 0;
      for (int32_t m : moves) {
        Game child = root;
        child.doMove(m);
        const float s = -search(child, root_acc, root_board, root_rest,
                                depth - 1, -kInf, -alpha, 1);
        if (aborted_)
          break;
        ++done;
        if (s > best_score) {
          best_score = s;
          best_move = m;
        }
        alpha = std::max(alpha, s);
      }
      if (aborted_) {
        // Out of time mid-iteration: the previous best (searched first) and
        // any move that beat it at this depth are valid results
        if (done > 0 && best_move != best.move)
          best.move = best_move;
        break;
      }
      best = {best_move, depth, best_score, nodes_};
      // The best move first in the next iteration
      std::stable_partition(moves.begin(), moves.end(),
                            [&](int32_t m) { return m == best_move; });
      if (std::abs(best_score) > 1.5F)
        break;  // a forced result
    }
    best.nodes = nodes_;
    return best;
  }

  /// For sampling a move (self-play, entry 25): a fixed-depth search, then
  /// every other root move searched with a window just below the best, so
  /// moves within `margin` of the best get their exact scores and the rest
  /// -kInf. `best` is the fixed-depth result.
  std::vector<std::pair<int32_t, float>> rootScores(const Game &root,
                                                    int32_t depth,
                                                    float margin,
                                                    Result &best) {
    best = choose(root, 1e30, depth);
    std::vector<std::pair<int32_t, float>> out;
    if (best.depth == 0 || std::abs(best.score) > 1.5F) {
      out.emplace_back(best.move, best.score);  // forced: no choice
      return out;
    }
    alignas(32) SmallNet::Acc root_acc[2 * SmallNet::kMaxHidden];
    uint64_t root_board, root_rest;
    root.key(root_board, root_rest);
    net_.rootAccumulators(root_board, root_rest, root_acc);
    const float floor = best.score - margin;
    MoveMask legal;
    root.getLegalMoves(legal);
    forEachMove(legal, [&](int32_t m) {
      if (m == best.move) {
        out.emplace_back(m, best.score);
        return;
      }
      Game child = root;
      child.doMove(m);
      const float s = -search(child, root_acc, root_board, root_rest,
                              best.depth - 1, -kInf, -floor, 1);
      out.emplace_back(m, s > floor ? s : -kInf);
    });
    best.nodes = nodes_;
    return out;
  }

  static constexpr float kInf = 1e9F;

 private:
  static constexpr float kMate = 2.0F;  // beyond any network value

  struct Entry {
    uint64_t board{0}, rest{~uint64_t{0}};
    float score{0};
    int8_t depth{-1}, flag{0}, move{-1};
  };
  enum : int8_t { kExact, kLower, kUpper };

  static uint64_t boardOf(const Game &game) {
    uint64_t board, rest;
    game.key(board, rest);
    return board;
  }
  /// The value at a leaf: from the incremental accumulator (AB_INCREMENTAL,
  /// default on) or the full evaluation
  float leafValue(const Game &game, const SmallNet::Acc *from_acc,
                  uint64_t from_board, uint64_t from_rest) const {
    if (!incremental_)
      return evaluate(net_, game);
    uint64_t board, rest;
    game.key(board, rest);
    const float v =
        net_.scoreFromParent(from_acc, from_board, from_rest, board, rest);
    if (check_incremental_ && v != evaluate(net_, game)) {
      std::fprintf(stderr, "incremental %.6f != full %.6f\n", v,
                   evaluate(net_, game));
      std::abort();
    }
    return v;
  }

  /// @param from_acc, from_board, from_rest the parent's two accumulator
  /// views (SmallNet::rootAccumulators) and Game::key words: this node's own
  /// are computed only when needed (at a leaf, in one pass; or before
  /// expanding), lazily (entry 23)
  float search(const Game &game, const SmallNet::Acc *from_acc,
               uint64_t from_board, uint64_t from_rest, int32_t depth,
               float alpha, float beta, int32_t ply) {
    if ((++nodes_ & 1023) == 0 && since(start_) > seconds_)
      aborted_ = true;
    if (aborted_)
      return 0.0F;
    PROF_START(t_gen);
    MoveMask legal;
    const bool lines = game.getLegalMoves(legal);
    PROF_ADD(0, t_gen);
    if (!legal.any())  // lost if a line stands, else drawn; prefer quick wins
      return lines ? -(kMate - 0.01F * static_cast<float>(ply)) : 0.0F;
    // A standing line forces the side to move to break it (like check):
    // keep searching instead of evaluating mid-sequence (AB_LINE_EXTEND=0
    // turns it off)
    if (depth <= 0 && !(lines && line_extend_)) {
      PROF_START(t_eval);
      const float v = leafValue(game, from_acc, from_board, from_rest);
      PROF_ADD(1, t_eval);
#ifdef AB_PROFILE
      ++g_prof[4];  // leaves evaluated
#endif
      return v;
    }
    uint64_t board, rest;
    game.key(board, rest);
    Entry &e = table_[(board * 0x9E3779B97F4A7C15ULL ^ rest) & mask_];
    int32_t tt_move = -1;
    if (e.board == board && e.rest == rest) {
      tt_move = e.move;
      if (e.depth >= depth) {
        if (e.flag == kExact || (e.flag == kLower && e.score >= beta) ||
            (e.flag == kUpper && e.score <= alpha))
          return e.score;
      }
    }
    int32_t moves[kNumMoves], n = 0;
    forEachMove(legal, [&](int32_t m) { moves[n++] = m; });
    const int32_t side = static_cast<int32_t>(rest & 1);
    // Order (the solver's, entries 08 and 10): the table's move; moves
    // that make a line, fewest opponent replies first (none: an immediate
    // win); then quiet moves: move a stack, place a base, a column, a
    // capital, each group by history. AB_ORDER_DEPTH: from this depth up,
    // by the network's value of each child instead.
    uint64_t key[kNumMoves];
    const uint64_t parent_board = board;
    PROF_START(t_order);
    for (int32_t i = 0; i < n; ++i) {
      const int32_t m = moves[i];
      if (m == tt_move) {
        key[m] = uint64_t{3} << 60;
        continue;
      }
      if (depth >= order_depth_) {
        Game child = game;
        child.doMove(m);
        key[m] = static_cast<uint64_t>((1.0F - evaluate(net_, child)) * 1e6F);
        key[m] = (uint64_t{2} << 60) - key[m];
        continue;
      }
      if (solver_order_) {
        // The child's board word as Game::doMove makes it (frozen bits do
        // not matter to lines): no Game copy, no move made unless it makes
        // a line (entry 23)
        uint64_t b = parent_board;
        const MoveInfo &mi = kMoveTable[m];
        if (mi.is_place) {
          b |= uint64_t{1} << (mi.to * 4 + mi.piece);
        } else {
          const uint64_t stack = (b >> (mi.from * 4)) & kStackMask;
          b &= ~(kStackMask << (mi.from * 4));
          b |= stack << (mi.to * 4);
        }
        const bool makes_line = Game::boardHasLine(b);
        if (check_lines_) {
          Game child = game;
          child.doMove(m);
          if (child.hasLine() != makes_line) {
            std::fprintf(stderr, "line test mismatch, move %d\n", m);
            std::abort();
          }
        }
        if (makes_line) {
          Game child = game;
          child.doMove(m);
          PROF_START(t_replies);
          MoveMask replies;
          child.getLegalMoves(replies);
          const int32_t r = replies.count();
          PROF_ADD(6, t_replies);
          if (r == 0)  // the opponent is stuck with a line: a win
            return kMate - 0.01F * static_cast<float>(ply + 1);
          key[m] =
              (uint64_t{2} << 60) + (static_cast<uint64_t>(100 - r) << 40);
          continue;
        }
        const MoveInfo &info = kMoveTable[m];
        const uint64_t rank = info.is_place ? 1 + info.piece : 0;
        key[m] =
            (uint64_t{1} << 60) + ((3 - rank) << 40) +
            std::min<uint64_t>(history_[side][m], (uint64_t{1} << 40) - 1);
        continue;
      }
      key[m] = static_cast<uint64_t>(history_[side][m]);
    }
    PROF_ADD(2, t_order);
#ifdef AB_PROFILE
    ++g_prof[5];  // interior nodes expanded
#endif
    const float alpha0 = alpha;
    float best = -kInf;
    int32_t best_move = moves[0];
    // This node's accumulator, for its children
    alignas(32) SmallNet::Acc acc[2 * SmallNet::kMaxHidden];
    if (incremental_)
      net_.updateAccumulators(from_acc, from_board, from_rest, board, rest,
                              acc);
    for (int32_t i = 0; i < n; ++i) {
      // Pick the best remaining move only when it is needed: a cutoff
      // usually comes after one or two moves, so sorting all of them
      // first was mostly wasted (selection, stable as std::stable_sort)
      PROF_START(t_sort);
      int32_t pick = i;
      for (int32_t j = i + 1; j < n; ++j)
        if (key[moves[j]] > key[moves[pick]])
          pick = j;
      if (pick != i) {
        const int32_t chosen = moves[pick];
        std::copy_backward(moves + i, moves + pick, moves + pick + 1);
        moves[i] = chosen;
      }
      PROF_ADD(7, t_sort);
      PROF_START(t_child);
      Game child = game;
      child.doMove(moves[i]);
      PROF_ADD(3, t_child);
      // Principal variation search: later moves with a null window first
      float s;
      if (i == 0) {
        s = -search(child, acc, board, rest, depth - 1, -beta, -alpha,
                    ply + 1);
      } else {
        // Late-move reductions: a late quiet move is first searched
        // shallower; only one that beats alpha gets the full depth
        const bool quiet = (key[moves[i]] >> 60) == 1;
        int32_t r = 0;
        if (lmr_ && quiet && depth >= 3 && i >= 3 && !lines)
          r = i >= 8 ? 2 : 1;
        s = -search(child, acc, board, rest, depth - 1 - r, -alpha - 1e-4F,
                    -alpha, ply + 1);
        if (r > 0 && s > alpha && !aborted_)
          s = -search(child, acc, board, rest, depth - 1, -alpha - 1e-4F,
                      -alpha, ply + 1);
        if (s > alpha && s < beta && !aborted_)
          s = -search(child, acc, board, rest, depth - 1, -beta, -alpha,
                      ply + 1);
      }
      if (aborted_)
        return 0.0F;
      if (s > best) {
        best = s;
        best_move = moves[i];
      }
      if (s > alpha)
        alpha = s;
      if (alpha >= beta) {
        history_[side][moves[i]] += depth * depth;
        break;
      }
    }
    e.board = board;
    e.rest = rest;
    e.score = best;
    e.depth = static_cast<int8_t>(depth);
    e.move = static_cast<int8_t>(best_move);
    e.flag = best <= alpha0 ? kUpper : best >= beta ? kLower : kExact;
    return best;
  }

  const SmallNet &net_;
  bool incremental_{getenv("AB_INCREMENTAL") == nullptr ||
                    atoi(getenv("AB_INCREMENTAL")) != 0};
  bool check_incremental_{getenv("AB_CHECK_INCREMENTAL") != nullptr};
  bool check_lines_{getenv("AB_CHECK_LINES") != nullptr};
  bool lmr_{getenv("AB_LMR") == nullptr || atoi(getenv("AB_LMR")) != 0};
  bool line_extend_{getenv("AB_LINE_EXTEND") == nullptr ||
                    atoi(getenv("AB_LINE_EXTEND")) != 0};
  /// The solver's move ordering (AB_SOLVER_ORDER=0: table move and history)
  bool solver_order_{getenv("AB_SOLVER_ORDER") == nullptr ||
                     atoi(getenv("AB_SOLVER_ORDER")) != 0};
  /// Nodes with at least this depth order moves by the children's values
  int32_t order_depth_{
      getenv("AB_ORDER_DEPTH") ? atoi(getenv("AB_ORDER_DEPTH")) : 99};
  std::vector<Entry> table_;
  uint64_t mask_;
  int32_t history_[2][kNumMoves]{};
  Clock::time_point start_;
  double seconds_{0};
  bool aborted_{false};
  uint64_t nodes_{0};
};

}  // namespace ab

#endif  // PIPELINE_ALPHABETA_H
