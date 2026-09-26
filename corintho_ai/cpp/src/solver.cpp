#include "solver.h"

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <new>
#include <type_traits>
#include <vector>

#include "move.h"
#include "util.h"

namespace {

// Rank of a quiet (non-line-making) move, searched in increasing order:
// move a stack, then place a base, a column, a capital. This is the order
// the move IDs happen to have; made explicit because it matters: placing
// capitals first instead (the reverse) visits 7.5x as many positions at
// P <= 24 (entry 08).
constexpr std::array<int32_t, kNumMoves> makeQuietRank() {
  std::array<int32_t, kNumMoves> rank{};
  for (int32_t m = 0; m < kNumMoves; ++m) {
    const MoveInfo &info = kMoveTable[m];
    rank[m] = info.is_place ? 1 + info.piece : 0;  // base 0, capital 2
  }
  return rank;
}
constexpr std::array<int32_t, kNumMoves> kQuietRank = makeQuietRank();

// The rest of a Game::key (side to move in bit 0, reserve i in bits
// 4 + 4i, each 0-4) in 19 bits: side to move, then 3 bits per reserve
uint64_t packRest(uint64_t rest) {
  uint64_t out = rest & 1;
  for (int32_t i = 0; i < 6; ++i)
    out |= ((rest >> (4 + 4 * i)) & 7) << (1 + 3 * i);
  return out;
}
constexpr uint64_t kRestMask = (uint64_t{1} << 19) - 1;

// Which of a bucket's four boards equal `board`, as bits 0-3
uint32_t boardMatches(const Solver::Bucket &bucket, uint64_t board) {
#if defined(__AVX2__)
  const __m256i boards =
      _mm256_load_si256(reinterpret_cast<const __m256i *>(bucket.board));
  const __m256i eq = _mm256_cmpeq_epi64(
      boards, _mm256_set1_epi64x(static_cast<int64_t>(board)));
  return static_cast<uint32_t>(_mm256_movemask_pd(_mm256_castsi256_pd(eq)));
#else
  uint32_t m = 0;
  for (int32_t w = 0; w < 4; ++w)
    m |= static_cast<uint32_t>(bucket.board[w] == board) << w;
  return m;
#endif
}

uint64_t mix(uint64_t board, uint64_t rest) {
  uint64_t x = board * 0x9E3779B97F4A7C15ULL ^ (rest + 0x632BE59BD9B4E019ULL);
  x ^= x >> 29;
  x *= 0xBF58476D1CE4E5B9ULL;
  x ^= x >> 32;
  return x;
}

}  // namespace

Solver::Solver(int32_t log2_entries)
    : table_(size_t{1} << std::max(0, log2_entries - 2)),
      mask_(table_.size() - 1) {}

void Solver::clear() noexcept {
  if (++epoch_ == (1U << kEpochBits)) {  // wrapped: really clear, rarely
    std::fill(table_.begin(), table_.end(), Bucket{});
    epoch_ = 1;
  }
}

int32_t Solver::solve(const Game &game, uint64_t max_nodes) {
  nodes_ = 0;
  std::fill(&history_[0][0], &history_[0][0] + 2 * kNumMoves, 0U);
  max_nodes_ = max_nodes;
  aborted_ = false;
  MoveMask legal;
  const bool lines = game.getLegalMoves(legal);
#ifdef SOLVER_FULL_WINDOW
  const int32_t result = search(game, legal, lines, -1, 1);
  return aborted_ ? kUnknown : result;
#else
  // Two yes/no questions instead of one three-way one (entry 12): is it a
  // win (window 0..1), and if not, is it at least a draw (window -1..0)?
  // Each null window prunes distinctions the answer does not need; the
  // second search reuses the table the first one filled.
  if (search(game, legal, lines, 0, 1) >= 1)
    return aborted_ ? kUnknown : 1;
  if (aborted_)
    return kUnknown;
  const int32_t at_least_draw = search(game, legal, lines, -1, 0);
  if (aborted_)
    return kUnknown;
  return at_least_draw >= 0 ? 0 : -1;
#endif
}

#ifdef SOLVER_STATS
// Move-ordering statistics (entry 10): one record per move tried
struct StatRecord {
  uint8_t p, n_moves, phase, k, type, from, to, dest_occupied, height, top,
      replies, dist_frozen, cut, pad[3];
  uint32_t cost;  // positions visited searching this move
};
static_assert(sizeof(StatRecord) == 20);
thread_local std::vector<StatRecord> g_stats;

void statsRecord(uint64_t board, uint64_t rest, int32_t move, int32_t phase,
                 int32_t k, int32_t n_moves, int32_t replies, bool cut,
                 uint64_t cost) {
  const MoveInfo &info = kMoveTable[move];
  uint64_t reserves = 0;
  for (int32_t i = 0; i < 6; ++i)
    reserves += (rest >> (4 + 4 * i)) & 0xF;
  const int32_t occupied = __builtin_popcountll(
      (board | board >> 1 | board >> 2) & 0x1111111111111111ULL);
  int32_t frozen = -1;
  for (int32_t s = 0; s < kBoardSize; ++s)
    if ((board >> (4 * s + 3)) & 1)
      frozen = s;
  StatRecord r{};
  r.p = static_cast<uint8_t>(2 * reserves + static_cast<uint64_t>(occupied));
  r.n_moves = static_cast<uint8_t>(n_moves);
  r.phase = static_cast<uint8_t>(phase);
  r.k = static_cast<uint8_t>(std::min(k, 255));
  r.type = static_cast<uint8_t>(info.is_place ? 1 + info.piece : 0);
  r.from = static_cast<uint8_t>(info.is_place ? 255 : info.from);
  r.to = static_cast<uint8_t>(info.to);
  const uint64_t dest = (board >> (4 * info.to)) & 7;
  r.dest_occupied = dest != 0;
  if (!info.is_place) {
    const uint64_t src = (board >> (4 * info.from)) & 7;
    r.height = static_cast<uint8_t>(__builtin_popcountll(src));
    r.top = static_cast<uint8_t>(63 - __builtin_clzll(src));
  }
  r.replies = static_cast<uint8_t>(std::min(replies, 255));
  if (frozen >= 0) {
    const int32_t dr = std::abs(info.to / 4 - frozen / 4);
    const int32_t dc = std::abs(info.to % 4 - frozen % 4);
    r.dist_frozen = static_cast<uint8_t>(std::max(dr, dc));
  } else {
    r.dist_frozen = 255;
  }
  r.cut = cut;
  r.cost = static_cast<uint32_t>(std::min<uint64_t>(cost, UINT32_MAX));
  g_stats.push_back(r);
}
#define STATS(...) statsRecord(__VA_ARGS__)
#else
#define STATS(...) ((void)0)
#endif

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
  const Bucket &bucket = table_[mix(board, rest) & mask_];
  const uint64_t key_rest = packRest(rest);
  uint32_t matches = boardMatches(bucket, board);
  const uint64_t *found = nullptr;
  while (matches != 0) {
    const int32_t w = __builtin_ctz(matches);
    matches &= matches - 1;
    const uint64_t meta = bucket.meta[w];
    if ((meta & kRestMask) == key_rest && (meta >> 36) == epoch_)
      found = &bucket.meta[w];
  }
  const int32_t alpha0 = alpha;
  int32_t table_move = -1;
  if (found != nullptr) {
    const Entry slot{board, *found};
    const int32_t s = static_cast<int32_t>((slot.meta >> 19) & 3) - 1;
    const int32_t bound = static_cast<int32_t>((slot.meta >> 21) & 3);
    if (bound == kExact || (bound == kLower && s >= beta) ||
        (bound == kUpper && s <= alpha))
      return s;
    if (bound == kLower)
      alpha = std::max(alpha, s);
    else
      beta = std::min(beta, s);
    table_move = static_cast<int32_t>((slot.meta >> 23) & 127) - 1;
  }
  int32_t best = -2, best_move = -1;
  // The table's move alone first: when it refutes, the other children are
  // never generated (entry 08)
  if (table_move >= 0) {
    Game child = game;
    child.doMove(table_move);
    MoveMask child_legal;
    const bool child_lines = child.getLegalMoves(child_legal);
    [[maybe_unused]] const uint64_t nodes_before = nodes_;
    const int32_t s =
        !child_legal.any() && child_lines
            ? 1
            : -search(child, child_legal, child_lines, -beta, -alpha);
    if (aborted_)
      return 0;
    best = s;
    best_move = table_move;
    alpha = std::max(alpha, s);
    STATS(board, rest, table_move, 0, 0, legal.count(),
          static_cast<int32_t>(child_legal.count()), alpha >= beta,
          nodes_ - nodes_before);
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
  // Line-making children first: only they can win at once (the opponent
  // stuck with a line), and they force replies. Their legal moves are
  // generated here, the others' only if the search reaches them.
  forEachMove(legal, [&](int32_t m) {
    if (win || m == table_move)
      return;
    Child &c = *new (&children[n])
                   Child{game, {}, m, kNumMoves + 1 + kQuietRank[m], false};
    c.game.doMove(m);
    if (c.game.hasLine()) {
      c.lines = c.game.getLegalMoves(c.legal);
      c.replies = c.legal.count();
      if (c.replies == 0) {
        win = true;  // the opponent is stuck with a line on the board
        win_move = m;
        return;
      }
    }
    ++n;
  });
  if (win) {
    best = 1;
    best_move = win_move;
    STATS(board, rest, win_move, 3, table_move >= 0 ? 1 : 0, legal.count(), 0,
          true, 0);
  } else {
    // Line-making children by fewest replies, then the quiet ones by rank
    // group and, within a group, by history (most cutoffs first); an
    // insertion sort, stable, with no allocation
    const uint32_t *history = history_[rest & 1];
    auto before = [&](int32_t x, int32_t y) {
      const Child &a = children[x], &b = children[y];
      if (a.replies != b.replies)
        return a.replies < b.replies;
#ifdef SOLVER_NO_HISTORY
      return false;
#else
      return a.replies > kNumMoves && history[a.move] > history[b.move];
#endif
    };
    int32_t order[kNumMoves];
    for (int32_t i = 0; i < n; ++i) {
      const int32_t x = i;
      int32_t j = i;
      for (; j > 0 && before(x, order[j - 1]); --j)
        order[j] = order[j - 1];
      order[j] = x;
    }
    // Every child's table line, fetched ahead of its probe (entry 13:
    // prefetching the first 0 / 2 / 4 / 8 / all children took 66.1 / 66.0 /
    // 61.8 / 58.7 / 56.5 s at P <= 30)
    for (int32_t k = 0; k < n; ++k) {
      uint64_t child_board, child_rest;
      children[order[k]].game.key(child_board, child_rest);
      _mm_prefetch(reinterpret_cast<const char *>(
                       &table_[mix(child_board, child_rest) & mask_]),
                   _MM_HINT_T0);
    }
    for (int32_t k = 0; k < n; ++k) {
      Child &c = children[order[k]];
      if (c.replies > kNumMoves) {
        // hasLine said no line stands: no line search needed (entry 08)
        c.game.getLegalMovesNoLines(c.legal);
        c.lines = false;
      }
      [[maybe_unused]] const uint64_t nodes_before = nodes_;
      const int32_t s = -search(c.game, c.legal, c.lines, -beta, -alpha);
      if (aborted_)
        return 0;
      if (s > best) {
        best = s;
        best_move = c.move;
      }
      alpha = std::max(alpha, s);
      STATS(board, rest, c.move, c.replies > kNumMoves ? 2 : 1,
            k + (table_move >= 0 ? 1 : 0), legal.count(),
            c.replies > kNumMoves ? 255 : c.replies, alpha >= beta,
            nodes_ - nodes_before);
      if (alpha >= beta) {
        if (c.replies > kNumMoves)
          recordCutoff(board, rest, c.move);
        break;
      }
    }
  }
  return store(board, rest, best, alpha0, beta, best_move);
}

void Solver::recordCutoff(uint64_t board, uint64_t rest, int32_t move) {
  // Weight by the horizon P = 2 x reserves + occupied spaces squared, so
  // cutoffs high in the tree count most
  uint64_t reserves = 0;
  for (int32_t i = 0; i < 6; ++i)
    reserves += (rest >> (4 + 4 * i)) & 0xF;
  const uint64_t occupied = static_cast<uint64_t>(__builtin_popcountll(
      (board | board >> 1 | board >> 2) & 0x1111111111111111ULL));
  const uint64_t p = 2 * reserves + occupied;
  uint32_t &h = history_[rest & 1][move];
  h = static_cast<uint32_t>(std::min<uint64_t>(h + p * p, UINT32_MAX / 2));
}

int32_t Solver::store(uint64_t board, uint64_t rest, int32_t best,
                      int32_t alpha0, int32_t beta, int32_t best_move) {
  // The search may have overwritten this position's slot; store afresh
  const int32_t bound = best <= alpha0 ? kUpper
                        : best >= beta ? kLower
                                       : kExact;
  uint64_t reserves = 0;
  for (int32_t i = 0; i < 6; ++i)
    reserves += (rest >> (4 + 4 * i)) & 0xF;
  const uint64_t p = 2 * reserves + static_cast<uint64_t>(__builtin_popcountll(
                                        (board | board >> 1 | board >> 2) &
                                        0x1111111111111111ULL));
  Bucket &bucket = table_[mix(board, rest) & mask_];
  const uint64_t key_rest = packRest(rest);
  int32_t slot = -1;
  uint32_t matches = boardMatches(bucket, board);
  while (matches != 0 && slot < 0) {
    const int32_t w = __builtin_ctz(matches);
    matches &= matches - 1;
    if ((bucket.meta[w] & kRestMask) == key_rest)
      slot = w;  // the same position (from this epoch or an old one)
  }
  if (slot < 0) {
    // Else a stale or empty way, else the one nearest the end of the game;
    // stale ways rank below every current one
    uint64_t lowest = UINT64_MAX;
    for (int32_t w = 0; w < 4; ++w) {
      const uint64_t meta = bucket.meta[w];
      const uint64_t rank =
          (meta >> 36) != epoch_ ? 0 : 1 + ((meta >> 30) & 63);
      if (rank < lowest) {
        lowest = rank;
        slot = w;
      }
    }
  }
  bucket.board[slot] = board;
  bucket.meta[slot] = key_rest | static_cast<uint64_t>(best + 1) << 19 |
                      static_cast<uint64_t>(bound) << 21 |
                      static_cast<uint64_t>(best_move + 1) << 23 |
                      (p & 63) << 30 | static_cast<uint64_t>(epoch_) << 36;
  return best;
}

#ifdef SOLVER_STATS
void solverStatsFlush(std::FILE *out) {
  std::fwrite(g_stats.data(), sizeof(StatRecord), g_stats.size(), out);
  g_stats.clear();
}
#endif
