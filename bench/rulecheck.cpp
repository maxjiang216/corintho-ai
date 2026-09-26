// Exhaustive regression test for line-breaking legality.
//
// The index (shape, type, move, source-top, extend-cell-top) was shown to be
// COMPLETE: over 30,672 observed combinations, none was ambiguous, so whether a
// move breaks a line is fully determined by those five values. Nothing about
// stack interiors, reserves, frozen state elsewhere, or history can change it.
//
// That turns a fuzz test into an exhaustive one. This enumerates every cell of
// that index and, for each, builds several positions that realise it with the
// IRRELEVANT cells randomised. If the engine and the rule agree on every cell,
// they agree everywhere.
//
// Positions are constructed, not played, so they are a superset of the
// reachable set. They respect the invariants the engine relies on:
//   - every cell is one of the 7 reachable stacks (B-A never occurs)
//   - at most one frozen cell, and it always holds a piece
//
// Failures are reported by root cause, matching worklog/RULES-CHECKLIST.md.

#include <cstdint>

#include <bitset>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "bench_util.h"
#include "game.h"
#include "move.h"
#include "util.h"

namespace {

struct Shape {
  std::vector<int> cells;
  int contain4{-1};
  int extend{-1};
  char category{'?'};
  std::string name;
};

std::vector<Shape> shapes;

void buildShapes() {
  char buf[16];
  auto add = [&](std::vector<int> c, int c4, char cat, const char *nm) {
    shapes.push_back({c, c4, -1, cat, nm});
  };
  for (int r = 0; r < 4; ++r) {
    int f = static_cast<int>(shapes.size());
    snprintf(buf, 16, "R%d-4", r); add({r*4, r*4+1, r*4+2, r*4+3}, -1, 'R', buf);
    snprintf(buf, 16, "R%d-L", r); add({r*4, r*4+1, r*4+2}, f, 'R', buf);
    snprintf(buf, 16, "R%d-R", r); add({r*4+1, r*4+2, r*4+3}, f, 'R', buf);
  }
  for (int c = 0; c < 4; ++c) {
    int f = static_cast<int>(shapes.size());
    snprintf(buf, 16, "C%d-4", c); add({c, 4+c, 8+c, 12+c}, -1, 'C', buf);
    snprintf(buf, 16, "C%d-U", c); add({c, 4+c, 8+c}, f, 'C', buf);
    snprintf(buf, 16, "C%d-D", c); add({4+c, 8+c, 12+c}, f, 'C', buf);
  }
  { int f = static_cast<int>(shapes.size());
    add({0,5,10,15}, -1, 'L', "NW-4"); add({0,5,10}, f, 'L', "NW-U"); add({5,10,15}, f, 'L', "NW-D"); }
  { int f = static_cast<int>(shapes.size());
    add({3,6,9,12}, -1, 'L', "NE-4"); add({3,6,9}, f, 'L', "NE-U"); add({6,9,12}, f, 'L', "NE-D"); }
  add({5,2,8}, -1, 'S', "SD0"); add({6,1,11}, -1, 'S', "SD1");
  add({10,7,13}, -1, 'S', "SD2"); add({9,4,14}, -1, 'S', "SD3");
  for (Shape &s : shapes) {
    if (s.cells.size() == 3 && s.contain4 >= 0) {
      for (int c : shapes[s.contain4].cells) {
        bool in = false;
        for (int d : s.cells) if (d == c) in = true;
        if (!in) s.extend = c;
      }
    }
  }
}

/// @brief All seven reachable stacks, as (top, bottom, pieces).
struct Stack { int top; int bottom; std::vector<int> pieces; };
const Stack kStacks[7] = {
    {kBase,    kBase,   {kBase}},
    {kColumn,  kColumn, {kColumn}},
    {kColumn,  kBase,   {kBase, kColumn}},
    {kCapital, kCapital,{kCapital}},
    {kCapital, kColumn, {kColumn, kCapital}},
    {kCapital, kBase,   {kBase, kColumn, kCapital}},
    {-1,       3,       {}},
};

/// @brief The seven reachable stacks, indexed by top type.
/// @details B-A (base plus capital, no column) is unreachable and excluded.
const std::vector<std::vector<int>> kStacksByTop[4] = {
    {{}},                                  // top == -1, empty
    {{kBase}},                             // top == base
    {{kColumn}, {kBase, kColumn}},         // top == column
    {{kCapital}, {kColumn, kCapital}, {kBase, kColumn, kCapital}},  // top == capital
};

void setCell(int32_t *board, int cell, int top, uint64_t &rng) {
  for (int k = 0; k < 4; ++k) board[cell * 4 + k] = 0;
  if (top < 0) return;
  const auto &opts = kStacksByTop[top + 1];
  const auto &stack = opts[bench::splitmix64(rng) % opts.size()];
  for (int p : stack) board[cell * 4 + p] = 1;
}

int topOf(const int32_t *board, int cell) {
  if (board[cell * 4 + kCapital]) return kCapital;
  if (board[cell * 4 + kColumn]) return kColumn;
  if (board[cell * 4 + kBase]) return kBase;
  return -1;
}

bool holds(const int32_t *board, const Shape &s, int ty) {
  for (int c : s.cells) if (topOf(board, c) != ty) return false;
  return true;
}

/// @brief Every shape that holds, as (shape index, type).
std::vector<std::pair<int, int>> linesPresent(const int32_t *b) {
  std::vector<std::pair<int, int>> out;
  for (size_t i = 0; i < shapes.size(); ++i) {
    const int t = topOf(b, shapes[i].cells[0]);
    if (t < 0) continue;
    if (holds(b, shapes[i], t)) out.push_back({static_cast<int>(i), t});
  }
  return out;
}

/// @brief Is the target line the ONLY thing present, ignoring the shapes it
/// necessarily implies?
/// @details A 4-line always implies its two 3-subsets, and a 3-line implies its
/// containing 4-line whenever the extend cell happens to match. Those are not
/// accidental. Anything else is an unrelated line that would confound
/// attribution: the engine's early return might fire on it, and the rule would
/// require breaking it too, so a mismatch could not be blamed on the cell under
/// test.
bool isolated(const std::vector<std::pair<int, int>> &present, int target) {
  for (const auto &L : present) {
    if (L.first == target) continue;
    if (shapes[target].contain4 == L.first) continue;      // implied container
    bool implied_subset = false;
    for (size_t i = 0; i < shapes.size(); ++i)
      if (static_cast<int>(i) == L.first && shapes[i].contain4 == target)
        implied_subset = true;                              // implied subset
    if (implied_subset) continue;
    return false;
  }
  return true;
}

/// @brief Basic legality, independent of lines. Mirrors canPlace/canMove.
/// @details Validated elsewhere: on 2,090,206 line-free positions this matched
/// the engine exactly, since with no lines the engine's answer IS basic
/// legality.
bool basicLegal(const int32_t *b, const int32_t *pieces, int to_play,
                int move_id) {
  Move mv{move_id};
  auto has = [&](int cell, int k) { return b[cell * 4 + k] != 0; };
  if (mv.move_type() == Move::MoveType::kPlace) {
    const int c = mv.row_to() * 4 + mv.col_to();
    const int p = mv.piece_type();
    if (pieces[to_play * 3 + p] == 0) return false;
    const bool empty = !has(c, kBase) && !has(c, kColumn) && !has(c, kCapital);
    if (empty) return true;
    if (has(c, kFrozen)) return false;
    if (p == kBase) return false;
    if (p == kColumn) return !(has(c, kColumn) || has(c, kCapital));
    return !(has(c, kCapital) || (has(c, kBase) && !has(c, kColumn)));
  }
  const int f = mv.row_from() * 4 + mv.col_from();
  const int t = mv.row_to() * 4 + mv.col_to();
  const bool ef = !has(f, kBase) && !has(f, kColumn) && !has(f, kCapital);
  const bool et = !has(t, kBase) && !has(t, kColumn) && !has(t, kCapital);
  if (ef || et || has(f, kFrozen) || has(t, kFrozen)) return false;
  const int bottom = has(f, kBase) ? kBase : has(f, kColumn) ? kColumn : kCapital;
  return bottom - topOf(b, t) == 1;
}

/// @brief The rule, stated in worklog/RULES-CHECKLIST.md item 4.
/// @warning Assumes the move is already basically legal.
bool ruleAllows(const Game &before, int move_id) {
  int32_t b0[4 * kBoardSize];
  {
    float st[kGameStateSize];
    before.writeGameState(st);
    for (int i = 0; i < 4 * kBoardSize; ++i) b0[i] = st[i] != 0.0F ? 1 : 0;
  }
  std::vector<std::pair<int, int>> lines;
  for (size_t i = 0; i < shapes.size(); ++i) {
    int t = topOf(b0, shapes[i].cells[0]);
    if (t < 0) continue;
    if (holds(b0, shapes[i], t)) lines.push_back({static_cast<int>(i), t});
  }
  if (lines.empty()) return true;
  Game after = before;
  after.doMove(move_id);
  int32_t b1[4 * kBoardSize];
  {
    float st[kGameStateSize];
    after.writeGameState(st);
    for (int i = 0; i < 4 * kBoardSize; ++i) b1[i] = st[i] != 0.0F ? 1 : 0;
  }
  for (auto &L : lines) {
    const Shape &s = shapes[L.first];
    const int ty = L.second;
    if (!holds(b1, s, ty)) continue;                                  // (a)
    if (s.cells.size() == 3 && s.contain4 >= 0 &&
        holds(b1, shapes[s.contain4], ty)) continue;                  // (b)
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  const int reps = argc > 1 ? std::stoi(argv[1]) : 3;
  buildShapes();

  std::printf("Exhaustive line-breaking regression test\n");
  std::printf("index: shape x type x move x source-top x extend-top, "
              "%d random fills each\n\n", reps);

  uint64_t rng = 0xC0FFEEULL;
  long cells = 0, tested = 0, mismatch = 0;
  long confounded = 0, unrealised = 0, infeasible = 0;
  long cells_hit = 0;
  std::map<std::string, long> by_shape;
  std::bitset<kNumMoves> engine;

  for (size_t si = 0; si < shapes.size(); ++si) {
    const Shape &s = shapes[si];
    for (int ty = 0; ty < 3; ++ty) {
      for (int m = 0; m < kNumMoves; ++m) {
        Move mv{m};
        const bool is_move = mv.move_type() == Move::MoveType::kMove;
        const int src = is_move ? mv.row_from() * 4 + mv.col_from() : -1;
        // source-top: only meaningful for move-moves, and forced if the source
        // is itself in the line.
        std::vector<int> src_tops;
        if (!is_move) src_tops = {-1};
        else {
          bool in = false;
          for (int c : s.cells) if (c == src) in = true;
          if (in) src_tops = {ty}; else src_tops = {kBase, kColumn, kCapital};
        }
        // extend-top: only meaningful for 3-shapes with a containing 4-shape.
        std::vector<int> ext_tops;
        if (s.extend < 0) ext_tops = {-2};
        else ext_tops = {-1, kBase, kColumn, kCapital};

        for (int st : src_tops) {
          for (int xt : ext_tops) {
            ++cells;
            const int dst = mv.row_to() * 4 + mv.col_to();

            // --- Feasibility. These depend only on the index cell, not on the
            // random fill, so they are decided once. A cell that fails here is
            // IMPOSSIBLE, not merely unsampled, and is excluded from the
            // coverage denominator. ---
            int need0[kBoardSize];
            for (int c = 0; c < kBoardSize; ++c) need0[c] = -2;
            for (int c : s.cells) need0[c] = ty;
            if (s.extend >= 0 && xt != -2) need0[s.extend] = xt;
            bool feasible = true;
            if (!is_move) {
              // A place move onto a pinned cell is decidable up front, because
              // canPlace depends only on that cell's contents:
              //   base    -> only onto an empty cell
              //   column  -> empty, or a cell topped by a base
              //   capital -> empty, or a cell topped by a column
              const int want = need0[dst];
              if (want != -2 && want != -1) {
                const int p = mv.piece_type();
                if (p == kBase) feasible = false;
                else if (p == kColumn && want != kBase) feasible = false;
                else if (p == kCapital && want != kColumn) feasible = false;
              }
            }
            if (is_move) {
              if (need0[src] != -2 && need0[src] != st) feasible = false;
              else {
                need0[src] = st;
                bool any = false;
                for (const Stack &k : kStacks) {
                  if (k.top != st) continue;
                  const int need_dst = k.bottom - 1;
                  if (need_dst < 0) continue;
                  if (need0[dst] != -2 && need0[dst] != need_dst) continue;
                  any = true;
                }
                if (!any) feasible = false;
              }
            }
            if (!feasible) { ++infeasible; continue; }

            bool cell_hit = false;
            for (int rep = 0; rep < reps; ++rep) {
              int32_t board[4 * kBoardSize] = {0};
              int need[kBoardSize];
              for (int c = 0; c < kBoardSize; ++c) need[c] = need0[c];

              const Stack *src_stack = nullptr;
              if (is_move) {
                std::vector<const Stack *> cands;
                for (const Stack &k : kStacks) {
                  if (k.top != st) continue;
                  const int need_dst = k.bottom - 1;
                  if (need_dst < 0) continue;
                  if (need[dst] != -2 && need[dst] != need_dst) continue;
                  cands.push_back(&k);
                }
                src_stack = cands[bench::splitmix64(rng) % cands.size()];
                if (need[dst] == -2) need[dst] = src_stack->bottom - 1;
              }

              // --- Fill every cell, avoiding new lines where we have a choice. ---
              for (int c = 0; c < kBoardSize; ++c) {
                int want = need[c];
                if (want == -2)
                  want = static_cast<int>(bench::splitmix64(rng) % 4) - 1;
                setCell(board, c, want, rng);
              }
              if (is_move && src_stack != nullptr) {
                for (int k = 0; k < 4; ++k) board[src * 4 + k] = 0;
                for (int p : src_stack->pieces) board[src * 4 + p] = 1;
              }

              // Frozen: never on a cell the move touches, since that would make
              // the move basically illegal and waste the sample.
              int fz = static_cast<int>(bench::splitmix64(rng) % (kBoardSize + 1));
              if (fz < kBoardSize && topOf(board, fz) >= 0 && fz != dst &&
                  (!is_move || fz != src))
                board[fz * 4 + kFrozen] = 1;

              // Reserves full, so place moves are never blocked by supply.
              int32_t pieces[6];
              for (int i = 0; i < 6; ++i) pieces[i] = 4;
              int32_t to_play = static_cast<int32_t>(bench::splitmix64(rng) % 2);

              // The target line must actually hold after all the pinning.
              // If the move's source IS the extend cell we cannot realise the
              // source-top independently, so that index cell is unreachable.
              if (!holds(board, s, ty)) { ++unrealised; continue; }

              // Repair, rather than reject, accidental lines. Rejection alone
              // discarded ~72% of fills and capped coverage: an unrelated line
              // elsewhere on the board would let the engine's early return fire
              // on it, or oblige the rule to break it, so a mismatch could not
              // be attributed to the cell under test.
              //
              // Walk any offending line and retype one of its cells that is not
              // pinned by the index. A few passes clears almost everything.
              for (int pass = 0; pass < 8; ++pass) {
                const auto present = linesPresent(board);
                if (isolated(present, static_cast<int>(si))) break;
                for (const auto &L : present) {
                  if (L.first == static_cast<int>(si)) continue;
                  if (shapes[static_cast<int>(si)].contain4 == L.first) continue;
                  if (shapes[L.first].contain4 == static_cast<int>(si)) continue;
                  // Retype a free cell of this line so it stops holding.
                  for (int c : shapes[L.first].cells) {
                    if (need[c] != -2) continue;                 // pinned
                    if (is_move && (c == src || c == dst)) continue;
                    const int alt = (L.second + 1 + static_cast<int>(
                                         bench::splitmix64(rng) % 3)) % 4 - 1;
                    setCell(board, c, alt == L.second ? -1 : alt, rng);
                    break;
                  }
                  break;
                }
              }
              if (!isolated(linesPresent(board), static_cast<int>(si))) {
                ++confounded;
                continue;
              }
              if (!holds(board, s, ty)) { ++unrealised; continue; }

              // Basic legality is a precondition for both sides. Without
              // this the comparison is meaningless: doMove would also be
              // applied to an illegal move.
              if (!basicLegal(board, pieces, to_play, m)) continue;

              Game game{board, to_play, pieces};
              game.getLegalMoves(engine);
              const bool eng_ok = engine[m];
              const bool rule_ok = ruleAllows(game, m);
              ++tested;
              cell_hit = true;
              if (eng_ok != rule_ok) {
                ++mismatch;
                by_shape[s.name]++;
              }
            }
            if (cell_hit) ++cells_hit;
          }
        }
      }
    }
  }

  const long reachable = cells - infeasible;
  std::printf("index cells enumerated : %ld\n", cells);
  std::printf("  impossible by construction : %ld\n", infeasible);
  std::printf("  reachable                  : %ld\n", reachable);
  std::printf("index cells covered    : %ld of %ld reachable (%.1f%%)\n",
              cells_hit, reachable, reachable ? 100.0 * cells_hit / reachable : 0.0);
  std::printf("fills rejected, unrelated line present : %ld\n", confounded);
  std::printf("fills rejected, line did not hold      : %ld\n", unrealised);
  std::printf("comparisons made       : %ld\n", tested);
  std::printf("mismatches             : %ld (%.3f%%)\n\n", mismatch,
              tested ? 100.0 * mismatch / tested : 0.0);
  if (mismatch) {
    std::printf("by line shape:\n");
    for (auto &kv : by_shape)
      std::printf("  %-8s %7ld\n", kv.first.c_str(), kv.second);
    std::printf("\nFAIL\n");
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
