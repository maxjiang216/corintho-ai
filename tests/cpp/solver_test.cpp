#include "solver.h"

#include <cstdint>

#include <bitset>
#include <map>
#include <random>
#include <utility>
#include <vector>

#include "game.h"
#include "util.h"
#include "gtest/gtest.h"

namespace {

// Plain minimax to the end of the game with memoized exact values: no
// alpha-beta, no bounds, no move ordering, so it shares none of the
// solver's logic beyond the rules
int32_t minimax(const Game &game,
                std::map<std::pair<uint64_t, uint64_t>, int32_t> &memo) {
  std::pair<uint64_t, uint64_t> key;
  game.key(key.first, key.second);
  const auto found = memo.find(key);
  if (found != memo.end())
    return found->second;
  std::bitset<kNumMoves> legal;
  const bool lines = game.getLegalMoves(legal);
  int32_t best = lines ? -1 : 0;  // no moves: lost if a line stands
  if (legal.any()) {
    best = -2;
    for (int32_t m = 0; m < kNumMoves; ++m) {
      if (!legal[m])
        continue;
      Game child = game;
      child.doMove(m);
      best = std::max(best, -minimax(child, memo));
    }
  }
  memo[key] = best;
  return best;
}

// P = 2 * reserves + occupied spaces bounds the plies left in any line
int32_t horizon(const Game &game) {
  float state[kGameStateSize];
  game.writeGameState(state);
  int32_t p = 0;
  for (int32_t s = 0; s < kBoardSize; ++s)
    p += (state[4 * s] + state[4 * s + 1] + state[4 * s + 2]) > 0.0F;
  for (int32_t i = 0; i < 6; ++i)
    p += 2 * static_cast<int32_t>(state[4 * kBoardSize + i] * 4.0F + 0.5F);
  return p;
}

// Positions from seeded random games once their horizon is at most max_p
std::vector<Game> endgames(int32_t count, int32_t max_p) {
  std::mt19937 rng{2026};
  std::vector<Game> out;
  while (static_cast<int32_t>(out.size()) < count) {
    Game game;
    for (;;) {
      std::bitset<kNumMoves> legal;
      game.getLegalMoves(legal);
      if (legal.none())
        break;
      if (horizon(game) <= max_p) {
        out.push_back(game);
        break;
      }
      std::vector<int32_t> moves;
      for (int32_t m = 0; m < kNumMoves; ++m)
        if (legal[m])
          moves.push_back(m);
      game.doMove(moves[rng() % moves.size()]);
    }
  }
  return out;
}

}  // namespace

TEST(SolverTest, MatchesMinimax) {
  Solver solver{16};
  int32_t results[3] = {};
  std::map<std::pair<uint64_t, uint64_t>, int32_t> memo;
  for (const Game &game : endgames(300, 13)) {
    solver.clear();
    const int32_t s = solver.solve(game, UINT64_MAX);
    ASSERT_EQ(s, minimax(game, memo));
    ++results[s + 1];
  }
  // All three results occur, so the test is not vacuous
  EXPECT_GT(results[0], 0);
  EXPECT_GT(results[1], 0);
  EXPECT_GT(results[2], 0);
}

TEST(SolverTest, TableReuseKeepsResults) {
  // Solving many positions without clearing the table must not change them
  Solver cold{16}, warm{16};
  for (const Game &game : endgames(200, 13)) {
    cold.clear();
    EXPECT_EQ(warm.solve(game, UINT64_MAX), cold.solve(game, UINT64_MAX));
  }
}

TEST(SolverTest, NodeCapGivesUnknown) {
  Solver solver{16};
  EXPECT_EQ(solver.solve(Game{}, 1000), Solver::kUnknown);
}
