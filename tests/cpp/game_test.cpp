#include "game.h"
#include "util.h"
#include "gtest/gtest.h"
#include <bitset>

TEST(GameTest, Constructor) {
  // Create a game and check the properties
  Game game;

  std::bitset<kNumMoves> legal_moves;
  ASSERT_FALSE(game.getLegalMoves(legal_moves));
}