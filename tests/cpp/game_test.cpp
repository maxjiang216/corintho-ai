#include "gtest/gtest.h"
#include "game.h"
#include "util.h"
#include <bitset>

TEST(GameTest, Constructor) {
    // Create a game and check the properties
    Game game;

    std::bitset<NUM_MOVES> legal_moves;
    ASSERT_FALSE(game.get_legal_moves(legal_moves));
}