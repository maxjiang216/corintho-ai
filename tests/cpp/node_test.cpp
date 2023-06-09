#include "gtest/gtest.h"
#include "node.h"
#include "game.h"
#include <bitset>

// Test the default constructor
TEST(NodeTest, DefaultConstructor) {
  Node node;
  std::bitset<NUM_MOVES> legal_moves;

  ASSERT_TRUE(node.get_legal_moves(legal_moves));
  ASSERT_EQ(0, node.is_terminal());
  ASSERT_FLOAT_EQ(0.0, node.get_probability(0));
}