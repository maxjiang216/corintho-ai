#include "gtest/gtest.h"
#include "node.h"
#include "game.h"
#include <bitset>

// Test the default constructor
TEST(NodeTest, DefaultConstructor) {
  Node node;
  std::bitset<NUM_MOVES> legal_moves;

  // Starting position should not be terminal
  ASSERT_FALSE(node.is_terminal());
  // Starting position should not have lines
  ASSERT_FALSE(node.get_legal_moves(legal_moves));
  // Starting position should have no children
  ASSERT_EQ(1, node.count_nodes());
}