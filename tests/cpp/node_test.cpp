#include "game.h"
#include "node.h"
#include "gtest/gtest.h"
#include <bitset>

// Test the default constructor
TEST(NodeTest, DefaultConstructor) {
  Node node;
  std::bitset<kNumMoves> legal_moves;

  // Starting position should not be terminal
  ASSERT_FALSE(node.terminal());
  // Starting position should not have lines
  ASSERT_FALSE(node.getLegalMoves(legal_moves));
  // Starting position should have no children
  ASSERT_EQ(1, node.countNodes());
}

TEST(NodeTest, Size) {
  // Make sure the size of the node is 64 bytes
  ASSERT_EQ(64, sizeof(Node));
}