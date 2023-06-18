#include "node.h"

#include <bitset>

#include "gtest/gtest.h"

#include "game.h"
#include "move.h"
#include "util.h"

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

TEST(NodeTest, Terminal) {
  // Basic tests for detecting terminal nodes
  for (int32_t row = 0; row < 4; ++row) {
    Node node1;
    EXPECT_FALSE(node1.terminal());
    Node node2{node1.game(), &node1, nullptr,
               encodePlace(Space{row, 0}, kCapital), 1};
    EXPECT_FALSE(node2.terminal());
    Node node3{node2.game(), &node2, nullptr,
               encodePlace(Space{row, 1}, kCapital), 2};
    EXPECT_FALSE(node3.terminal());
    Node node4{node3.game(), &node3, nullptr,
               encodePlace(Space{row, 2}, kCapital), 3};
    EXPECT_FALSE(node4.terminal());
    Node node5{node4.game(), &node4, nullptr,
               encodePlace(Space{row, 3}, kCapital), 4};
    EXPECT_TRUE(node5.terminal());
  }
}