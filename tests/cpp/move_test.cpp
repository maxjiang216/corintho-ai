#include "move.h"

#include <iostream>

#include "gtest/gtest.h"

#include "util.h"

// Very basic test for the Move class constructor
TEST(MoveTest, Constructor) {
  // Create a move with id 0 and check the properties
  Move move0{0};
  EXPECT_EQ(move0.move_type(), Move::MoveType::kMove);
  EXPECT_EQ(move0.row_from(), 0);
  EXPECT_EQ(move0.col_from(), 0);
  EXPECT_EQ(move0.row_to(), 0);
  EXPECT_EQ(move0.col_to(), 1);

  // Create a move with id 48 and check the properties
  Move move48{48};
  EXPECT_EQ(move48.move_type(), Move::MoveType::kPlace);
  EXPECT_EQ(move48.piece_type(), kBase);
  EXPECT_EQ(move48.row_to(), 0);
  EXPECT_EQ(move48.col_to(), 0);
}

TEST(MoveTest, EncodePlace) {
  // Test encoding of a place
  EXPECT_EQ(encodePlace(Space{0, 0}, kBase), 48);
  EXPECT_EQ(encodePlace(Space{2, 3}, kColumn), 75);
  EXPECT_EQ(encodePlace(Space{3, 1}, kCapital), 93);
}

TEST(MoveTest, EncodeMove) {
  // Test encoding of a move
  EXPECT_EQ(encodeMove(Space{0, 0}, Space{0, 1}), 0);
  EXPECT_EQ(encodeMove(Space{1, 2}, Space{2, 2}), 18);
}

TEST(MoveTest, RecoverMove) {
  // Test that decoding and encoding a move returns the same move
  for (int32_t id = 0; id < 48; ++id) {
    Move move{id};
    EXPECT_EQ(encodeMove(move.space_from(), move.spaceTo()), id);
  }
  for (int32_t id = 48; id < kNumMoves; ++id) {
    Move move{id};
    EXPECT_EQ(encodePlace(move.spaceTo(), move.piece_type()), id);
  }
}

TEST(MoveTest, GetColName) {
  // Test column name conversion
  EXPECT_EQ(getColName(0), 'a');
  EXPECT_EQ(getColName(1), 'b');
  EXPECT_EQ(getColName(2), 'c');
  EXPECT_EQ(getColName(3), 'd');
}

TEST(MoveTest, PrintMove) {
  // Test printing of a move
  Move move0{0};
  std::stringstream ss;
  ss << move0;
  EXPECT_EQ(ss.str(), "a4R");
}
