#include "move.h"
#include "gtest/gtest.h"
#include <iostream>

// Very basic test for the Move class constructor
TEST(MoveTest, Constructor) {
  // Create a move with id 0 and check the properties
  Move m0(0);
  EXPECT_EQ(m0.move_type(), Move::MoveType::kMove);
  EXPECT_EQ(m0.row1(), 0);
  EXPECT_EQ(m0.col1(), 0);
  EXPECT_EQ(m0.row2(), 0);
  EXPECT_EQ(m0.col2(), 1);

  // Create a move with id 48 and check the properties
  Move m48(48);
  EXPECT_EQ(m48.move_type(), Move::MoveType::kPlace);
  EXPECT_EQ(m48.piece_type(), 0);
  EXPECT_EQ(m48.row1(), 0);
  EXPECT_EQ(m48.col1(), 0);
}

TEST(MoveTest, EncodePlace) {
  // Test encoding of a place
  EXPECT_EQ(encodePlace(0, 0, 0), 48);
  EXPECT_EQ(encodePlace(1, 2, 3), 75);
}

TEST(MoveTest, EncodeMove) {
  // Test encoding of a move
  EXPECT_EQ(encodeMove(0, 0, 0, 1), 0);
  EXPECT_EQ(encodeMove(1, 2, 2, 2), 18);
}

TEST(MoveTest, RecoverMove) {
  // Test that decoding and encoding a move returns the same move
  for (int i = 0; i < 48; i++) {
    Move m(i);
    //std::cerr << i << encodeMove(m.row1(), m.col1(), m.row2(), m.col2()) << std::endl;
    EXPECT_EQ(encodeMove(m.row1(), m.col1(), m.row2(), m.col2()), i);
  }
  for (int i = 48; i < 96; i++) {
    Move m(i);
    //std::cerr << i << encodePlace(m.piece_type(), m.row1(), m.col1()) << std::endl;
    EXPECT_EQ(encodePlace(m.piece_type(), m.row1(), m.col1()), i);
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
  Move m0(0);
  std::stringstream ss;
  ss << m0;
  EXPECT_EQ(ss.str(), "a4R");
}
