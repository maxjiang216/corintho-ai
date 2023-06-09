#include "gtest/gtest.h"
#include "move.h"

TEST(MoveTest, Constructor) {
    // Create a move with id 0 and check the properties
    Move m0(0);
    EXPECT_EQ(m0.mtype, 0);
    EXPECT_EQ(m0.row1, 0);
    EXPECT_EQ(m0.col1, 0);
    EXPECT_EQ(m0.row2, 0);
    EXPECT_EQ(m0.col2, 1);

    // Create a move with id 48 and check the properties
    Move m48(48);
    EXPECT_EQ(m48.mtype, 1);
    EXPECT_EQ(m48.ptype, 0);
    EXPECT_EQ(m48.row1, 0);
    EXPECT_EQ(m48.col1, 0);
}

TEST(MoveTest, EncodePlace) {
    // Test encoding of a place
    EXPECT_EQ(encode_place(0, 0, 0), 48);
    EXPECT_EQ(encode_place(1, 2, 3), 81);
}

TEST(MoveTest, EncodeMove) {
    // Test encoding of a move
    EXPECT_EQ(encode_move(0, 0, 0, 1), 0);
    EXPECT_EQ(encode_move(1, 2, 2, 2), 17);
}

TEST(MoveTest, GetColName) {
    // Test column name conversion
    EXPECT_EQ(get_col_name(0), 'a');
    EXPECT_EQ(get_col_name(3), 'd');
}

TEST(MoveTest, PrintMove) {
    // Test printing of a move
    Move m0(0);
    std::stringstream ss;
    ss << m0;
    EXPECT_EQ(ss.str(), "a4R");
}
