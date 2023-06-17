#include "game.h"
#include "util.h"
#include "gtest/gtest.h"
#include <bitset>

TEST(GameTest, DefaultConstructor) {
  // Test that the default constructor creates a game in the starting position
  Game game;

  std::bitset<kNumMoves> legal_moves;
  bool has_lines = game.getLegalMoves(legal_moves);
  EXPECT_FALSE(has_lines);
  // All place moves should be legal in the starting position
  for (int32_t row = 0; row < 4; ++row) {
    for (int32_t col = 0; col < 4; ++col) {
      for (PieceType piece_type : kPieceTypes) {
        int32_t move_id = encodePlace(Space{row, col}, piece_type);
        EXPECT_TRUE(legal_moves[move_id]);
      }
    }
  }
  // No move moves should be legal in the starting position
  for (int32_t id = 0; id < 48; ++id) {
    EXPECT_FALSE(legal_moves[id]);
  }
}

TEST(GameTest, PlaceOnEmptyBoard) {
  // Test that placing a piece on an empty board works
  for (int32_t row = 0; row < 4; ++row) {
    for (int32_t col = 0; col < 4; ++col) {
      for (PieceType piece_type : kPieceTypes) {
        int32_t move_id = encodePlace(Space{row, col}, piece_type);
        Game game;
        game.doMove(move_id);
        // Test that all place moves are legal
        // except placing on the same space twice
        std::bitset<kNumMoves> legal_moves;
        bool has_lines = game.getLegalMoves(legal_moves);
        EXPECT_FALSE(has_lines);  // Check no line
        for (int32_t row2 = 0; row2 < 4; ++row2) {
          for (int32_t col2 = 0; col2 < 4; ++col2) {
            for (PieceType piece_type2 : kPieceTypes) {
              int32_t move_id2 = encodePlace(Space{row2, col2}, piece_type2);
              if (row == row2 && col == col2) {
                EXPECT_FALSE(legal_moves[move_id2]);
              } else {
                EXPECT_TRUE(legal_moves[move_id2]);
              }
            }
          }
        }
        // Check that all move moves are illegal
        for (int32_t id = 0; id < 48; ++id) {
          EXPECT_FALSE(legal_moves[id]);
        }
        // Write out the game state and check that it is correct
        float game_state[kGameStateSize];
        game.writeGameState(game_state);
        // All spaces should be empty except the piece placed and the frozen
        // indicator
        for (int32_t i = 0; i < 4 * kBoardSize; ++i) {
          if (i == row * 16 + col * 4 + piece_type ||
              i == row * 16 + col * 4 + kFrozen) {
            EXPECT_EQ(game_state[i], 1.0);
          } else {
            EXPECT_EQ(game_state[i], 0.0);
          }
        }
        // The second (current) player should have all his pieces
        for (int32_t i = 0; i < 3; ++i) {
          EXPECT_EQ(game_state[4 * kBoardSize + i], 1.0);
        }
        // The first (other) player should have a piece missing
        for (int32_t i = 0; i < 3; ++i) {
          if (i == piece_type) {
            EXPECT_EQ(game_state[4 * kBoardSize + 3 + i], 0.75);
          } else {
            EXPECT_EQ(game_state[4 * kBoardSize + 3 + i], 1.0);
          }
        }
      }
    }
  }
}