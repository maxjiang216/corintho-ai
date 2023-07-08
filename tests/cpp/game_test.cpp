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

TEST(GameTest, WebAppConstructor) {
  // Starting position
  int32_t board[4 * kBoardSize] = {0};
  int32_t to_play = 0;
  int32_t pieces[6] = {4, 4, 4, 4, 4, 4};
  Game game = Game(board, to_play, pieces);
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

  // Terminal position with a line
  board[2 * 4 + kCapital] = 1;
  board[5 * 4 + kCapital] = 1;
  board[8 * 4 + kCapital] = 1;
  board[8 * 4 + kFrozen] = 1;
  to_play = 1;
  pieces[2] = 2;
  pieces[5] = 3;
  game = Game(board, to_play, pieces);
  has_lines = game.getLegalMoves(legal_moves);
  EXPECT_TRUE(has_lines);
  // No legal moves
  EXPECT_TRUE(legal_moves.none());
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

TEST(GameTest, MoveCapitalOnBaseColumn) {
  // Offsets for more testing
  for (int32_t y = 0; y < 4; ++y) {
    // Move doesn't work with wrap-around
    for (int32_t x = 0; x < 3; ++x) {
      // Move capital on base and column tower
      Game game;
      game.doMove(encodePlace(Space{y, x}, kBase));
      game.doMove(encodePlace(Space{y, x + 1}, kCapital));
      game.doMove(encodePlace(Space{y, x}, kColumn));
      game.doMove(encodePlace(Space{(y + 1) % 4, x}, kBase));
      game.doMove(encodeMove(Space{y, x + 1}, Space{y, x}));
      std::bitset<kNumMoves> legal_moves;
      bool has_lines = game.getLegalMoves(legal_moves);
      EXPECT_FALSE(has_lines);
      // Check that all move moves are illegal
      for (int32_t id = 0; id < 48; ++id) {
        EXPECT_FALSE(legal_moves[id]);
      }
      // Check that all place moves are legal
      // except at (y, x) and base or capital at ((y+1)%4, x)
      for (int32_t row = 0; row < 4; ++row) {
        for (int32_t col = 0; col < 4; ++col) {
          for (PieceType piece_type : kPieceTypes) {
            int32_t move_id = encodePlace(Space{row, col}, piece_type);
            if (row == y && col == x ||
                row == (y + 1) % 4 && col == x &&
                    (piece_type == kBase || piece_type == kCapital)) {
              EXPECT_FALSE(legal_moves[move_id]);
            } else {
              EXPECT_TRUE(legal_moves[move_id]);
            }
          }
        }
      }
      // Write out the game state and check that it is correct
      float game_state[kGameStateSize];
      game.writeGameState(game_state);
      // All spaces should be empty except (0, 0) (which is full)
      // and (1, 0) (which has a base)
      for (int32_t i = 0; i < 4 * kBoardSize; ++i) {
        if (i / 4 == y * 4 + x || i == ((y + 1) % 4) * 16 + x * 4 + kBase) {
          EXPECT_EQ(game_state[i], 1.0);
        } else {
          EXPECT_EQ(game_state[i], 0.0);
        }
      }
    }
  }
}

TEST(GameTest, ColumnCapitalOnBase) {
  // Offsets for more testing
  for (int32_t y = 0; y < 4; ++y) {
    // Move doesn't work with wrap-around
    for (int32_t x = 0; x < 3; ++x) {
      // Move column and capital tower on base
      Game game;
      game.doMove(encodePlace(Space{y, x}, kColumn));
      game.doMove(encodePlace(Space{y, x + 1}, kBase));
      game.doMove(encodePlace(Space{y, x}, kCapital));
      game.doMove(encodePlace(Space{(y + 1) % 4, x}, kCapital));
      game.doMove(encodeMove(Space{y, x}, Space{y, x + 1}));
      std::bitset<kNumMoves> legal_moves;
      bool has_lines = game.getLegalMoves(legal_moves);
      EXPECT_FALSE(has_lines);
      // Check that all move moves are illegal
      for (int32_t id = 0; id < 48; ++id) {
        EXPECT_FALSE(legal_moves[id]);
      }
      // Check that all place moves are legal except at (0, 0) and (1, 0)
      for (int32_t row = 0; row < 4; ++row) {
        for (int32_t col = 0; col < 4; ++col) {
          for (PieceType piece_type : kPieceTypes) {
            int32_t move_id = encodePlace(Space{row, col}, piece_type);
            if ((row == y && col == x + 1) ||
                (row == (y + 1) % 4 && col == x)) {
              EXPECT_FALSE(legal_moves[move_id]);
            } else {
              EXPECT_TRUE(legal_moves[move_id]);
            }
          }
        }
      }
      // Write out the game state and check that it is correct
      float game_state[kGameStateSize];
      game.writeGameState(game_state);
      // All spaces should be empty except (0, 1) (which is full)
      // and (1, 0) (which has a capital)
      for (int32_t i = 0; i < 4 * kBoardSize; ++i) {
        if (i / 4 == y * 4 + (x + 1) ||
            i == ((y + 1) % 4) * 16 + x * 4 + kCapital) {
          EXPECT_EQ(game_state[i], 1.0);
        } else {
          EXPECT_EQ(game_state[i], 0.0);
        }
      }
    }
  }
}

TEST(GameTest, NoPieceLeft) {
  // Test that a player cannot place a piece
  // if he has none left
  for (int32_t y = 0; y < 4; ++y) {
    for (int32_t x = 0; x < 4; ++x) {
      Game game;
      game.doMove(encodePlace(Space{y, x}, kBase));
      game.doMove(encodePlace(Space{y, (x + 1) % 4}, kColumn));
      game.doMove(encodePlace(Space{y, (x + 2) % 4}, kBase));
      game.doMove(encodePlace(Space{y, (x + 3) % 4}, kCapital));
      game.doMove(encodePlace(Space{(y + 1) % 4, x}, kBase));
      game.doMove(encodePlace(Space{(y + 1) % 4, (x + 1) % 4}, kColumn));
      game.doMove(encodePlace(Space{(y + 1) % 4, (x + 2) % 4}, kBase));
      game.doMove(encodePlace(Space{(y + 1) % 4, (x + 3) % 4}, kCapital));
      std::bitset<kNumMoves> legal_moves;
      bool has_lines = game.getLegalMoves(legal_moves);
      EXPECT_FALSE(has_lines);
      // Check that player 1 cannot place a base
      for (int32_t row = 0; row < 4; ++row) {
        for (int32_t col = 0; col < 4; ++col) {
          int32_t move_id = encodePlace(Space{row, col}, kBase);
          EXPECT_FALSE(legal_moves[move_id]);
        }
      }
    }
  }
}

TEST(GameTest, PlacePieceOnSame) {
  for (int32_t row = 0; row < 4; ++row) {
    for (int32_t col = 0; col < 4; ++col) {
      for (PieceType piece_type : kPieceTypes) {
        Game game;
        game.doMove(encodePlace(Space{row, col}, piece_type));
        game.doMove(
            encodePlace(Space{(row + 1) % 4, (col + 1) % 4}, piece_type));
        std::bitset<kNumMoves> legal_moves;
        bool has_lines = game.getLegalMoves(legal_moves);
        EXPECT_FALSE(has_lines);
        // Check that placing the same piece on (row, col) is illegal
        EXPECT_FALSE(legal_moves[encodePlace(Space{row, col}, piece_type)]);
      }
    }
  }
}

TEST(GameTest, MoveWithFrozen) {
  for (int32_t y = 0; y < 4; ++y) {
    for (int32_t x = 0; x < 4; ++x) {
      Game game;
      game.doMove(encodePlace(Space{y, (x + 3) % 4}, kCapital));
      game.doMove(encodePlace(Space{y, (x + 1) % 4}, kBase));
      game.doMove(encodePlace(Space{y, x}, kColumn));
      std::bitset<kNumMoves> legal_moves;
      bool has_lines = game.getLegalMoves(legal_moves);
      EXPECT_FALSE(has_lines);
      // Check that all move moves are illegal
      for (int32_t id = 0; id < 48; ++id) {
        EXPECT_FALSE(legal_moves[id]);
      }
    }
  }
}

TEST(GameTest, TestLongRowCols) {
  for (bool flip : {false, true}) {
    for (int32_t row = 0; row < 4; ++row) {
      for (PieceType piece_type : kPieceTypes) {
        // Test long lines that must be broken so that there are no lines left
        Game game;
        game.doMove(encodePlace(Space{row, 0, flip}, piece_type));
        game.doMove(encodePlace(Space{row, 1, flip}, piece_type));
        game.doMove(encodePlace(Space{row, 2, flip}, piece_type));
        std::bitset<kNumMoves> legal_moves;
        bool has_lines = game.getLegalMoves(legal_moves);
        EXPECT_TRUE(has_lines);  // there is a line
        // Check legal move count (2 breaks if not capital and 1 extend)
        EXPECT_EQ(legal_moves.count(), piece_type == kCapital ? 1 : 3)
            << "row: " << row << ", piece_type: " << piece_type;
        // Check that extending the line is legal
        EXPECT_TRUE(legal_moves[encodePlace(Space{row, 3, flip}, piece_type)])
            << "row: " << row << ", piece_type: " << piece_type
            << ", legal_moves: " << legal_moves;
        game.doMove(encodePlace(Space{row, 3, flip}, piece_type));
        has_lines = game.getLegalMoves(legal_moves);
        EXPECT_TRUE(has_lines);  // there is a line
        if (piece_type == kCapital) {
          // No legal moves in this case
          EXPECT_TRUE(legal_moves.none());
        } else {
          // Check legal move count
          EXPECT_EQ(legal_moves.count(), 2);
          // Check that all move moves are illegal
          for (int32_t id = 0; id < 48; ++id) {
            EXPECT_FALSE(legal_moves[id]);
          }
          // Check that all place moves not in the middle of the line are
          // illegal and that all legal moves leave no lines
          for (int32_t row2 = 0; row2 < 4; ++row2) {
            for (int32_t col2 = 0; col2 < 4; ++col2) {
              for (PieceType piece_type2 : kPieceTypes) {
                int32_t move_id =
                    encodePlace(Space{row2, col2, flip}, piece_type2);
                if (row2 == row && (col2 == 1 || col2 == 2)) {
                  if (legal_moves[move_id]) {
                    Game game2 = game;
                    game2.doMove(move_id);
                    std::bitset<kNumMoves> legal_moves2;
                    bool has_lines2 = game2.getLegalMoves(legal_moves2);
                    EXPECT_FALSE(has_lines2);  // All lines should be broken
                  }
                } else {
                  EXPECT_FALSE(legal_moves[move_id]);
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(GameTest, TestShortRowCols) {
  for (bool flip : {false, true}) {
    for (int32_t row = 0; row < 4; ++row) {
      for (PieceType piece_type : {kBase, kColumn}) {
        // Test short lines that cannot be extended
        Game game;
        game.doMove(encodePlace(Space{row, 0, flip}, kCapital));
        game.doMove(encodePlace(Space{row, 1, flip}, piece_type));
        game.doMove(encodePlace(Space{row, 2, flip}, piece_type));
        game.doMove(encodePlace(Space{row, 3, flip}, piece_type));
        std::bitset<kNumMoves> legal_moves;
        bool has_lines = game.getLegalMoves(legal_moves);
        EXPECT_TRUE(has_lines);  // there is a line
        // Check legal move count
        EXPECT_EQ(legal_moves.count(), piece_type == kColumn ? 3 : 2);
        // Cannot move the capital onto a base
        if (piece_type == kBase) {
          for (int32_t id = 0; id < 48; ++id) {
            EXPECT_FALSE(legal_moves[id]);
          }
        } else {
          // Check that legal move moves break the line
          for (int32_t id = 0; id < 48; ++id) {
            if (legal_moves[id]) {
              Game game2 = game;
              game2.doMove(id);
              std::bitset<kNumMoves> legal_moves2;
              bool has_lines2 = game2.getLegalMoves(legal_moves2);
              EXPECT_FALSE(has_lines2);  // All lines should be broken
            }
          }
        }
        // Check that all place moves not on the line are illegal
        // and that all legal moves leave no lines
        for (int32_t row2 = 0; row2 < 4; ++row2) {
          for (int32_t col2 = 0; col2 < 4; ++col2) {
            for (PieceType piece_type2 : kPieceTypes) {
              int32_t move_id =
                  encodePlace(Space{row2, col2, flip}, piece_type2);
              if (row2 == row && col2 > 0) {
                if (legal_moves[move_id]) {
                  Game game2 = game;
                  game2.doMove(move_id);
                  std::bitset<kNumMoves> legal_moves2;
                  bool has_lines2 = game2.getLegalMoves(legal_moves2);
                  EXPECT_FALSE(has_lines2);  // All lines should be broken
                }
              } else {
                EXPECT_FALSE(legal_moves[move_id]);
              }
            }
          }
        }
      }
    }
  }
}

TEST(GameTest, TestLongDiags) {
  for (bool flip : {false, true}) {
    for (PieceType piece_type : kPieceTypes) {
      Game game;
      game.doMove(encodePlace(Space{0, flip ? 3 : 0}, piece_type));
      game.doMove(encodePlace(Space{1, flip ? 2 : 1}, piece_type));
      game.doMove(encodePlace(Space{2, flip ? 1 : 2}, piece_type));
      std::bitset<kNumMoves> legal_moves;
      bool has_lines = game.getLegalMoves(legal_moves);
      EXPECT_TRUE(has_lines);  // there is a line
      // Check legal move count (2 breaks if not capital and 1 extend)
      EXPECT_EQ(legal_moves.count(), piece_type == kCapital ? 1 : 3);
      // Check that extending the line is legal
      EXPECT_TRUE(
          legal_moves[encodePlace(Space{3, flip ? 0 : 3}, piece_type)]);
      game.doMove(encodePlace(Space{3, flip ? 0 : 3}, piece_type));
      has_lines = game.getLegalMoves(legal_moves);
      EXPECT_TRUE(has_lines);  // there is a line
      if (piece_type == kCapital) {
        // No legal moves in this case
        EXPECT_TRUE(legal_moves.none());
      } else {
        // Check legal move count
        EXPECT_EQ(legal_moves.count(), 2);
        // Check that all move moves are illegal
        for (int32_t id = 0; id < 48; ++id) {
          EXPECT_FALSE(legal_moves[id]);
        }
        // Check that all place moves not in the middle of the line are illegal
        // and that all legal moves leave no lines
        for (int32_t row = 0; row < 4; ++row) {
          for (int32_t col = 0; col < 4; ++col) {
            int32_t move_id = encodePlace(Space{row, col, flip}, piece_type);
            if ((row == 1 && col == (flip ? 2 : 1)) ||
                (row == 2 && col == (flip ? 1 : 2))) {
              if (legal_moves[move_id]) {
                Game game2 = game;
                game2.doMove(move_id);
                std::bitset<kNumMoves> legal_moves2;
                bool has_lines2 = game2.getLegalMoves(legal_moves2);
                EXPECT_FALSE(has_lines2);  // All lines should be broken
              }
            } else {
              EXPECT_FALSE(legal_moves[move_id]);
            }
          }
        }
      }
    }
  }
}

TEST(GameTest, TestShortDiags) {
  Space sequences[4][3] = {{{2, 0}, {1, 1}, {0, 2}},
                           {{0, 1}, {1, 2}, {2, 3}},
                           {{1, 3}, {2, 2}, {3, 1}},
                           {{3, 2}, {2, 1}, {1, 0}}};
  for (PieceType piece_type : kPieceTypes) {
    for (auto sequence : sequences) {
      Game game;
      game.doMove(encodePlace(sequence[0], piece_type));
      game.doMove(encodePlace(sequence[1], piece_type));
      game.doMove(encodePlace(sequence[2], piece_type));
      std::bitset<kNumMoves> legal_moves;
      bool has_lines = game.getLegalMoves(legal_moves);
      EXPECT_TRUE(has_lines);  // there is a line
      if (piece_type == kCapital) {
        // No legal moves in this case
        EXPECT_TRUE(legal_moves.none());
      } else {
        // Check legal move count
        EXPECT_EQ(legal_moves.count(), 2);
        // Check that all move moves are illegal
        for (int32_t id = 0; id < 48; ++id) {
          EXPECT_FALSE(legal_moves[id]);
        }
        // Check that we must place on a space in the line
        // and that all legal moves leave no lines
        for (int32_t row = 0; row < 4; ++row) {
          for (int32_t col = 0; col < 4; ++col) {
            int32_t move_id = encodePlace(Space{row, col}, piece_type);
            if (row == sequence[0].row && col == sequence[0].col ||
                row == sequence[1].row && col == sequence[1].col ||
                row == sequence[2].row && col == sequence[2].col) {
              if (legal_moves[move_id]) {
                Game game2 = game;
                game2.doMove(move_id);
                std::bitset<kNumMoves> legal_moves2;
                bool has_lines2 = game2.getLegalMoves(legal_moves2);
                EXPECT_FALSE(has_lines2);  // All lines should be broken
              }
            } else {
              EXPECT_FALSE(legal_moves[move_id]);
            }
          }
        }
      }
    }
  }
}

TEST(GameTest, TwoLines) {
  // Test that there are no legal moves if there are two lines
  for (int32_t row = 1; row < 3; ++row) {
    for (int32_t col = 1; col < 3; ++col) {
      for (PieceType piece_type : kPieceTypes) {
        Game game;
        game.doMove(encodePlace(Space{row - 1, col}, piece_type));
        game.doMove(encodePlace(Space{row + 1, col}, piece_type));
        game.doMove(encodePlace(Space{row, col - 1}, piece_type));
        game.doMove(encodePlace(Space{row, col + 1}, piece_type));
        game.doMove(encodePlace(Space{row, col}, piece_type));
        std::bitset<kNumMoves> legal_moves;
        bool has_lines = game.getLegalMoves(legal_moves);
        EXPECT_TRUE(has_lines);  // there are lines
        // Check that there are no legal moves
        EXPECT_TRUE(legal_moves.none());
      }
    }
  }
}