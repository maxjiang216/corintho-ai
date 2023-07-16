#include "trainmc.h"

#include <bitset>
#include <random>

#include "gtest/gtest.h"

#include "game.h"
#include "move.h"
#include "node.h"
#include "util.h"

// Test the training constructor
TEST(TrainMCTest, TrainingConstructor) {
  std::mt19937 generator;
  float to_eval[kGameStateSize];
  TrainMC trainmc(&generator, to_eval);

  EXPECT_EQ(trainmc.num_requests(), 0);
  EXPECT_EQ(trainmc.uninitialized(), true);
  EXPECT_EQ(trainmc.num_nodes(), 0);
}

// Test doing the first iteration
TEST(TrainMCTest, FirstIteration) {
  std::mt19937 generator;
  float to_eval[kGameStateSize];
  TrainMC trainmc(&generator, to_eval);
  EXPECT_EQ(trainmc.doIteration(nullptr, nullptr), false);
  EXPECT_EQ(trainmc.no_requests(), false);
  EXPECT_EQ(trainmc.num_requests(), 1);
  EXPECT_EQ(trainmc.uninitialized(), false);
  EXPECT_EQ(trainmc.num_nodes(), 1);
}

// Test receiving the opponent's move
TEST(TrainMCTest, ReceiveOpponentMove) {
  std::mt19937 generator;
  float to_eval[kGameStateSize];
  TrainMC trainmc(&generator, to_eval);
  trainmc.createRoot(Game(), 0);

  EXPECT_EQ(trainmc.uninitialized(), false);
  EXPECT_EQ(trainmc.num_nodes(), 1);
  EXPECT_EQ(trainmc.root()->depth(), 0);

  Game game;
  int32_t move_id = encodePlace(Space(0, 0), kBase);
  game.doMove(move_id);

  EXPECT_EQ(trainmc.receiveOpponentMove(move_id, game, 1), true);
  EXPECT_EQ(trainmc.root()->depth(), 1);
}

// Test playing out a game with a small number of searches per move
TEST(TrainMCTest, FewPerMove) {
  for (int32_t max_searches = 2; max_searches <= 16; ++max_searches) {
    for (int32_t searches_per_eval = 1; searches_per_eval <= max_searches;
         ++searches_per_eval) {
      std::mt19937 generator(12345);
      float to_eval[kGameStateSize * searches_per_eval];
      TrainMC trainmc(&generator, to_eval, max_searches, searches_per_eval);
      int32_t depth = 0;
      while (trainmc.uninitialized() || !trainmc.root()->terminal()) {
        // Generate random evaluations
        float eval[searches_per_eval];
        float probs[searches_per_eval * kNumMoves];
        std::uniform_real_distribution<float> dist(0.0, 1.0);
        for (int32_t i = 0; i < searches_per_eval; ++i) {
          eval[i] = dist(generator);
        }
        for (int32_t i = 0; i < searches_per_eval; ++i) {
          float sum = 0.0;
          for (int32_t j = 0; j < kNumMoves; ++j) {
            probs[i * kNumMoves + j] = dist(generator);
            sum += probs[i * kNumMoves + j];
          }
          for (int32_t j = 0; j < kNumMoves; ++j) {
            probs[i * kNumMoves + j] /= sum;
          }
        }
        while (!trainmc.doIteration(eval, probs)) {
          assert(trainmc.num_requests() <= searches_per_eval);
        }
        float game_state[kGameStateSize];
        float prob_sample[kNumMoves];
        // Check that the chosen move is legal
        std::bitset<kNumMoves> legal_moves;
        trainmc.root()->getLegalMoves(legal_moves);
        int32_t choice = trainmc.chooseMove(game_state, prob_sample);
        EXPECT_TRUE(legal_moves[choice]);
        ++depth;
        EXPECT_EQ(trainmc.root()->depth(), depth);
      }
      EXPECT_TRUE(trainmc.done());
    }
  }
}

// Test playing out a game with 1600 searches per move
TEST(TrainMCTest, FullGame) {
  for (auto searches_per_eval : {1, 16}) {
    std::mt19937 generator(12345);
    float to_eval[kGameStateSize * searches_per_eval];
    TrainMC trainmc(&generator, to_eval, 1600, searches_per_eval);
    int32_t depth = 0;
    while (trainmc.uninitialized() || !trainmc.root()->terminal()) {
      // Generate random evaluations
      float eval[searches_per_eval];
      float probs[searches_per_eval * kNumMoves];
      std::uniform_real_distribution<float> dist(0.0, 1.0);
      for (int32_t i = 0; i < searches_per_eval; ++i) {
        eval[i] = dist(generator);
      }
      for (int32_t i = 0; i < searches_per_eval; ++i) {
        float sum = 0.0;
        for (int32_t j = 0; j < kNumMoves; ++j) {
          probs[i * kNumMoves + j] = dist(generator);
          sum += probs[i * kNumMoves + j];
        }
        for (int32_t j = 0; j < kNumMoves; ++j) {
          probs[i * kNumMoves + j] /= sum;
        }
      }
      while (!trainmc.doIteration(eval, probs)) {
        for (int32_t i = 0; i < searches_per_eval; ++i) {
          eval[i] = dist(generator);
        }
        for (int32_t i = 0; i < searches_per_eval; ++i) {
          float sum = 0.0;
          for (int32_t j = 0; j < kNumMoves; ++j) {
            probs[i * kNumMoves + j] = dist(generator);
            sum += probs[i * kNumMoves + j];
          }
          for (int32_t j = 0; j < kNumMoves; ++j) {
            probs[i * kNumMoves + j] /= sum;
          }
        }
      }
      float game_state[kGameStateSize];
      float prob_sample[kNumMoves];
      // Check that the chosen move is legal
      std::bitset<kNumMoves> legal_moves;
      trainmc.root()->getLegalMoves(legal_moves);
      int32_t choice = trainmc.chooseMove(game_state, prob_sample);
      EXPECT_TRUE(legal_moves[choice]);
      ++depth;
      EXPECT_EQ(trainmc.root()->depth(), depth);
    }
    EXPECT_TRUE(trainmc.done());
  }
}