#include "game.h"
#include "move.h"
#include "node.h"
#include "trainmc.h"
#include "util.h"
#include "gtest/gtest.h"
#include <bitset>
#include <random>

// Test the training constructor
TEST(TrainMCTest, TrainingConstructor) {
  std::mt19937 generator;
  TrainMC trainmc(&generator);

  EXPECT_EQ(trainmc.numNodesSearched(), 0);
  EXPECT_EQ(trainmc.isUninitialized(), true);
  EXPECT_EQ(trainmc.numNodes(), 0);
}

// Test doing the first iteration
TEST(TrainMCTest, FirstIteration) {
  std::mt19937 generator;
  TrainMC trainmc(&generator);
  float to_eval[kGameStateSize];
  trainmc.set_to_eval(to_eval);

  EXPECT_EQ(trainmc.doIteration(nullptr, nullptr), false);
  EXPECT_EQ(trainmc.noEvalsRequested(), false);
  EXPECT_EQ(trainmc.numNodesSearched(), 1);
  EXPECT_EQ(trainmc.isUninitialized(), false);
  EXPECT_EQ(trainmc.numNodes(), 1);
}

// Test receiving the opponent's move
TEST(TrainMCTest, ReceiveOpponentMove) {
  std::mt19937 generator;
  TrainMC trainmc(&generator);
  float to_eval[kGameStateSize];
  trainmc.set_to_eval(to_eval);
  trainmc.createRoot(Game(), 0);

  EXPECT_EQ(trainmc.isUninitialized(), false);
  EXPECT_EQ(trainmc.numNodes(), 1);
  EXPECT_EQ(trainmc.root()->depth(), 0);

  Game game;
  int32_t move_id = encodePlace(Space(0, 0), kBase);
  game.doMove(move_id);

  EXPECT_EQ(trainmc.receiveOpponentMove(move_id, game, 1), true);
  EXPECT_EQ(trainmc.root()->depth(), 1);
}

// Test playing out a game with 2 searches per move
TEST(TrainMCTest, TwoPerMove) {
  std::mt19937 generator(12345);
  TrainMC trainmc(&generator, 2, 2);
  float to_eval[kGameStateSize];
  trainmc.set_to_eval(to_eval);
  int32_t depth = 0;
  while (trainmc.isUninitialized() || !trainmc.root()->terminal()) {
    // Generate random evaluations
    float eval[2];
    float probs[2 * kNumMoves];
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    eval[0] = dist(generator);
    eval[1] = dist(generator);
    for (int32_t i = 0; i < 2; ++i) {
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
}

// Test playing out a game with 1600 searches per move
TEST(TrainMCTest, FullGame) {
  std::mt19937 generator(12345);
  TrainMC trainmc(&generator, 1600, 16);
  float to_eval[kGameStateSize];
  trainmc.set_to_eval(to_eval);
  int32_t depth = 0;
  while (trainmc.isUninitialized() || !trainmc.root()->terminal()) {
    // Generate random evaluations
    float eval[16];
    float probs[16 * kNumMoves];
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    for (int32_t i = 0; i < 16; ++i) {
      eval[i] = dist(generator);
    }
    for (int32_t i = 0; i < 16; ++i) {
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
      for (int32_t i = 0; i < 16; ++i) {
        eval[i] = dist(generator);
      }
      for (int32_t i = 0; i < 16; ++i) {
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
}