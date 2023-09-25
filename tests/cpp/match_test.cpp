#include "match.h"

#include <memory>
#include <string>

#include "gtest/gtest.h"

#include "trainmc.h"
#include "util.h"

// Test the constructor
TEST(MatchTest, Constructor) {
  Player player1{0, 0, 1600, 16, 1.0, 0.25};
  Player player2{1, 1, 1600, 16, 1.0, 0.25};
  Match match{0, player1, player2, ""};
  EXPECT_EQ(match.to_play(), 0);
}

// Test the constructor with a random player
TEST(MatchTest, ConstructorRandom) {
  Player player1{0, 0, 1600, 16, 1.0, 0.25, true};
  Player player2{1, 1, 1600, 16, 1.0, 0.25};
  Match match{0, player1, player2, ""};
  EXPECT_EQ(match.to_play(), 0);
}

TEST(MatchTest, FewSearches) {
  for (int32_t max_searches = 2; max_searches <= 16; ++max_searches) {
    for (int32_t searches_per_eval = 1; searches_per_eval <= max_searches;
         ++searches_per_eval) {
      for (int32_t random_id = 0; random_id < 3; ++random_id) {
        Player player1{0,   0,    max_searches,  searches_per_eval,
                       1.0, 0.25, random_id == 0};
        Player player2{1,   1,    max_searches,  searches_per_eval,
                       1.0, 0.25, random_id == 1};
        Match match{12345, player1, player2, ""};
        float eval[searches_per_eval];
        float probs[searches_per_eval * kNumMoves];
        float game_states[searches_per_eval * kGameStateSize];
        std::mt19937 generator(12345);
        std::uniform_real_distribution<float> prob_dist(0.0, 1.0);
        std::uniform_real_distribution<float> eval_dist(-1.0, 1.0);
        while (!match.doIteration(eval, probs)) {
          // Generate random evaluations
          int32_t num_requests = match.num_requests();
          EXPECT_TRUE(num_requests > 0);
          EXPECT_TRUE(num_requests <= searches_per_eval);
          // Sanity check for writing out game states
          match.writeRequests(game_states);
          for (int32_t i = 0; i < num_requests * kGameStateSize; ++i) {
            EXPECT_TRUE(game_states[i] >= 0.0 && game_states[i] <= 1.0);
          }
          // Generate random evaluations
          for (int32_t i = 0; i < num_requests; ++i) {
            eval[i] = eval_dist(generator);
          }
          for (int32_t i = 0; i < num_requests * kNumMoves; ++i) {
            probs[i] = prob_dist(generator);
          }
          // Normalize the probabilities
          for (int32_t i = 0; i < num_requests; ++i) {
            float sum = 0.0;
            for (int32_t j = 0; j < kNumMoves; ++j) {
              sum += probs[i * kNumMoves + j];
            }
            for (int32_t j = 0; j < kNumMoves; ++j) {
              probs[i * kNumMoves + j] /= sum;
            }
          }
        }
      }
    }
  }
}

TEST(MatchTest, FullGame) {
  int32_t max_searches = 1600;
  int32_t searches_per_eval = 16;
  for (int32_t random_id = 0; random_id < 3; ++random_id) {
    Player player1{0,   0,    max_searches,  searches_per_eval,
                   1.0, 0.25, random_id == 0};
    Player player2{1,   1,    max_searches,  searches_per_eval,
                   1.0, 0.25, random_id == 1};
    Match match{12345, player1, player2, ""};
    float eval[searches_per_eval];
    float probs[searches_per_eval * kNumMoves];
    float game_states[searches_per_eval * kGameStateSize];
    std::mt19937 generator(12345);
    std::uniform_real_distribution<float> prob_dist(0.0, 1.0);
    std::uniform_real_distribution<float> eval_dist(-1.0, 1.0);
    while (!match.doIteration(eval, probs)) {
      // Generate random evaluations
      int32_t num_requests = match.num_requests();
      EXPECT_TRUE(num_requests > 0);
      EXPECT_TRUE(num_requests <= searches_per_eval);
      // Sanity check for writing out game states
      match.writeRequests(game_states);
      for (int32_t i = 0; i < num_requests * kGameStateSize; ++i) {
        EXPECT_TRUE(game_states[i] >= 0.0 && game_states[i] <= 1.0);
      }
      // Generate random evaluations
      for (int32_t i = 0; i < num_requests; ++i) {
        eval[i] = eval_dist(generator);
      }
      for (int32_t i = 0; i < num_requests * kNumMoves; ++i) {
        probs[i] = prob_dist(generator);
      }
      // Normalize the probabilities
      for (int32_t i = 0; i < num_requests; ++i) {
        float sum = 0.0;
        for (int32_t j = 0; j < kNumMoves; ++j) {
          sum += probs[i * kNumMoves + j];
        }
        for (int32_t j = 0; j < kNumMoves; ++j) {
          probs[i * kNumMoves + j] /= sum;
        }
      }
    }
  }
}