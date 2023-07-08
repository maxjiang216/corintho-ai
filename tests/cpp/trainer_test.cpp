#include "trainer.h"
#include "gtest/gtest.h"

#include <iostream>
using namespace std;

// Test the default constructor
TEST(TrainerTest, DefaultConstructor) {
  Trainer trainer;

  EXPECT_EQ(trainer.numRequests(), 0);
  EXPECT_EQ(trainer.numSamples(), 0);
}

// Test the constructor
TEST(TrainerTest, Constructor) {
  Trainer trainer{10, "test", 12345, 1600, 16, 1.0, 0.25, 1, 1, false};

  EXPECT_EQ(trainer.numRequests(), 0);
  EXPECT_EQ(trainer.numSamples(), 0);
}

TEST(TrainerTest, FullGameTraining) {
  for (const auto &num_games : {1, 3}) {
    for (const auto &max_searches : {2, 16, 96, 400}) {
      for (const auto &searches_per_eval : {1, 16, max_searches}) {
        if (searches_per_eval > max_searches)
          continue;
        for (int32_t num_threads = 1; num_threads <= 2; ++num_threads) {
          Trainer trainer{num_games,         "test", 12345, max_searches,
                          searches_per_eval, 1.0,    0.25,  1,
                          num_threads,       false};
          float game_states[num_games * searches_per_eval * kGameStateSize];
          float eval[num_games * searches_per_eval];
          float probs[num_games * searches_per_eval * kNumMoves];
          std::mt19937 generator(12345);
          std::uniform_real_distribution<float> prob_dist(0.0, 1.0);
          std::uniform_real_distribution<float> eval_dist(-1.0, 1.0);
          while (!trainer.doIteration(eval, probs)) {
            EXPECT_TRUE(trainer.numRequests() > 0);
            EXPECT_TRUE(trainer.numRequests() <= num_games * searches_per_eval);
            trainer.writeRequests(game_states);
            for (int32_t i = 0; i < trainer.numRequests(); ++i) {
              for (int32_t j = 0; j < kNumMoves; ++j) {
                probs[i * kNumMoves + j] = prob_dist(generator);
              }
              eval[i] = eval_dist(generator);
            }
          }
          EXPECT_TRUE(trainer.numRequests() == 0);
          EXPECT_TRUE(trainer.numSamples() > 0);
          // Sanity checks for samples
          float sample_game_states[kNumSymmetries * trainer.numSamples() *
                                   kGameStateSize] = {0.0};
          float sample_evals[kNumSymmetries * trainer.numSamples()] = {0.0};
          float sample_probs[kNumSymmetries * trainer.numSamples() *
                             kNumMoves] = {0.0};
          trainer.writeSamples(sample_game_states, sample_evals, sample_probs);
          for (int32_t i = 0; i < trainer.numSamples(); ++i) {
            for (int32_t j = 0; j < kNumSymmetries; ++j) {
              for (int32_t k = 0; k < kGameStateSize; ++k) {
                EXPECT_TRUE(
                    sample_game_states[i * kNumSymmetries * kGameStateSize +
                                       j * kGameStateSize + k] >= 0.0 &&
                    sample_game_states[i * kNumSymmetries * kGameStateSize +
                                       j * kGameStateSize + k] <= 1.0);
              }
              EXPECT_TRUE(sample_evals[i * kNumSymmetries + j] >= -1.0 &&
                          sample_evals[i * kNumSymmetries + j] <= 1.0);
              float sum = 0.0;
              for (int32_t k = 0; k < kNumMoves; ++k) {
                EXPECT_TRUE(sample_probs[i * kNumSymmetries * kNumMoves +
                                         j * kNumMoves + k] >= 0.0 &&
                            sample_probs[i * kNumSymmetries * kNumMoves +
                                         j * kNumMoves + k] <= 1.0);
                sum += sample_probs[i * kNumSymmetries * kNumMoves +
                                    j * kNumMoves + k];
              }
              EXPECT_TRUE(sum >= 0.99 && sum <= 1.01);
            }
          }
          float sorted_game_states[trainer.numSamples() * kGameStateSize];
          float sorted_probs[trainer.numSamples() * kNumMoves];
          for (int32_t i = 0; i < trainer.numSamples(); ++i) {
            for (int32_t j = 0; j < kGameStateSize; ++j) {
              sorted_game_states[i * kGameStateSize + j] =
                  sample_game_states[i * kNumSymmetries * kGameStateSize + j];
            }
            for (int32_t j = 0; j < kNumMoves; ++j) {
              sorted_probs[i * kNumMoves + j] =
                  sample_probs[i * kNumSymmetries * kNumMoves + j];
            }
            // Sort
            std::sort(sorted_game_states + i * kGameStateSize,
                      sorted_game_states + (i + 1) * kGameStateSize,
                      std::greater<float>());
            std::sort(sorted_probs + i * kNumMoves,
                      sorted_probs + (i + 1) * kNumMoves,
                      std::greater<float>());
          }
          // Check that the sorted game states and probabilities for the other
          // symmetries are the same
          for (int32_t i = 1; i < kNumSymmetries; ++i) {
            for (int32_t j = 0; j < trainer.numSamples(); ++j) {
              float game_state[kGameStateSize];
              float prob_sample[kNumMoves];
              for (int32_t k = 0; k < kGameStateSize; ++k) {
                game_state[k] =
                    sample_game_states[j * kNumSymmetries * kGameStateSize +
                                       i * kGameStateSize + k];
              }
              for (int32_t k = 0; k < kNumMoves; ++k) {
                prob_sample[k] = sample_probs[j * kNumSymmetries * kNumMoves +
                                              i * kNumMoves + k];
              }
              // Sort
              std::sort(game_state, game_state + kGameStateSize,
                        std::greater<float>());
              std::sort(prob_sample, prob_sample + kNumMoves,
                        std::greater<float>());
              for (int32_t k = 0; k < kGameStateSize; ++k) {
                EXPECT_TRUE(
                    std::abs(game_state[k] -
                             sorted_game_states[j * kGameStateSize + k]) <
                    1e-6);
              }
              for (int32_t k = 0; k < kNumMoves; ++k) {
                EXPECT_TRUE(std::abs(prob_sample[k] -
                                     sorted_probs[j * kNumMoves + k]) < 1e-6);
              }
            }
          }
        }
      }
    }
  }
}

TEST(TrainerTest, FullGameTesting) {
  for (const auto &num_games : {1, 3}) {
    for (const auto &max_searches : {2, 16, 96, 400}) {
      for (const auto &searches_per_eval : {1, 16, max_searches}) {
        if (searches_per_eval > max_searches)
          continue;
        for (int32_t num_threads = 1; num_threads <= 2; ++num_threads) {
          Trainer trainer{num_games,         "test", 12345, max_searches,
                          searches_per_eval, 1.0,    0.25,  1,
                          num_threads,       true};
          float game_states[num_games * searches_per_eval * kGameStateSize];
          float eval[num_games * searches_per_eval];
          float probs[num_games * searches_per_eval * kNumMoves];
          std::mt19937 generator(12345);
          std::uniform_real_distribution<float> prob_dist(0.0, 1.0);
          std::uniform_real_distribution<float> eval_dist(-1.0, 1.0);
          int32_t to_play = 0;
          while (!trainer.doIteration(eval, probs, to_play)) {
            EXPECT_TRUE(trainer.numRequests(to_play) <=
                        num_games * searches_per_eval);
            if (trainer.numRequests(to_play) == 0) {
              to_play = 1 - to_play;
              continue;
            }
            trainer.writeRequests(game_states, to_play);
            for (int32_t i = 0; i < trainer.numRequests(to_play); ++i) {
              for (int32_t j = 0; j < kNumMoves; ++j) {
                probs[i * kNumMoves + j] = prob_dist(generator);
              }
              eval[i] = eval_dist(generator);
            }
          }
          EXPECT_TRUE(trainer.numSamples() > 0);
        }
      }
    }
  }
}
