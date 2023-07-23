#include "selfplayer.h"

#include "gtest/gtest.h"

#include "util.h"

// Test the training constructor
TEST(SelfPlayerTest, TrainingConstructor) {
  SelfPlayer selfplayer{12345};

  EXPECT_EQ(selfplayer.num_samples(), 0);
}

// Test do first iteration
TEST(SelfPlayerTest, DoFirstIteration) {
  SelfPlayer selfplayer{12345};

  selfplayer.doIteration();
  EXPECT_EQ(selfplayer.num_requests(), 1);
}

TEST(SelfPlayerTest, FewSearches) {
  for (int32_t max_searches = 1; max_searches <= 16; ++max_searches) {
    for (int32_t searches_per_eval = 1; searches_per_eval <= max_searches;
         ++searches_per_eval) {
      SelfPlayer selfplayer{12345, max_searches, searches_per_eval};
      float eval[searches_per_eval];
      float probs[searches_per_eval * kNumMoves];
      float game_states[searches_per_eval * kGameStateSize];
      std::mt19937 generator(12345);
      std::uniform_real_distribution<float> prob_dist(0.0, 1.0);
      std::uniform_real_distribution<float> eval_dist(-1.0, 1.0);
      while (!selfplayer.doIteration(eval, probs)) {
        // Generate random evaluations
        int32_t num_requests = selfplayer.num_requests();
        EXPECT_TRUE(num_requests > 0);
        EXPECT_TRUE(num_requests <= searches_per_eval);
        // Sanity check for writing out game states
        selfplayer.writeRequests(game_states);
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
      EXPECT_EQ(selfplayer.num_requests(), 0);
      EXPECT_TRUE(selfplayer.num_samples() > 0);
      // Maximum number of moves
      EXPECT_TRUE(selfplayer.num_samples() <= 40);
      float sample_game_states[kNumSymmetries * selfplayer.num_samples() *
                               kGameStateSize] = {0.0};
      float sample_evals[kNumSymmetries * selfplayer.num_samples()] = {0.0};
      float sample_probs[kNumSymmetries * selfplayer.num_samples() *
                         kNumMoves] = {0.0};
      selfplayer.writeSamples(sample_game_states, sample_evals, sample_probs);
      for (int32_t i = 0; i < selfplayer.num_samples(); ++i) {
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
      float sorted_game_states[selfplayer.num_samples() * kGameStateSize];
      float sorted_probs[selfplayer.num_samples() * kNumMoves];
      for (int32_t i = 0; i < selfplayer.num_samples(); ++i) {
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
                  sorted_probs + (i + 1) * kNumMoves, std::greater<float>());
      }
      // Check that the sorted game states and probabilities for the other
      // symmetries are the same
      for (int32_t i = 1; i < kNumSymmetries; ++i) {
        for (int32_t j = 0; j < selfplayer.num_samples(); ++j) {
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
            EXPECT_TRUE(std::abs(game_state[k] -
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

TEST(SelfPlayerTest, FullGame) {
  const int32_t max_searches = 1600;
  const int32_t searches_per_eval = 16;
  SelfPlayer selfplayer{12345, max_searches, searches_per_eval};
  float eval[searches_per_eval];
  float probs[searches_per_eval * kNumMoves];
  float game_states[searches_per_eval * kGameStateSize];
  std::mt19937 generator(12345);
  std::uniform_real_distribution<float> prob_dist(0.0, 1.0);
  std::uniform_real_distribution<float> eval_dist(-1.0, 1.0);
  while (!selfplayer.doIteration(eval, probs)) {
    // Generate random evaluations
    int32_t num_requests = selfplayer.num_requests();
    EXPECT_TRUE(num_requests <= searches_per_eval);
    if (num_requests > 0) {
      // Sanity check for writing out game states
      selfplayer.writeRequests(game_states);
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
  EXPECT_EQ(selfplayer.num_requests(), 0);
  EXPECT_TRUE(selfplayer.num_samples() > 0);
  // Maximum number of moves
  EXPECT_TRUE(selfplayer.num_samples() <= 40);
  float sample_game_states[kNumSymmetries * selfplayer.num_samples() *
                           kGameStateSize] = {0.0};
  float sample_evals[kNumSymmetries * selfplayer.num_samples()] = {0.0};
  float sample_probs[kNumSymmetries * selfplayer.num_samples() * kNumMoves] = {
      0.0};
  selfplayer.writeSamples(sample_game_states, sample_evals, sample_probs);
  for (int32_t i = 0; i < selfplayer.num_samples(); ++i) {
    for (int32_t j = 0; j < kNumSymmetries; ++j) {
      for (int32_t k = 0; k < kGameStateSize; ++k) {
        EXPECT_TRUE(sample_game_states[i * kNumSymmetries * kGameStateSize +
                                       j * kGameStateSize + k] >= 0.0 &&
                    sample_game_states[i * kNumSymmetries * kGameStateSize +
                                       j * kGameStateSize + k] <= 1.0);
      }
      EXPECT_TRUE(sample_evals[i * kNumSymmetries + j] >= -1.0 &&
                  sample_evals[i * kNumSymmetries + j] <= 1.0);
      float sum = 0.0;
      for (int32_t k = 0; k < kNumMoves; ++k) {
        EXPECT_TRUE(
            sample_probs[i * kNumSymmetries * kNumMoves + j * kNumMoves + k] >=
                0.0 &&
            sample_probs[i * kNumSymmetries * kNumMoves + j * kNumMoves + k] <=
                1.0);
        sum +=
            sample_probs[i * kNumSymmetries * kNumMoves + j * kNumMoves + k];
      }
      EXPECT_TRUE(sum >= 0.99 && sum <= 1.01);
    }
  }
  float sorted_game_states[selfplayer.num_samples() * kGameStateSize];
  float sorted_probs[selfplayer.num_samples() * kNumMoves];
  for (int32_t i = 0; i < selfplayer.num_samples(); ++i) {
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
    std::sort(sorted_probs + i * kNumMoves, sorted_probs + (i + 1) * kNumMoves,
              std::greater<float>());
  }
  // Check that the sorted game states and probabilities for the other
  // symmetries are the same
  for (int32_t i = 1; i < kNumSymmetries; ++i) {
    for (int32_t j = 0; j < selfplayer.num_samples(); ++j) {
      float game_state[kGameStateSize];
      float prob_sample[kNumMoves];
      for (int32_t k = 0; k < kGameStateSize; ++k) {
        game_state[k] =
            sample_game_states[j * kNumSymmetries * kGameStateSize +
                               i * kGameStateSize + k];
      }
      for (int32_t k = 0; k < kNumMoves; ++k) {
        prob_sample[k] =
            sample_probs[j * kNumSymmetries * kNumMoves + i * kNumMoves + k];
      }
      // Sort
      std::sort(game_state, game_state + kGameStateSize,
                std::greater<float>());
      std::sort(prob_sample, prob_sample + kNumMoves, std::greater<float>());
      for (int32_t k = 0; k < kGameStateSize; ++k) {
        EXPECT_TRUE(std::abs(game_state[k] -
                             sorted_game_states[j * kGameStateSize + k]) <
                    1e-6);
      }
      for (int32_t k = 0; k < kNumMoves; ++k) {
        EXPECT_TRUE(
            std::abs(prob_sample[k] - sorted_probs[j * kNumMoves + k]) < 1e-6);
      }
    }
  }
}
