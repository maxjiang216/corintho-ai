#include "tourney.h"

#include <cstdint>

#include "gtest/gtest.h"

#include "util.h"

TEST(TourneyTest, Constructor) {
  Tourney tourney{1};
  EXPECT_EQ(tourney.num_requests(0), 0);
}

TEST(TourneyTest, AddPlayer) {
  Tourney tourney{1};
  tourney.addPlayer(0, 0);
  EXPECT_EQ(tourney.num_requests(0), 0);
}

TEST(TourneyTest, AddMatch) {
  Tourney tourney{1};
  tourney.addPlayer(0, 0);
  tourney.addPlayer(1, 0);
  tourney.addMatch(0, 1);
  tourney.addMatch(1, 0);
  EXPECT_EQ(tourney.num_requests(0), 0);
  EXPECT_EQ(tourney.num_requests(1), 0);
}

TEST(TourneyTest, FewSearches) {
  Tourney tourney{1};
  int32_t counter = 1;
  for (const auto max_searches : {2, 3, 4}) {
    for (const auto searches_per_eval : {1, max_searches}) {
      tourney.addPlayer(counter, counter, max_searches, searches_per_eval);
      ++counter;
    }
  }
  tourney.addPlayer(0, 0, 1, 1, 0.0, 0.0, true);
  for (int32_t i = 0; i < counter; ++i) {
    for (int32_t j = 0; j < counter; ++j) {
      if (i != j) {
        tourney.addMatch(i, j);
      }
    }
  }
  float game_states[4 * kGameStateSize * 6] = {0.0};
  float eval[4 * kGameStateSize * 6] = {0.0};
  float probs[4 * kNumMoves * 6] = {0.0};
  std::mt19937 generator(12345);
  std::uniform_real_distribution<float> prob_dist(0.0, 1.0);
  std::uniform_real_distribution<float> eval_dist(-1.0, 1.0);
  while (true) {
    // Do iterations
    bool all_done = true;
    for (int32_t i = 0; i < counter; ++i) {
      int32_t num_requests = tourney.num_requests(i);
      if (num_requests > 0) {
        tourney.writeRequests(game_states, i);
        for (int32_t j = 0; j < num_requests; ++j) {
          for (int32_t k = 0; k < kNumMoves; ++k) {
            probs[j * kNumMoves + k] = prob_dist(generator);
          }
          eval[j] = eval_dist(generator);
        }
      }
      all_done &= tourney.doIteration(eval, probs, i);
    }
    if (all_done)
      break;
  }
}