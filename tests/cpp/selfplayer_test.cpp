#include "selfplayer.h"
#include "gtest/gtest.h"

// Test the training constructor
TEST(SelfPlayerTest, TrainingConstructor) {
  std::mt19937 *generator = new std::mt19937;
  SelfPlayer selfplayer(1, generator);

  EXPECT_EQ(selfplayer.count_samples(), 0);
}

// Test do first iteration
TEST(SelfPlayerTest, DoFirstIteration) {
  std::mt19937 *generator = new std::mt19937;
  SelfPlayer selfplayer(1, generator);

  selfplayer.do_first_iteration();
  EXPECT_EQ(selfplayer.count_requests(), 1);
}