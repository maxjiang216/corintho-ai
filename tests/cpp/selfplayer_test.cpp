#include "selfplayer.h"
#include "gtest/gtest.h"

// Test the training constructor
TEST(SelfPlayerTest, TrainingConstructor) {
  SelfPlayer selfplayer{12345};

  EXPECT_EQ(selfplayer.numSamples(), 0);
}

// Test do first iteration
TEST(SelfPlayerTest, DoFirstIteration) {
  SelfPlayer selfplayer{12345};

  selfplayer.doIteration();
  EXPECT_EQ(selfplayer.numRequests(), 1);
}