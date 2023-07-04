#include "trainmc.h"
#include "gtest/gtest.h"
#include <random>

// Test the training constructor
TEST(TrainMCTest, TrainingConstructor) {
  std::mt19937 generator;
  TrainMC trainmc(&generator);

  EXPECT_EQ(trainmc.isUninitialized(), true);
  EXPECT_EQ(trainmc.numNodes(), 0);
}