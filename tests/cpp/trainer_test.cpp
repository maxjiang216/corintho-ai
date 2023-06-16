#include "trainer.h"
#include "gtest/gtest.h"

// Test the default constructor
TEST(TrainerTest, DefaultConstructor) {
  Trainer trainer;

  EXPECT_EQ(trainer.count_samples(), 0);
}