#include "playmc.h"
#include "gtest/gtest.h"

// Test the local constructor
TEST(PlayMCTest, LocalConstructor) {
  PlayMC playmc(10, 1, 1.0, 0.0, false, 0);

  EXPECT_EQ(playmc.is_done(), false);
}