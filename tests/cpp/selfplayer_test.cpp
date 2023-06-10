#include "selfplayer.h"
#include "gtest/gtest.h"
#include <iostream>
using namespace std;

// Test the training constructor
TEST(SelfPlayerTest, TrainingConstructor) {
    std::mt19937 *generator = new std::mt19937;
    SelfPlayer selfplayer(1, generator);
    
    EXPECT_EQ(selfplayer.count_samples(), 0);
    cerr << "12";
}

// Test do first iteration
TEST(SelfPlayerTest, DoFirstIteration) {
    std::mt19937 *generator = new std::mt19937;
    SelfPlayer selfplayer(1, generator);

    selfplayer.do_first_iteration();
    EXPECT_EQ(selfplayer.count_requests(), 1);
}