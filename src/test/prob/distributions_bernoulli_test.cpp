#include <gtest/gtest.h>
#include "stan/prob/distributions_bernoulli.hpp"

TEST(ProbDistributions,Bernoulli) {
  EXPECT_FLOAT_EQ(std::log(0.25), stan::prob::bernoulli_log(1,0.25));
  EXPECT_FLOAT_EQ(std::log(1.0 - 0.25), stan::prob::bernoulli_log(0,0.25));
}
