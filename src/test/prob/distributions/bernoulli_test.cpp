#include <gtest/gtest.h>
#include "stan/prob/distributions/bernoulli.hpp"

TEST(ProbDistributions,Bernoulli) {
  EXPECT_FLOAT_EQ(std::log(0.25), stan::prob::bernoulli_log(1,0.25));
  EXPECT_FLOAT_EQ(std::log(1.0 - 0.25), stan::prob::bernoulli_log(0,0.25));
}
TEST(ProbDistributions,BernoulliPropto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::bernoulli_log<true>(1,0.25));
  EXPECT_FLOAT_EQ(0.0, stan::prob::bernoulli_log<true>(0,0.25));
}
