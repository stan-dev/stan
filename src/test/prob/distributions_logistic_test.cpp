#include <gtest/gtest.h>
#include "stan/prob/distributions/logistic.hpp"

TEST(ProbDistributions,Logistic) {
  EXPECT_FLOAT_EQ(-2.129645, stan::prob::logistic_log(1.2,0.3,2.0));
  EXPECT_FLOAT_EQ(-3.430098, stan::prob::logistic_log(-1.0,0.2,0.25));
}
TEST(ProbDistributions,LogisticPropto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::logistic_log<true>(1.2,0.3,2.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::logistic_log<true>(-1.0,0.2,0.25));
}
