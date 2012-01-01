#include <gtest/gtest.h>
#include "stan/prob/distributions/hypergeometric.hpp"

TEST(ProbDistributions,Hypergeometric) {
  EXPECT_FLOAT_EQ(-4.119424, stan::prob::hypergeometric_log(5,15,10,10));
  EXPECT_FLOAT_EQ(-2.302585, stan::prob::hypergeometric_log(0,2,3,2));
}
TEST(ProbDistributions,HypergeometricPropto) {
  EXPECT_FLOAT_EQ(-4.119424, stan::prob::hypergeometric_log<true>(5,15,10,10));
  EXPECT_FLOAT_EQ(-2.302585, stan::prob::hypergeometric_log<true>(0,2,3,2));
}
