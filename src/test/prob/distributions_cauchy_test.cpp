#include <gtest/gtest.h>
#include "stan/prob/distributions/cauchy.hpp"

TEST(ProbDistributions,Cauchy) {
  EXPECT_FLOAT_EQ(-1.837877, stan::prob::cauchy_log(1.0, 0.0, 1.0));
  EXPECT_FLOAT_EQ(-2.323385, stan::prob::cauchy_log(-1.5, 0.0, 1.0));
  EXPECT_FLOAT_EQ(-2.323385, stan::prob::cauchy_log(-2.5, -1.0, 1.0));
  // need test with scale != 1
}
TEST(ProbDistributions,CauchyPropto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::cauchy_log<true>(1.0, 0.0, 1.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::cauchy_log<true>(-1.5, 0.0, 1.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::cauchy_log<true>(-2.5, -1.0, 1.0));
  // need test with scale != 1
}

