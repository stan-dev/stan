#include <gtest/gtest.h>
#include <stan/prob/distributions/univariate/continuous/cauchy.hpp>

TEST(ProbDistributionsCauchy,Cumulative) {
  using stan::prob::cauchy_cdf;
  EXPECT_FLOAT_EQ(0.75, cauchy_cdf(1.0, 0.0, 1.0));
  EXPECT_FLOAT_EQ(0.187167, cauchy_cdf(-1.5, 0.0, 1.0));
  EXPECT_FLOAT_EQ(0.187167, cauchy_cdf(-2.5, -1.0, 1.0));
}
