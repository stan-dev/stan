#include <gtest/gtest.h>
#include <stan/prob/distributions/univariate/continuous/double_exponential.hpp>

TEST(ProbDistributionsDoubleExponential,Cumulative) {
  EXPECT_FLOAT_EQ(0.5, stan::prob::double_exponential_cdf(1.0,1.0,1.0));
  EXPECT_FLOAT_EQ(0.8160603, stan::prob::double_exponential_cdf(2.0,1.0,1.0));
  EXPECT_FLOAT_EQ(0.003368973, stan::prob::double_exponential_cdf(-3.0,2.0,1.0));
  EXPECT_FLOAT_EQ(0.6967347, stan::prob::double_exponential_cdf(1.0,0.0,2.0));
  EXPECT_FLOAT_EQ(0.2246645, stan::prob::double_exponential_cdf(1.9,2.3,0.5));
  EXPECT_FLOAT_EQ(0.10094826, stan::prob::double_exponential_cdf(1.9,2.3,0.25));
}
