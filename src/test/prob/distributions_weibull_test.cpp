#include <gtest/gtest.h>
#include "stan/prob/distributions_weibull.hpp"

TEST(ProbDistributions,Weibull) {
  EXPECT_FLOAT_EQ(-2.0, stan::prob::weibull_log(2.0,1.0,1.0));
  EXPECT_FLOAT_EQ(-3.277094, stan::prob::weibull_log(0.25,2.9,1.8));
  EXPECT_FLOAT_EQ(-102.8962, stan::prob::weibull_log(3.9,1.7,0.25));
}
TEST(ProbDistributions,WeibullPropto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::weibull_log<true>(2.0,1.0,1.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::weibull_log<true>(0.25,2.9,1.8));
  EXPECT_FLOAT_EQ(0.0, stan::prob::weibull_log<true>(3.9,1.7,0.25));
}

TEST(ProbDistributionsCumulative,Weibull) {
  EXPECT_FLOAT_EQ(0.86466472, stan::prob::weibull_p(2.0,1.0,1.0));
  EXPECT_FLOAT_EQ(0.0032585711, stan::prob::weibull_p(0.25,2.9,1.8));
  EXPECT_FLOAT_EQ(1.0, stan::prob::weibull_p(3.9,1.7,0.25));
}
TEST(ProbDistributionsCumulative,WeibullPropto) {
  EXPECT_FLOAT_EQ(1.0, stan::prob::weibull_p<true>(2.0,1.0,1.0));
  EXPECT_FLOAT_EQ(1.0, stan::prob::weibull_p<true>(0.25,2.9,1.8));
  EXPECT_FLOAT_EQ(1.0, stan::prob::weibull_p<true>(3.9,1.7,0.25));
}
