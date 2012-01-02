#include <gtest/gtest.h>
#include <stan/prob/distributions/poisson.hpp>

TEST(ProbDistributions,Poisson) {
  EXPECT_FLOAT_EQ(-2.900934, stan::prob::poisson_log(17,13.0));
  EXPECT_FLOAT_EQ(-145.3547, stan::prob::poisson_log(192,42.0));
  EXPECT_FLOAT_EQ(-3.0, stan::prob::poisson_log(0, 3.0));
  EXPECT_FLOAT_EQ(log(0.0), stan::prob::poisson_log (0, 0.0));
}
TEST(ProbDistributions,PoissonPropto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::poisson_log<true>(17,13.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::poisson_log<true>(192,42.0));
  EXPECT_FLOAT_EQ(0.0, stan::prob::poisson_log<true>(0, 3.0));
  EXPECT_FLOAT_EQ(log(0.0), stan::prob::poisson_log<true>(0, 0.0));
}
