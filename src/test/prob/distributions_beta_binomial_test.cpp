#include <gtest/gtest.h>
#include "stan/prob/distributions_beta_binomial.hpp"

TEST(ProbDistributions,BetaBinomial) {
  EXPECT_FLOAT_EQ(-1.854007, stan::prob::beta_binomial_log(5,20,10.0,25.0));
  EXPECT_FLOAT_EQ(-4.376696, stan::prob::beta_binomial_log(25,100,30.0,50.0));
}
