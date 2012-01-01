#include <gtest/gtest.h>
#include "stan/prob/distributions/neg_binomial.hpp"

TEST(ProbDistributions,NegBinomial) {
  EXPECT_FLOAT_EQ(-7.786663, stan::prob::neg_binomial_log(10,2.0,1.5));
  EXPECT_FLOAT_EQ(-142.6147, stan::prob::neg_binomial_log(100,3.0,3.5));
}
TEST(ProbDistributions,NegBinomialPropto) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::neg_binomial_log<true>(10,2.0,1.5));
  EXPECT_FLOAT_EQ(0.0, stan::prob::neg_binomial_log<true>(100,3.0,3.5));
}
