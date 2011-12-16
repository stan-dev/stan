#include <gtest/gtest.h>
#include "stan/prob/distributions_binomial.hpp"

TEST(ProbDistributions,Binomial) {
  EXPECT_FLOAT_EQ(-2.144372, stan::prob::binomial_log(10,20,0.4));
  EXPECT_FLOAT_EQ(-16.09438, stan::prob::binomial_log(0,10,0.8));
}
TEST(ProbDistributions,BinomialPropto) {
  EXPECT_FLOAT_EQ(-2.144372, stan::prob::binomial_log(10,20,0.4));
  EXPECT_FLOAT_EQ(-16.09438, stan::prob::binomial_log(0,10,0.8));
}
