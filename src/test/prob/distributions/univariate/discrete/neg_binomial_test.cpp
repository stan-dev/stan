#include <stan/prob/distributions/univariate/discrete/neg_binomial.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>

TEST(ProbDistributionsNegBinomial, random) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::neg_binomial_rng(6, 2.0, rng));
}
