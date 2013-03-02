#include <stan/prob/distributions/univariate/discrete/binomial.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>

TEST(ProbDistributionBinomiali, random) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::binomial_rng(4,0.6,rng));
}
