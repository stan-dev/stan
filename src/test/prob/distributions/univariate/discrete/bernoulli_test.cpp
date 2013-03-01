#include <stan/prob/distributions/univariate/discrete/bernoulli.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>

TEST(ProbDistributionsBernoulli, random) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::bernoulli_rng(0.6,rng));
}
