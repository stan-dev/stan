#include <stan/prob/distributions/univariate/discrete/poisson.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>

TEST(ProbDistributionsPoisson, random) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::poisson_rng(6, rng));
}
