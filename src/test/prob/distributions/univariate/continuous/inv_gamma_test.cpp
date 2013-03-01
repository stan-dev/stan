#include <stan/prob/distributions/univariate/continuous/inv_gamma.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>

TEST(ProbDistributionsInvGamma, random) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::inv_gamma_rng(4.0,3.0,rng));
}
