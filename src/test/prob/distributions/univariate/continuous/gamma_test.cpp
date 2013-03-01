#include <stan/prob/distributions/univariate/continuous/gamma.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>

TEST(ProbDistributionsGamma, random) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::gamma_rng(2.0,3.0,rng));
}
