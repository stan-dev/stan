#include <stan/prob/distributions/univariate/continuous/cauchy.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>

TEST(ProbDistributionsCauchy, random) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::cauchy_rng(2.0,1.0,rng));
}
