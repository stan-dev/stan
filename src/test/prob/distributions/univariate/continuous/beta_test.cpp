#include <stan/prob/distributions/univariate/continuous/beta.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>

TEST(ProbDistributionsCauchy, random) {
  boost::random::mt19937 rng1;
  boost::random::mt19937 rng2;
  EXPECT_NO_THROW(stan::prob::beta_rng(2.0,1.0,rng1, rng2));
}
