#include <stan/prob/distributions/univariate/continuous/weibull.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>

TEST(ProbDistributionsWeibull, random) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::weibull_rng(2.0,3.0,rng));
}
