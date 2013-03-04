#include <stan/prob/distributions/univariate/continuous/trunc_normal.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>
#include<boost/math/distributions.hpp>

TEST(ProbDistributionsTruncNormal, random) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::trunc_normal_rng(4.0,3.0,0.0,50.0,rng));
}
