#include <stan/prob/distributions/univariate/continuous/exponential_normal.hpp>
#include<boost/math/distributions.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>

TEST(ProbDistributionsExpNormal, random) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::exponential_normal_rng(10.0,2.0,1.0,rng));
}

