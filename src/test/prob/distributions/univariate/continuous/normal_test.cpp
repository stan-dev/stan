#include <stan/prob/distributions/univariate/continuous/normal.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>

TEST(ProbDistributionsNormal, random) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::normal_rng(2.0,1.0,rng));
}
