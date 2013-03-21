#include <stan/prob/distributions/univariate/continuous/gumbel.hpp>
#include<boost/math/distributions.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>

TEST(ProbDistributionsGumbel, random) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::gumbel_rng(10.0,2.0,rng));
}

