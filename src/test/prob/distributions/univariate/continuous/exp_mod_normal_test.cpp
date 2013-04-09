#include <stan/prob/distributions/univariate/continuous/exp_mod_normal.hpp>
#include<boost/math/distributions.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>

TEST(ProbDistributionsExpModNormal, random) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::exp_mod_normal_rng(10.0,2.0,1.0,rng));
}

