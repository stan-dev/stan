#include <stan/prob/distributions/univariate/continuous/uniform.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>

TEST(ProbDistributionsUniform, random) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::uniform_rng(1.0,2.0,rng));
}
