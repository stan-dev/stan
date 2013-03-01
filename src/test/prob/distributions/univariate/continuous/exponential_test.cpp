#include <stan/prob/distributions/univariate/continuous/exponential.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>

TEST(ProbDistributionsExponential, random) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::exponential_rng(2.0,rng));
}
