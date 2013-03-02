#include <stan/prob/distributions/univariate/continuous/logistic.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>

TEST(ProbDistributionsLogistic, random) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::logistic_rng(4.0,3.0,rng));
}
