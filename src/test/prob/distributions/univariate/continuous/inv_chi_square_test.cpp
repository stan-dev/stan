#include <stan/prob/distributions/univariate/continuous/inv_chi_square.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>

TEST(ProbDistributionsInvChiSquare, random) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::inv_chi_square_rng(4.0,rng));
}
