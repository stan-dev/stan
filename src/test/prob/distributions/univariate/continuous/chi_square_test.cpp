#include <stan/prob/distributions/univariate/continuous/chi_square.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>

TEST(ProbDistributionsChiSquare, random) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::chi_square_rng(2.0,rng));
}
