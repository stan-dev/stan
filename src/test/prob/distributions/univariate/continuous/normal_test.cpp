#include <stan/prob/distributions/univariate/continuous/normal.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>
#include <iostream>

using std::vector;
using std::numeric_limits;

TEST(ProbDistributionsNormal, random) {
  boost::random::mt19937 rng;
  double variate = stan::prob::normal_random(2.0,1.0,rng);
  std::cout << "variate=" << variate << std::endl;
  EXPECT_EQ(1,1);
}
