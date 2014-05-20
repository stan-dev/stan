#include <stan/prob/distributions/univariate/continuous/von_mises.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>

TEST(ProbDistributionsVonMises, error_check) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::von_mises_rng(1.0,2.0,rng));

  EXPECT_NO_THROW(stan::prob::von_mises_rng(stan::math::negative_infinity(),2.0,rng));
  EXPECT_THROW(stan::prob::von_mises_rng(1,stan::math::positive_infinity(),rng),
               std::domain_error);
  EXPECT_THROW(stan::prob::von_mises_rng(1,-3,rng), std::domain_error);
  EXPECT_NO_THROW(stan::prob::von_mises_rng(2,1,rng));
}
