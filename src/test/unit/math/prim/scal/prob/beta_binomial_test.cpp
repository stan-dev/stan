#include <stan/math/prim/scal/prob/beta_binomial_rng.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>

TEST(ProbDistributionBetaBinomial, error_check) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::math::beta_binomial_rng(4,0.6,2.0,rng));

  EXPECT_THROW(stan::math::beta_binomial_rng(-4,0.6,2,rng),std::domain_error);
  EXPECT_THROW(stan::math::beta_binomial_rng(4,-0.6,2,rng),std::domain_error);
  EXPECT_THROW(stan::math::beta_binomial_rng(4,0.6,-2,rng),std::domain_error);
  EXPECT_THROW(stan::math::beta_binomial_rng(4,stan::math::positive_infinity(),
                                             2,rng),
               std::domain_error);
  EXPECT_THROW(stan::math::beta_binomial_rng(4,0.6,
                                             stan::math::positive_infinity(),
                                             rng),
               std::domain_error);
}
