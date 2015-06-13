#include <stan/math/prim/scal/prob/exp_mod_normal_rng.hpp>
#include <boost/math/distributions.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>

TEST(ProbDistributionsExpModNormal, error_check) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::math::exp_mod_normal_rng(10.0,2.0,1.0,rng));

  EXPECT_THROW(stan::math::exp_mod_normal_rng(10.0,2.0,-1.0,rng),
               std::domain_error);
  EXPECT_THROW(stan::math::exp_mod_normal_rng(10.0,-2.0,1.0,rng),
               std::domain_error);
  EXPECT_THROW(stan::math::exp_mod_normal_rng(10.0,2,
                                              stan::math::positive_infinity(),
                                              rng),
               std::domain_error);
  EXPECT_THROW(stan::math::exp_mod_normal_rng(stan::math::positive_infinity(),2,
                                              1.0,rng)
               ,std::domain_error);
}

