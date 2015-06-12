#include <stan/math/prim/scal/fun/digamma.hpp>
#include <stan/math/prim/scal/fun/inc_beta_dda.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, inc_beta_dda) {
  using stan::math::digamma;
  using stan::math::inc_beta_dda;
  
  double small_a = 1.5;
  double large_a = 15000;
  
  double small_b = 1.25;
  double large_b = 12500;
  
  double small_z = 0.001;
  double mid_z = 0.6;
  double large_z = 0.999;
  
  EXPECT_FLOAT_EQ(-0.00028665637, inc_beta_dda(small_a, small_b, small_z,
                                               digamma(small_a),
                                               digamma(small_a + small_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(-0.23806757, inc_beta_dda(small_a, small_b, mid_z,
                                            digamma(small_a),
                                            digamma(small_a + small_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(-0.00022264493, inc_beta_dda(small_a, small_b, large_z,
                                               digamma(small_a),
                                               digamma(small_a + small_b)))
    << "reasonable values for a, b, x";
  

  EXPECT_FLOAT_EQ(0.0, inc_beta_dda(large_a, small_b, small_z,
                                    digamma(large_a),
                                    digamma(large_a + small_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0, inc_beta_dda(large_a, small_b, mid_z,
                                    digamma(large_a),
                                    digamma(large_a + small_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.00091716705, inc_beta_dda(large_a, small_b, large_z,
                                              digamma(large_a),
                                              digamma(large_a + small_b)))
    << "reasonable values for a, b, x";
  
  EXPECT_FLOAT_EQ(1.8226241, inc_beta_dda(small_a, large_b, small_z,
                                          digamma(small_a),
                                          digamma(small_a + large_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0, inc_beta_dda(small_a, large_b, mid_z,
                                    digamma(small_a),
                                    digamma(small_a + large_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0, inc_beta_dda(small_a, large_b, large_z,
                                    digamma(small_a),
                                    digamma(small_a + large_b)))
    << "reasonable values for a, b, x";
  
  EXPECT_FLOAT_EQ(0.0, inc_beta_dda(large_a, large_b, small_z,
                                    digamma(large_a),
                                    digamma(large_a + large_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(-4.0856207e-14, inc_beta_dda(large_a, large_b, mid_z,
                                               digamma(large_a),
                                               digamma(large_a + large_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0, inc_beta_dda(large_a, large_b, large_z,
                                    digamma(large_a),
                                    digamma(large_a + large_b)))
    << "reasonable values for a, b, x";
}
