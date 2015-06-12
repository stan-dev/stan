#include <stan/math/prim/scal/fun/digamma.hpp>
#include <stan/math/prim/scal/fun/inc_beta_ddb.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, inc_beta_ddb) {
  using stan::math::digamma;
  using stan::math::inc_beta_ddb;
  
  double small_a = 1.5;
  double large_a = 15000;
  
  double small_b = 1.25;
  double large_b = 12500;
  
  double small_z = 0.001;
  double mid_z = 0.5;
  double large_z = 0.999;
  
  EXPECT_FLOAT_EQ(3.2996082e-05, inc_beta_ddb(small_a, small_b, small_z,
                                              digamma(small_a),
                                              digamma(small_a + small_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.17945045, inc_beta_ddb(small_a, small_b, mid_z,
                                           digamma(small_a),
                                           digamma(small_a + small_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0019721228, inc_beta_ddb(small_a, small_b, large_z,
                                             digamma(small_a),
                                             digamma(small_a + small_b)))
    << "reasonable values for a, b, x";
  
  EXPECT_FLOAT_EQ(0.0, inc_beta_ddb(large_a, small_b, small_z,
                                    digamma(large_a),
                                    digamma(large_a + small_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0, inc_beta_ddb(large_a, small_b, mid_z,
                                    digamma(large_a),
                                    digamma(large_a + small_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(7.7076669, inc_beta_ddb(large_a, small_b, large_z,
                                          digamma(large_a),
                                          digamma(large_a + small_b)))
    << "reasonable values for a, b, x";
  
  EXPECT_FLOAT_EQ(9.3959293, inc_beta_ddb(small_a, large_b, small_z,
                                          digamma(small_a),
                                          digamma(small_a + large_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0, inc_beta_ddb(small_a, large_b, mid_z,
                                    digamma(small_a),
                                    digamma(small_a + large_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0, inc_beta_ddb(small_a, large_b, large_z,
                                    digamma(small_a),
                                    digamma(small_a + large_b)))
    << "reasonable values for a, b, x";

  EXPECT_FLOAT_EQ(0.0, inc_beta_ddb(large_a, large_b, small_z,
                                    digamma(large_a),
                                    digamma(large_a + large_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0, inc_beta_ddb(large_a, large_b, mid_z,
                                    digamma(large_a),
                                    digamma(large_a + large_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0, inc_beta_ddb(large_a, large_b, large_z,
                                    digamma(large_a),
                                    digamma(large_a + large_b)))
    << "reasonable values for a, b, x";
  
}

