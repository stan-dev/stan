#include <stan/math/prim/scal/fun/digamma.hpp>
#include <stan/math/prim/scal/fun/inc_beta_derivatives.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, ddz_inc_beta) {
  using stan::math::ddz_inc_beta;
  
  double small_a = 1.5;
  double large_a = 15000;
  
  double small_b = 1.25;
  double large_b = 12500;
  
  double small_z = 0.001;
  double mid_z = 0.5;
  double large_z = 0.999;
  
  EXPECT_FLOAT_EQ(0.063300692, ddz_inc_beta(small_a, small_b, small_z))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(1.1905416, ddz_inc_beta(small_a, small_b, mid_z))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.35587692, ddz_inc_beta(small_a, small_b, large_z))
    << "reasonable values for a, b, x";
  
  EXPECT_FLOAT_EQ(0.0, ddz_inc_beta(large_a, small_b, small_z))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0, ddz_inc_beta(large_a, small_b, mid_z))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.009898182, ddz_inc_beta(large_a, small_b, large_z))
    << "reasonable values for a, b, x";
  
  EXPECT_FLOAT_EQ(0.1848717, ddz_inc_beta(small_a, large_b, small_z))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0, ddz_inc_beta(small_a, large_b, mid_z))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0, ddz_inc_beta(small_a, large_b, large_z))
    << "reasonable values for a, b, x";
  
  EXPECT_FLOAT_EQ(0.0, ddz_inc_beta(large_a, large_b, small_z))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0, ddz_inc_beta(large_a, large_b, mid_z))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0, ddz_inc_beta(large_a, large_b, large_z))
    << "reasonable values for a, b, x";
  
}

TEST(MathFunctions, dda_inc_beta) {
  using stan::math::digamma;
  using stan::math::dda_inc_beta;
  
  double small_a = 1.5;
  double large_a = 15000;
  
  double small_b = 1.25;
  double large_b = 12500;
  
  double small_z = 0.001;
  double mid_z = 0.6;
  double large_z = 0.999;
  
  EXPECT_FLOAT_EQ(-0.00028665637, dda_inc_beta(small_a, small_b, small_z,
                                    digamma(small_a),
                                    digamma(small_a + small_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(-0.23806757, dda_inc_beta(small_a, small_b, mid_z,
                                    digamma(small_a),
                                    digamma(small_a + small_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(-0.00022264493, dda_inc_beta(small_a, small_b, large_z,
                                    digamma(small_a),
                                    digamma(small_a + small_b)))
    << "reasonable values for a, b, x";
  

  EXPECT_FLOAT_EQ(0.0, dda_inc_beta(large_a, small_b, small_z,
                                    digamma(large_a),
                                    digamma(large_a + small_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0, dda_inc_beta(large_a, small_b, mid_z,
                                    digamma(large_a),
                                    digamma(large_a + small_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.00091716705, dda_inc_beta(large_a, small_b, large_z,
                                    digamma(large_a),
                                    digamma(large_a + small_b)))
    << "reasonable values for a, b, x";
  
  EXPECT_FLOAT_EQ(1.8226241, dda_inc_beta(small_a, large_b, small_z,
                                    digamma(small_a),
                                    digamma(small_a + large_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0, dda_inc_beta(small_a, large_b, mid_z,
                                    digamma(small_a),
                                    digamma(small_a + large_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0, dda_inc_beta(small_a, large_b, large_z,
                                    digamma(small_a),
                                    digamma(small_a + large_b)))
    << "reasonable values for a, b, x";
  
  EXPECT_FLOAT_EQ(0.0, dda_inc_beta(large_a, large_b, small_z,
                                    digamma(large_a),
                                    digamma(large_a + large_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(-4.0856207e-14, dda_inc_beta(large_a, large_b, mid_z,
                                    digamma(large_a),
                                    digamma(large_a + large_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0, dda_inc_beta(large_a, large_b, large_z,
                                    digamma(large_a),
                                    digamma(large_a + large_b)))
    << "reasonable values for a, b, x";
  
}

TEST(MathFunctions, ddb_inc_beta) {
  using stan::math::digamma;
  using stan::math::ddb_inc_beta;
  
  double small_a = 1.5;
  double large_a = 15000;
  
  double small_b = 1.25;
  double large_b = 12500;
  
  double small_z = 0.001;
  double mid_z = 0.5;
  double large_z = 0.999;
  
  EXPECT_FLOAT_EQ(3.2996082e-05, ddb_inc_beta(small_a, small_b, small_z,
                                    digamma(small_a),
                                    digamma(small_a + small_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.17945045, ddb_inc_beta(small_a, small_b, mid_z,
                                    digamma(small_a),
                                    digamma(small_a + small_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0019721228, ddb_inc_beta(small_a, small_b, large_z,
                                    digamma(small_a),
                                    digamma(small_a + small_b)))
    << "reasonable values for a, b, x";
  
  EXPECT_FLOAT_EQ(0.0, ddb_inc_beta(large_a, small_b, small_z,
                                    digamma(large_a),
                                    digamma(large_a + small_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0, ddb_inc_beta(large_a, small_b, mid_z,
                                    digamma(large_a),
                                    digamma(large_a + small_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(7.7076669, ddb_inc_beta(large_a, small_b, large_z,
                                    digamma(large_a),
                                    digamma(large_a + small_b)))
    << "reasonable values for a, b, x";
  
  EXPECT_FLOAT_EQ(9.3959293, ddb_inc_beta(small_a, large_b, small_z,
                                    digamma(small_a),
                                    digamma(small_a + large_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0, ddb_inc_beta(small_a, large_b, mid_z,
                                    digamma(small_a),
                                    digamma(small_a + large_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0, ddb_inc_beta(small_a, large_b, large_z,
                                    digamma(small_a),
                                    digamma(small_a + large_b)))
    << "reasonable values for a, b, x";

  EXPECT_FLOAT_EQ(0.0, ddb_inc_beta(large_a, large_b, small_z,
                                    digamma(large_a),
                                    digamma(large_a + large_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0, ddb_inc_beta(large_a, large_b, mid_z,
                                    digamma(large_a),
                                    digamma(large_a + large_b)))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0, ddb_inc_beta(large_a, large_b, large_z,
                                    digamma(large_a),
                                    digamma(large_a + large_b)))
    << "reasonable values for a, b, x";
  
}

