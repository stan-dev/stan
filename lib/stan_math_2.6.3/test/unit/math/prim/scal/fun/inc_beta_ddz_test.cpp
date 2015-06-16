#include <stan/math/prim/scal/fun/digamma.hpp>
#include <stan/math/prim/scal/fun/inc_beta_ddz.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, inc_beta_ddz) {
  using stan::math::inc_beta_ddz;
  
  double small_a = 1.5;
  double large_a = 15000;
  
  double small_b = 1.25;
  double large_b = 12500;
  
  double small_z = 0.001;
  double mid_z = 0.5;
  double large_z = 0.999;
  
  EXPECT_FLOAT_EQ(0.063300692, inc_beta_ddz(small_a, small_b, small_z))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(1.1905416, inc_beta_ddz(small_a, small_b, mid_z))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.35587692, inc_beta_ddz(small_a, small_b, large_z))
    << "reasonable values for a, b, x";
  
  EXPECT_FLOAT_EQ(0.0, inc_beta_ddz(large_a, small_b, small_z))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0, inc_beta_ddz(large_a, small_b, mid_z))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.009898182, inc_beta_ddz(large_a, small_b, large_z))
    << "reasonable values for a, b, x";
  
  EXPECT_FLOAT_EQ(0.1848717, inc_beta_ddz(small_a, large_b, small_z))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0, inc_beta_ddz(small_a, large_b, mid_z))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0, inc_beta_ddz(small_a, large_b, large_z))
    << "reasonable values for a, b, x";
  
  EXPECT_FLOAT_EQ(0.0, inc_beta_ddz(large_a, large_b, small_z))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0, inc_beta_ddz(large_a, large_b, mid_z))
    << "reasonable values for a, b, x";
  EXPECT_FLOAT_EQ(0.0, inc_beta_ddz(large_a, large_b, large_z))
    << "reasonable values for a, b, x";
  
}

