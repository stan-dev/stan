#include "stan/math/functions/log1p_exp.hpp"
#include <gtest/gtest.h>

TEST(MathsSpecialFunctions, log1p_exp) {
  using stan::math::log1p_exp;

  // exp(10000.0) overflows
  EXPECT_FLOAT_EQ(10000.0,log1p_exp(10000.0));
  EXPECT_FLOAT_EQ(0.0,log1p_exp(-10000.0));
}
