#include "stan/math/functions/fma.hpp"
#include <gtest/gtest.h>

TEST(MathFunctions, fma) {
  using stan::math::fma;
  
  EXPECT_FLOAT_EQ(5.0, fma(1.0,2.0,3.0));
  EXPECT_FLOAT_EQ(10.0, fma(2.0,3.0,4.0));
}

