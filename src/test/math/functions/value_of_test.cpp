#include "stan/math/functions/value_of.hpp"
#include <gtest/gtest.h>

TEST(MathFunctions, value_of) {
  using stan::math::value_of;
  double x = 5.0;
  EXPECT_FLOAT_EQ(5.0,value_of(x));
  EXPECT_FLOAT_EQ(5.0,value_of(5));
}
