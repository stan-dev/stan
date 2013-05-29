#include "stan/math/functions/inv.hpp"
#include <gtest/gtest.h>
#include <stan/math/constants.hpp>

TEST(MathsSpecialFunctions, inv) {
  double y = 2.0;
  EXPECT_FLOAT_EQ(1 / y, stan::math::inv(y));

  y = 0.0;
  EXPECT_FLOAT_EQ(stan::math::positive_infinity(),stan::math::inv(y));

  y = -32.7;
  EXPECT_FLOAT_EQ(1 / y, stan::math::inv(y));
}
