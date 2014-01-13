#include "stan/math/functions/inv_sqrt.hpp"
#include <gtest/gtest.h>
#include <stan/math/constants.hpp>

TEST(MathsSpecialFunctions, inv_square) {
  double y = 4.0;
  EXPECT_FLOAT_EQ(1 / 2.0, stan::math::inv_sqrt(y));

  y = 25.0;
  EXPECT_FLOAT_EQ(1 / 5.0, stan::math::inv_sqrt(y));

  y = 0.0;
  EXPECT_FLOAT_EQ(stan::math::positive_infinity(),stan::math::inv_sqrt(y));

  y = -50.0;
  std::isnan(stan::math::inv_sqrt(y));
}
