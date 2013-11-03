#include "stan/math/functions/abs.hpp"
#include <gtest/gtest.h>

TEST(MathsSpecialFunctions, square) {
  using stan::math::abs;

  double y = 2.0;
  EXPECT_FLOAT_EQ(2.0, abs(y));

  y = 128745.72;
  EXPECT_FLOAT_EQ(128745.72, abs(y));

  y = -y;
  EXPECT_FLOAT_EQ(128745.72, abs(y));

  y = -1.3;
  EXPECT_FLOAT_EQ(1.3, abs(y));

  int z = 10; // promoted to double by abs(double)
  EXPECT_FLOAT_EQ(10.0, abs(z));
}
