#include <stan/math/functions/sign.hpp>
#include <stan/agrad/rev.hpp>

#include <gtest/gtest.h>

TEST(MathFunctions, sign) {
  using stan::agrad::var;
  var x;
  x = 0;
  EXPECT_EQ(0, stan::math::sign(x));
  x = 0.0000001;
  EXPECT_EQ(1, stan::math::sign(x));
  x = -0.001;
  EXPECT_EQ(-1, stan::math::sign(x));
}
