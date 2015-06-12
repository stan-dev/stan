#include <stan/math/prim/scal/fun/int_step.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, int_step_double) {
  using stan::math::int_step;
  EXPECT_EQ(0U, int_step(-1.0));
  EXPECT_EQ(0U, int_step(0.0));
  EXPECT_EQ(1U, int_step(0.00000000001));
  EXPECT_EQ(1U, int_step(100.0));
}

TEST(MathFunctions, int_step_int) {
  using stan::math::int_step;

  EXPECT_EQ(0U, int_step(int(-1)));
  EXPECT_EQ(0U, int_step(int(0)));
  EXPECT_EQ(1U, int_step(int(100)));
}

TEST(MathFunctions, int_step_inf) {
  using stan::math::int_step;

  EXPECT_EQ(1U, int_step(std::numeric_limits<double>::infinity()));
  EXPECT_EQ(0U, int_step(-std::numeric_limits<double>::infinity()));
}

TEST(MathFunctions, int_step_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_EQ(0U, stan::math::int_step(nan));
}
