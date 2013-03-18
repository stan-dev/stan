#include "stan/math/functions/int_step.hpp"
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
