#include "stan/math/functions/int_step.hpp"
#include <gtest/gtest.h>

TEST(MathFunctionsIntStep, Double) {
  EXPECT_EQ(0U, stan::math::int_step(-1.0));
  EXPECT_EQ(0U, stan::math::int_step(0.0));
  EXPECT_EQ(1U, stan::math::int_step(0.00000000001));
  EXPECT_EQ(1U, stan::math::int_step(100.0));
}

TEST(MathFunctionsIntStep, Int) {
  EXPECT_EQ(0U, stan::math::int_step(int(-1)));
  EXPECT_EQ(0U, stan::math::int_step(int(0)));
  EXPECT_EQ(1U, stan::math::int_step(int(100)));
}
