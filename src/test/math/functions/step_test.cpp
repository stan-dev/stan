#include "stan/math/functions/step.hpp"
#include <gtest/gtest.h>

TEST(MathFunctionsStep, Double) {
  EXPECT_EQ(1, stan::math::step(3.7));
  EXPECT_EQ(1, stan::math::step(0.0));
  EXPECT_EQ(0, stan::math::step(-2.93));
}

TEST(MathFunctionsStep, Int) {
  EXPECT_EQ(1, stan::math::step(int(4)));
  EXPECT_EQ(1, stan::math::step(int(0)));
  EXPECT_EQ(0, stan::math::step(int(-3)));
}
