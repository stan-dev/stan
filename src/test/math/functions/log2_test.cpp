#include "stan/math/functions/log2.hpp"
#include <gtest/gtest.h>

TEST(MathFunctionsLog2, NoArgs) {
  EXPECT_FLOAT_EQ(std::log(2.0), stan::math::log2());
}

TEST(MathFunctionsLog2, Double) {
  EXPECT_FLOAT_EQ(1.0, stan::math::log2(2.0));
  EXPECT_FLOAT_EQ(2.0, stan::math::log2(4.0));
  EXPECT_FLOAT_EQ(3.0, stan::math::log2(8.0));
}
