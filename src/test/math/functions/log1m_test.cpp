#include "stan/math/functions/log1m.hpp"
#include <gtest/gtest.h>

TEST(MathsSpecialFunctions, log1m) {
  EXPECT_FLOAT_EQ(stan::math::log1p(-0.1),stan::math::log1m(0.1));
}
