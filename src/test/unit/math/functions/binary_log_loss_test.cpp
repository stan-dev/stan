#include "stan/math/functions/binary_log_loss.hpp"
#include <gtest/gtest.h>

TEST(MathFunctions, binary_log_loss) {
  EXPECT_FLOAT_EQ(0.0, stan::math::binary_log_loss(0,0.0));
  EXPECT_FLOAT_EQ(0.0, stan::math::binary_log_loss(1,1.0));
  EXPECT_FLOAT_EQ(-log(0.5), stan::math::binary_log_loss(0,0.5));
  EXPECT_FLOAT_EQ(-log(0.5), stan::math::binary_log_loss(1,0.5));
  EXPECT_FLOAT_EQ(-log(0.75), stan::math::binary_log_loss(0,0.25));
  EXPECT_FLOAT_EQ(-log(0.75), stan::math::binary_log_loss(1,0.75));
}

