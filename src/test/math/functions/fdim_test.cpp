#include "stan/math/functions/fdim.hpp"
#include <gtest/gtest.h>

TEST(MathFunctionsFdim, Double) {
  EXPECT_FLOAT_EQ(1.0, fdim(3.0,2.0));
  EXPECT_FLOAT_EQ(0.0, fdim(2.0,3.0));

  EXPECT_FLOAT_EQ(2.5, fdim(4.5,2.0));
}

TEST(MathFunctionsFdim, Int) {
  // promotes results to double
  EXPECT_FLOAT_EQ(1.0, fdim(int(3),int(2)));
  EXPECT_FLOAT_EQ(0.0, fdim(int(2),int(3)));
}
