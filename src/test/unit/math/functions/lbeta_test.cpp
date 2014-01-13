#include "stan/math/functions/lbeta.hpp"
#include <gtest/gtest.h>

TEST(MathFunctions, lbeta) {
  using stan::math::lbeta;
  
  EXPECT_FLOAT_EQ(0.0, lbeta(1.0,1.0));
  EXPECT_FLOAT_EQ(2.981361, lbeta(0.1,0.1));
  EXPECT_FLOAT_EQ(-4.094345, lbeta(3.0,4.0));
  EXPECT_FLOAT_EQ(-4.094345, lbeta(4.0,3.0));
}
