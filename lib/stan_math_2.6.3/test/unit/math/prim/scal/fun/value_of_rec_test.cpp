#include <stan/math/prim/scal/fun/value_of_rec.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, value_of_rec) {
  using stan::math::value_of_rec;
  double x = 5.0;
  EXPECT_FLOAT_EQ(5.0,value_of_rec(x));
  EXPECT_FLOAT_EQ(5.0,value_of_rec(5));
}

