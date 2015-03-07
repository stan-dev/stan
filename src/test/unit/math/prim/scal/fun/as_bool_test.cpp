#include <stan/math/prim/scal/fun/as_bool.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, as_bool) {
  using stan::math::as_bool;
  EXPECT_TRUE(as_bool(1));
  EXPECT_TRUE(as_bool(1.7));
  EXPECT_TRUE(as_bool(-1.7));
  EXPECT_TRUE(as_bool(std::numeric_limits<double>::infinity()));
  EXPECT_TRUE(as_bool(-std::numeric_limits<double>::infinity()));

  EXPECT_FALSE(as_bool(0));
  EXPECT_FALSE(as_bool(0.0));
  EXPECT_FALSE(as_bool(0.0f));

  EXPECT_TRUE(as_bool(10));
  EXPECT_TRUE(as_bool(-1));
  EXPECT_FALSE(as_bool(0));
}

TEST(MathFunctions, as_bool_nan) {
  // don't like this behavior, but it's what C++ does
  EXPECT_TRUE(stan::math::as_bool(std::numeric_limits<double>::quiet_NaN()));
}
