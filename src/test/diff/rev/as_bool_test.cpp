#include <stan/diff/rev/as_bool.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/functions/as_bool.hpp>

TEST(DiffRev,asBool) {
  using stan::math::as_bool;
  using stan::diff::var;

  EXPECT_TRUE(as_bool(var(1)));
  EXPECT_TRUE(as_bool(var(-10L)));
  EXPECT_TRUE(as_bool(var(1.7)));
  EXPECT_TRUE(as_bool(var(-1.7)));
  EXPECT_TRUE(as_bool(var(std::numeric_limits<double>::infinity())));
  EXPECT_TRUE(as_bool(var(-std::numeric_limits<double>::infinity())));
  // don't like this behavior, but it's what C++ does
  EXPECT_TRUE(as_bool(var(std::numeric_limits<double>::quiet_NaN())));

  EXPECT_FALSE(as_bool(var(0)));
  EXPECT_FALSE(as_bool(var(0.0)));
  EXPECT_FALSE(as_bool(var(0.0f)));
}
