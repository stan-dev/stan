#include <stan/diff/rev/value_of.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/functions/value_of.hpp>

TEST(AgradRev,value_of) {
  using stan::diff::var;
  using stan::math::value_of;
  using stan::diff::value_of;

  var a = 5.0;
  EXPECT_FLOAT_EQ(5.0, value_of(a));
  EXPECT_FLOAT_EQ(5.0, value_of(5.0)); // make sure all work together
  EXPECT_FLOAT_EQ(5.0, value_of(5));
}
