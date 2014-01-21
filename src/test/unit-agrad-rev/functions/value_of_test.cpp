#include <stan/agrad/rev/functions/value_of.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/functions/value_of.hpp>

TEST(AgradRev,value_of) {
  using stan::agrad::var;
  using stan::math::value_of;
  using stan::agrad::value_of;

  var a = 5.0;
  EXPECT_FLOAT_EQ(5.0, value_of(a));
  EXPECT_FLOAT_EQ(5.0, value_of(5.0)); // make sure all work together
  EXPECT_FLOAT_EQ(5.0, value_of(5));
}
