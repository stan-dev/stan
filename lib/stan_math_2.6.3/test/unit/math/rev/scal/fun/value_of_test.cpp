#include <stan/math/rev/scal/fun/value_of.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/value_of.hpp>

TEST(AgradRev,value_of) {
  using stan::math::var;
  using stan::math::value_of;
  using stan::math::value_of;

  var a = 5.0;
  EXPECT_FLOAT_EQ(5.0, value_of(a));
  EXPECT_FLOAT_EQ(5.0, value_of(5.0)); // make sure all work together
  EXPECT_FLOAT_EQ(5.0, value_of(5));
}
