#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/primitive_value.hpp>
#include <stan/math/rev/scal/fun/primitive_value.hpp>

#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,primitiveValue) {
  using stan::math::var;
  using stan::math::primitive_value;

  var a = 5.0;
  EXPECT_FLOAT_EQ(5.0, primitive_value(a));

  // make sure all work together
  EXPECT_FLOAT_EQ(5.0, primitive_value(5.0)); 
  EXPECT_EQ(5, primitive_value(5));
}
