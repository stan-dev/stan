#include <stan/agrad/rev/functions/value_of.hpp>
#include <stan/math/functions/primitive_value.hpp>
#include <stan/agrad/rev/functions/primitive_value.hpp>

#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,primitiveValue) {
  using stan::agrad::var;
  using stan::math::primitive_value;

  var a = 5.0;
  EXPECT_FLOAT_EQ(5.0, primitive_value(a));

  // make sure all work together
  EXPECT_FLOAT_EQ(5.0, primitive_value(5.0)); 
  EXPECT_EQ(5, primitive_value(5));
}
