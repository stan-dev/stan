#include <stan/math/prim/scal/fun/primitive_value.hpp>
#include <stan/math/fwd/scal/fun/primitive_value.hpp>
#include <stan/math/rev/scal/fun/primitive_value.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>

TEST(AgradFwd,primitiveValueRevNested) {
  using stan::math::var;
  using stan::math::fvar;
  using stan::math::primitive_value;

  fvar<var> a = 5.2;
  EXPECT_FLOAT_EQ(5.2, primitive_value(a));
  
  // make sure all work together
  EXPECT_FLOAT_EQ(5.3, primitive_value(5.3));
  EXPECT_EQ(3, primitive_value(3));
}

TEST(AgradFwd,primitiveValueNanRevNested) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::primitive_value;
  double nan = std::numeric_limits<double>::quiet_NaN();

  fvar<var> a = nan;
  EXPECT_TRUE(boost::math::isnan(primitive_value(a)));
  EXPECT_TRUE(boost::math::isnan(primitive_value(nan)));
}
