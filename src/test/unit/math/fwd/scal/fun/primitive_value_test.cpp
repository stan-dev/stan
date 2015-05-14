#include <stan/math/prim/scal/fun/primitive_value.hpp>
#include <stan/math/fwd/scal/fun/primitive_value.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(AgradFwd,primitiveValue) {
  using stan::math::fvar;
  using stan::math::primitive_value;

  fvar<double> a = 5.0;
  EXPECT_FLOAT_EQ(5.0, primitive_value(a));
  
  // make sure all work together
  EXPECT_FLOAT_EQ(5.0, primitive_value(5.0));
  EXPECT_EQ(5, primitive_value(5));
}

TEST(AgradFwd,primitiveValueNan) {
  using stan::math::fvar;
  using stan::math::primitive_value;
  double nan = std::numeric_limits<double>::quiet_NaN();

  fvar<double> a = nan;
  EXPECT_TRUE(boost::math::isnan(primitive_value(a)));
  EXPECT_TRUE(boost::math::isnan(primitive_value(nan)));
}

TEST(AgradFwd,primitiveValueNested) {
  using stan::math::fvar;
  using stan::math::primitive_value;

  fvar<fvar<double> > a = 5.0;
  EXPECT_FLOAT_EQ(5.0, primitive_value(a));
  
  // make sure all work together
  EXPECT_FLOAT_EQ(5.0, primitive_value(5.0));
  EXPECT_EQ(5, primitive_value(5));
}

TEST(AgradFwd,primitiveValueNanNested) {
  using stan::math::fvar;
  using stan::math::primitive_value;
  double nan = std::numeric_limits<double>::quiet_NaN();

  fvar<fvar<double> > a = nan;
  EXPECT_TRUE(boost::math::isnan(primitive_value(a)));
  EXPECT_TRUE(boost::math::isnan(primitive_value(nan)));
}

