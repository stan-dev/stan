#include <stan/math/functions/primitive_value.hpp>
#include <stan/agrad/fwd/functions/primitive_value.hpp>
#include <stan/agrad/rev/functions/primitive_value.hpp>

#include <boost/math/special_functions/fpclassify.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradFwd,primitiveValue) {
  using stan::agrad::fvar;
  using stan::math::primitive_value;

  fvar<double> a = 5.0;
  EXPECT_FLOAT_EQ(5.0, primitive_value(a));
  
  // make sure all work together
  EXPECT_FLOAT_EQ(5.0, primitive_value(5.0));
  EXPECT_EQ(5, primitive_value(5));
}

TEST(AgradFwd,primitiveValueNan) {
  using stan::agrad::fvar;
  using stan::math::primitive_value;
  double nan = std::numeric_limits<double>::quiet_NaN();

  fvar<double> a = nan;
  EXPECT_TRUE(boost::math::isnan(primitive_value(a)));
  EXPECT_TRUE(boost::math::isnan(primitive_value(nan)));
}

TEST(AgradFwd,primitiveValueNested) {
  using stan::agrad::fvar;
  using stan::math::primitive_value;

  fvar<fvar<double> > a = 5.0;
  EXPECT_FLOAT_EQ(5.0, primitive_value(a));
  
  // make sure all work together
  EXPECT_FLOAT_EQ(5.0, primitive_value(5.0));
  EXPECT_EQ(5, primitive_value(5));
}

TEST(AgradFwd,primitiveValueNanNested) {
  using stan::agrad::fvar;
  using stan::math::primitive_value;
  double nan = std::numeric_limits<double>::quiet_NaN();

  fvar<fvar<double> > a = nan;
  EXPECT_TRUE(boost::math::isnan(primitive_value(a)));
  EXPECT_TRUE(boost::math::isnan(primitive_value(nan)));
}

TEST(AgradFwd,primitiveValueRevNested) {
  using stan::agrad::var;
  using stan::agrad::fvar;
  using stan::math::primitive_value;

  fvar<var> a = 5.2;
  EXPECT_FLOAT_EQ(5.2, primitive_value(a));
  
  // make sure all work together
  EXPECT_FLOAT_EQ(5.3, primitive_value(5.3));
  EXPECT_EQ(3, primitive_value(3));
}

TEST(AgradFwd,primitiveValueNanRevNested) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::primitive_value;
  double nan = std::numeric_limits<double>::quiet_NaN();

  fvar<var> a = nan;
  EXPECT_TRUE(boost::math::isnan(primitive_value(a)));
  EXPECT_TRUE(boost::math::isnan(primitive_value(nan)));
}
