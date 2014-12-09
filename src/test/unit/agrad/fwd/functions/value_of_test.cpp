#include <stan/agrad/fwd/functions/value_of.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/functions/value_of.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

TEST(AgradFwd,value_of) {
  using stan::agrad::fvar;
  using stan::math::value_of;
  using stan::agrad::value_of;

  fvar<double> a = 5.0;
  EXPECT_FLOAT_EQ(5.0, value_of(a));
  EXPECT_FLOAT_EQ(5.0, value_of(5.0)); // make sure all work together
  EXPECT_FLOAT_EQ(5.0, value_of(5));
}

TEST(AgradFwd,value_of_nan) {
  using stan::agrad::fvar;
  using stan::math::value_of;
  using stan::agrad::value_of;
  double nan = std::numeric_limits<double>::quiet_NaN();

  fvar<double> a = nan;
  EXPECT_TRUE(boost::math::isnan(value_of(a)));
  EXPECT_TRUE(boost::math::isnan(value_of(nan)));
}
