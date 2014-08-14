#include <stan/agrad/rev/functions/value_of.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/functions/value_of.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

TEST(AgradRev,value_of) {
  using stan::agrad::var;
  using stan::math::value_of;
  using stan::agrad::value_of;

  var a = 5.0;
  EXPECT_FLOAT_EQ(5.0, value_of(a));
  EXPECT_FLOAT_EQ(5.0, value_of(5.0)); // make sure all work together
  EXPECT_FLOAT_EQ(5.0, value_of(5));
}

TEST(AgradRev,value_of_nan) {
  stan::agrad::var nan = std::numeric_limits<double>::quiet_NaN();

  EXPECT_PRED1(boost::math::isnan<double>,
               value_of(nan));
}
