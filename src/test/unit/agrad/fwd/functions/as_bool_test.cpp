#include <gtest/gtest.h>
#include <stan/math/functions/as_bool.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/fwd/operators/operator_not_equal.hpp>

TEST(AgradFwd,asBool) {
  using stan::math::as_bool;
  using stan::agrad::fvar;

  EXPECT_TRUE(as_bool(fvar<double>(1)));
  EXPECT_TRUE(as_bool(fvar<double>(-10L)));
  EXPECT_TRUE(as_bool(fvar<double>(1.7)));
  EXPECT_TRUE(as_bool(fvar<double>(-1.7)));
  EXPECT_TRUE(as_bool(fvar<double>(std::numeric_limits<double>::infinity())));
  EXPECT_TRUE(as_bool(fvar<double>(-std::numeric_limits<double>::infinity())));
  // don't like this behavior, but it's what C++ does
  EXPECT_TRUE(as_bool(fvar<double>(std::numeric_limits<double>::quiet_NaN())));

  EXPECT_FALSE(as_bool(fvar<double>(0)));
  EXPECT_FALSE(as_bool(fvar<double>(0.0)));
  EXPECT_FALSE(as_bool(fvar<double>(0.0f)));
}
TEST(AgradFwd,as_bool_nan) {
  stan::agrad::fvar<double> nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(stan::math::as_bool(nan));
}
