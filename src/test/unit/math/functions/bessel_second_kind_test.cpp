#include <stan/math/functions/bessel_second_kind.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, bessel_second_kind) {
  using stan::math::bessel_second_kind;
  
  EXPECT_FLOAT_EQ(-0.01694073932506499190363513444715321824049258989801, 
                  bessel_second_kind(0,4.0));
  EXPECT_FLOAT_EQ(0.3246744247917999784370128392879532396692751433723549, 
                  bessel_second_kind(1,3.0));
  EXPECT_THROW(bessel_second_kind(-1,-3.0), std::domain_error);
}

TEST(MathFunctions, bessel_second_kind_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::bessel_second_kind(1, nan));
}
