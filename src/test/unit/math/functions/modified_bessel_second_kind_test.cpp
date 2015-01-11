#include <stan/math/functions/modified_bessel_second_kind.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, modified_bessel_second_kind) {
  using stan::math::modified_bessel_second_kind;
  
  EXPECT_FLOAT_EQ(0.011159676085853024269745195979833489225, 
                  modified_bessel_second_kind(0,4.0));
  EXPECT_THROW(modified_bessel_second_kind(1,-3.0), std::domain_error);
  EXPECT_THROW(modified_bessel_second_kind(-1,-3.0), std::domain_error);
}

TEST(MathFunctions, modified_bessel_second_kind_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::modified_bessel_second_kind(0, nan));
}
