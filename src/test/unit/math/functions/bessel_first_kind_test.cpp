#include <stan/math/functions/bessel_first_kind.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, bessel_first_kind) {
  using stan::math::bessel_first_kind;
  
  EXPECT_FLOAT_EQ(-0.39714980986384737228659076845169804197561868528938, 
                  bessel_first_kind(0,4.0));
  EXPECT_FLOAT_EQ(-0.33905895852593645892551459720647889697308041819800, 
                  bessel_first_kind(1,-3.0));
  EXPECT_FLOAT_EQ(0.33905895852593645892551459720647889697308041819800, 
                  bessel_first_kind(-1,-3.0));
}

TEST(MathFunctions, bessel_first_kind_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_THROW(stan::math::bessel_first_kind(3, nan), std::domain_error);
}
