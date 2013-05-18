#include <stan/math/error_handling/check_not_nan.hpp>
#include <gtest/gtest.h>

using stan::math::check_not_nan;

TEST(MathErrorHandling,CheckNotNanDefaultPolicyDefaultResult) {
  const char* function = "check_not_nan(%1%)";
  double x = 0;
 
  EXPECT_TRUE(check_not_nan(function, x, "x")) 
    << "check_not_nan should be true with finite x: " << x;

  x = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_not_nan(function, x, "x")) 
    << "check_not_nan should be true with x = Inf: " << x;

  x = -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_not_nan(function, x, "x")) 
    << "check_not_nan should be true with x = -Inf: " << x;

  x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(check_not_nan(function, x, "x"), std::domain_error) 
    << "check_not_nan should throw exception on NaN: " << x;
}
