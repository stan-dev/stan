#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/agrad/rev.hpp>
#include <gtest/gtest.h>

TEST(AgradRevErrorHandlingScalar,CheckNotNan) {
  using stan::agrad::var;
  using stan::error_handling::check_not_nan;
  const std::string function = "check_not_nan";
  var x = 0;
  double x_d = 0;
 
  EXPECT_TRUE(check_not_nan(function, "x", x))
    << "check_not_nan should be true with finite x: " << x;
  EXPECT_TRUE(check_not_nan(function, "x", x_d))
    << "check_not_nan should be true with finite x: " << x_d;
  
  x = std::numeric_limits<var>::infinity();
  x_d = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_not_nan(function, "x", x))
    << "check_not_nan should be true with x = Inf: " << x;
  EXPECT_TRUE(check_not_nan(function, "x", x_d))
    << "check_not_nan should be true with x = Inf: " << x_d;

  x = -std::numeric_limits<var>::infinity();
  x_d = -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_not_nan(function, "x", x))
    << "check_not_nan should be true with x = -Inf: " << x;
  EXPECT_TRUE(check_not_nan(function, "x", x_d))
    << "check_not_nan should be true with x = -Inf: " << x_d;

  x = std::numeric_limits<var>::quiet_NaN();
  x_d = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(check_not_nan(function, "x", x), std::domain_error)
    << "check_not_nan should throw exception on NaN: " << x;
  EXPECT_THROW(check_not_nan(function, "x", x_d), std::domain_error)
    << "check_not_nan should throw exception on NaN: " << x_d;
}

