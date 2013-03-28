#include <stan/math/error_handling.hpp>
#include <stan/agrad.hpp>
#include <gtest/gtest.h>

typedef boost::math::policies::policy<
  boost::math::policies::domain_error<boost::math::policies::errno_on_error>
  > errno_policy;

TEST(AgradErrorHandling,CheckNotNanDefaultPolicy) {
  using stan::agrad::var;
  using stan::math::default_policy;
  using stan::math::check_not_nan;
  const char* function = "check_not_nan(%1%)";
  var x = 0;
  double x_d = 0;
  var result = 0;
 
  EXPECT_TRUE(check_not_nan(function, x, "x", &result, default_policy())) << "check_not_nan should be true with finite x: " << x;
  EXPECT_TRUE(check_not_nan(function, x_d, "x", &result, default_policy())) << "check_not_nan should be true with finite x: " << x_d;
  
  x = std::numeric_limits<var>::infinity();
  x_d = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_not_nan(function, x, "x", &result, default_policy())) << "check_not_nan should be true with x = Inf: " << x;
  EXPECT_TRUE(check_not_nan(function, x_d, "x", &result, default_policy())) << "check_not_nan should be true with x = Inf: " << x_d;

  x = -std::numeric_limits<var>::infinity();
  x_d = -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_not_nan(function, x, "x", &result, default_policy())) << "check_not_nan should be true with x = -Inf: " << x;
  EXPECT_TRUE(check_not_nan(function, x_d, "x", &result, default_policy())) << "check_not_nan should be true with x = -Inf: " << x_d;

  x = std::numeric_limits<var>::quiet_NaN();
  x_d = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(check_not_nan(function, x, "x", &result, default_policy()), std::domain_error) << "check_not_nan should throw exception on NaN: " << x;
  EXPECT_THROW(check_not_nan(function, x_d, "x", &result, default_policy()), std::domain_error) << "check_not_nan should throw exception on NaN: " << x_d;
}

TEST(AgradErrorHandling,CheckNotNanErrnoPolicy) {
  using stan::agrad::var;
  using stan::math::check_not_nan;
  const char* function = "check_not_nan(%1%)";
  var x = 0;
  double x_d = 0;
  var result = 0;
 
  EXPECT_TRUE(check_not_nan(function, x, "x", &result, errno_policy())) << "check_not_nan should be true with finite x: " << x;
  EXPECT_TRUE(check_not_nan(function, x_d, "x", &result, errno_policy())) << "check_not_nan should be true with finite x: " << x;
  
  x = std::numeric_limits<var>::infinity();
  EXPECT_TRUE(check_not_nan(function, x, "x", &result, errno_policy())) << "check_not_nan should be true with x = Inf: " << x;
  EXPECT_FALSE(std::isnan (result)) << "check_not_nan should not have returned nan: " << x;
  x_d = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_not_nan(function, x_d, "x", &result, errno_policy())) << "check_not_nan should be true with x = Inf: " << x_d;
  EXPECT_FALSE(std::isnan (result)) << "check_not_nan should not have returned nan: " << x_d;

  x = -std::numeric_limits<var>::infinity();
  EXPECT_TRUE(check_not_nan (function, x, "x", &result, errno_policy())) << "check_not_nan should be true with x = -Inf: " << x;
  EXPECT_FALSE(std::isnan (result)) << "check_not_nan should have returned nan: " << x;
  x_d = -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_not_nan (function, x_d, "x", &result, errno_policy())) << "check_not_nan should be true with x = -Inf: " << x_d;
  EXPECT_FALSE(std::isnan (result)) << "check_not_nan should have returned nan: " << x_d;
 
  x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(check_not_nan (function, x, "x", &result, errno_policy())) << "check_not_nan should return FALSE on nan: " << x;
  EXPECT_TRUE(std::isnan (result)) << "check_not_nan should have returned nan: " << x;
  x_d = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(check_not_nan (function, x_d, "x", &result, errno_policy())) << "check_not_nan should return FALSE on nan: " << x_d;
  EXPECT_TRUE(std::isnan (result)) << "check_not_nan should have returned nan: " << x_d;
}
