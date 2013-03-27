#include <stan/math/error_handling/check_not_nan.hpp>
#include <gtest/gtest.h>

typedef boost::math::policies::policy<
  boost::math::policies::domain_error<boost::math::policies::errno_on_error>
  > errno_policy;
using stan::math::check_not_nan;
using stan::math::default_policy;

TEST(MathErrorHandling,CheckNotNanDefaultPolicy) {
  const char* function = "check_not_nan(%1%)";
  double x = 0;
  double result;
 
  EXPECT_TRUE(check_not_nan(function, x, "x", &result, default_policy())) << "check_not_nan should be true with finite x: " << x;
  x = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_not_nan(function, x, "x", &result, default_policy())) << "check_not_nan should be true with x = Inf: " << x;
  x = -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_not_nan(function, x, "x", &result, default_policy())) << "check_not_nan should be true with x = -Inf: " << x;

  x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(check_not_nan(function, x, "x", &result, default_policy()), std::domain_error) << "check_not_nan should throw exception on NaN: " << x;
}

TEST(MathErrorHandling,CheckNotNanErrnoPolicy) {
  const char* function = "check_not_nan(%1%)";
  double x = 0;
  double result = 0;

  EXPECT_TRUE(check_not_nan(function, x, "x", &result, errno_policy())) << "check_not_nan should be true with finite x: " << x;
  x = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_not_nan(function, x, "x", &result, errno_policy())) << "check_not_nan should be true with x = Inf: " << x;
  EXPECT_FALSE(std::isnan (result)) << "check_not_nan should not have returned nan: " << x;

  x = -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_not_nan (function, x, "x", &result, errno_policy())) << "check_not_nan should be true with x = -Inf: " << x;
  EXPECT_FALSE(std::isnan (result)) << "check_not_nan should have returned nan: " << x;
 
  x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(check_not_nan (function, x, "x", &result, errno_policy())) << "check_not_nan should return FALSE on nan: " << x;
  EXPECT_TRUE(std::isnan (result)) << "check_not_nan should have returned nan: " << x;
}

TEST(MathErrorHandling,CheckNotNanDefaultPolicyDefaultResult) {
  const char* function = "check_not_nan(%1%)";
  double x = 0;
 
  EXPECT_TRUE(check_not_nan(function, x, "x")) << "check_not_nan should be true with finite x: " << x;
  x = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_not_nan(function, x, "x")) << "check_not_nan should be true with x = Inf: " << x;
  x = -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_not_nan(function, x, "x")) << "check_not_nan should be true with x = -Inf: " << x;

  x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(check_not_nan(function, x, "x"), std::domain_error) << "check_not_nan should throw exception on NaN: " << x;
}
