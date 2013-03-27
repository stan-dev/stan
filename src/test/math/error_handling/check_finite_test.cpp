#include <stan/math/error_handling/check_finite.hpp>
#include <gtest/gtest.h>

using stan::math::default_policy;
typedef boost::math::policies::policy<
  boost::math::policies::domain_error<boost::math::policies::errno_on_error>
  > errno_policy;

using stan::math::check_finite;

TEST(MathErrorHandling,CheckFiniteDefaultPolicy) {
  const char* function = "check_finite (%1%)";
  double x = 0;
  double result;
 
  EXPECT_TRUE (check_finite (function, x, "x", &result, default_policy())) << "check_finite should be true with finite x: " << x;
  x = std::numeric_limits<double>::infinity();
  EXPECT_THROW (check_finite (function, x, "x", &result, default_policy()), std::domain_error) << "check_finite should throw exception on Inf: " << x;
  x = -std::numeric_limits<double>::infinity();
  EXPECT_THROW (check_finite (function, x, "x", &result, default_policy()), std::domain_error) << "check_finite should throw exception on -Inf: " << x;

  x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW (check_finite (function, x, "x", &result, default_policy()), std::domain_error) << "check_finite should throw exception on NaN: " << x;
}

TEST(MathErrorHandling,CheckFiniteErrnoPolicy) {
  const char* function = "check_finite (%1%)";
  double x = 0;
  double result;
 
  EXPECT_TRUE (check_finite (function, x, "x", &result, errno_policy())) << "check_finite should be true with finite x: " << x;
  x = std::numeric_limits<double>::infinity();
  EXPECT_FALSE (check_finite (function, x, "x", &result, errno_policy())) << "check_finite should return FALSE on Inf: " << x;
  EXPECT_TRUE (std::isnan (result)) << "check_finite should have returned NaN: " << x;

  x = -std::numeric_limits<double>::infinity();
  EXPECT_FALSE (check_finite (function, x, "x", &result, errno_policy())) << "check_finite should return FALSE on -Inf: " << x;
  EXPECT_TRUE (std::isnan (result)) << "check_finite should have returned NaN: " << x;
 
  x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE (check_finite (function, x, "x", &result, errno_policy())) << "check_finite should return FALSE on NaN: " << x;
  EXPECT_TRUE (std::isnan (result)) << "check_finite should have returned NaN: " << x;
}

