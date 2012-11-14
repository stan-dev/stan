#include <gtest/gtest.h>
#include <stan/agrad/error_handling.hpp>

typedef boost::math::policies::policy<
  boost::math::policies::domain_error<boost::math::policies::errno_on_error>
  > errno_policy;

//---------- check_not_nan tests ----------
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

// ---------- check_bounded tests ----------
TEST(AgradErrorHandling,CheckBoundedDefaultPolicyX) {
  using stan::agrad::var;
  using stan::math::default_policy;
  using stan::math::check_bounded;
 
  const char* function = "check_bounded (%1%)";
  const char* name = "x";
  var x = 0;
  var low = -1;
  var high = 1;
  var result = 0;
 
  EXPECT_TRUE (check_bounded (function, x, low, high, name, &result, default_policy())) << "check_bounded should be TRUE with x: " << x << " and bounds: " << low << ", " << high;
  
  x = low;
  EXPECT_TRUE (check_bounded (function, x, low, high, name, &result, default_policy())) 
    << "check_bounded should be TRUE with x: " << x << " equal to the lower bound: " << low;

  x = high;
  EXPECT_TRUE (check_bounded (function, x, low, high, name, &result, default_policy())) 
    << "check_bounded should be TRUE with x: " << x << " equal to the lower bound: " << low;

  x = low-1;
  EXPECT_THROW (check_bounded (function, x, low, high, name, &result, default_policy()), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;  
  
  x = high+1;
  EXPECT_THROW (check_bounded (function, x, low, high, name, &result, default_policy()), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;

  x = std::numeric_limits<var>::quiet_NaN();
  EXPECT_THROW (check_bounded (function, x, low, high, name, &result, default_policy()), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;

  x = -std::numeric_limits<var>::infinity();
  EXPECT_THROW (check_bounded (function, x, low, high, name, &result, default_policy()), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;

  x = std::numeric_limits<var>::infinity();
  EXPECT_THROW (check_bounded (function, x, low, high, name, &result, default_policy()), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;
  
}
TEST(AgradErrorHandling,CheckBoundedDefaultPolicyLow) {
  using stan::agrad::var;
  using stan::math::default_policy;
  using stan::math::check_bounded;

  const char* function = "check_bounded (%1%)";
  const char* name = "x";
  var x = 0;
  var low = -1;
  var high = 1;
  var result = 0;
 
  EXPECT_TRUE (check_bounded (function, x, low, high, name, &result, default_policy())) << "check_bounded should be true x: " << x << " and bounds: " << low << ", " << high;
  
  low = -std::numeric_limits<var>::infinity();
  EXPECT_TRUE (check_bounded (function, x, low, high, name, &result, default_policy())) 
    << "check_bounded should be TRUE with x: " << x << " and bounds: " << low << ", " << high;

  low = std::numeric_limits<var>::quiet_NaN();
  EXPECT_THROW (check_bounded (function, x, low, high, name, &result, default_policy()), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;

  low = std::numeric_limits<var>::infinity();
  EXPECT_THROW (check_bounded (function, x, low, high, name, &result, default_policy()), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;
}
TEST(AgradErrorHandling,CheckBoundedDefaultPolicyHigh) {
  using stan::agrad::var;
  using stan::math::default_policy;
  using stan::math::check_bounded;

  const char* function = "check_bounded (%1%)";
  const char* name = "x";
  var x = 0;
  var low = -1;
  var high = 1;
  var result = 0;
 
  EXPECT_TRUE (check_bounded (function, x, low, high, name, &result, default_policy())) << "check_bounded should be true x: " << x << " and bounds: " << low << ", " << high;

  high = std::numeric_limits<var>::infinity();
  EXPECT_TRUE (check_bounded (function, x, low, high, name, &result, default_policy())) 
    << "check_bounded should be TRUE with x: " << x << " and bounds: " << low << ", " << high;
  
  high = std::numeric_limits<var>::quiet_NaN();
  EXPECT_THROW (check_bounded (function, x, low, high, name, &result, default_policy()), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;

  high = -std::numeric_limits<var>::infinity();
  EXPECT_THROW (check_bounded (function, x, low, high, name, &result, default_policy()), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;

}


TEST(AgradErrorHandling,CheckBoundedErrnoPolicyX) {
  using stan::agrad::var;
  using stan::math::check_bounded;
 
  const char* function = "check_bounded (%1%)";
  const char* name = "x";
  var x = 0;
  var low = -1;
  var high = 1;
  var result = 0;
 
  result = 0;
  EXPECT_TRUE (check_bounded (function, x, low, high, name, &result, errno_policy())) << "check_bounded should be TRUE with x: " << x << " and bounds: " << low << ", " << high;

  result = 0;
  x = low;
  EXPECT_TRUE (check_bounded (function, x, low, high, name, &result, errno_policy())) 
    << "check_bounded should be TRUE with x: " << x << " equal to the lower bound: " << low;

  result = 0;
  x = high;
  EXPECT_TRUE (check_bounded (function, x, low, high, name, &result, errno_policy())) 
    << "check_bounded should be TRUE with x: " << x << " equal to the lower bound: " << low;

  result = 0;
  x = low-1;
  EXPECT_FALSE (check_bounded (function, x, low, high, name, &result, errno_policy()))
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;  
  EXPECT_TRUE (std::isnan (result)) << "check_bounded should set return value to NaN: " << result;


  result = 0;
  x = high+1;
  EXPECT_FALSE (check_bounded (function, x, low, high, name, &result, errno_policy()))
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;
  EXPECT_TRUE (std::isnan (result)) << "check_bounded should set return value to NaN: " << result;

  result = 0;
  x = std::numeric_limits<var>::quiet_NaN();
  EXPECT_FALSE (check_bounded (function, x, low, high, name, &result, errno_policy()))
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;

  result = 0;
  x = -std::numeric_limits<var>::infinity();
  EXPECT_FALSE (check_bounded (function, x, low, high, name, &result, errno_policy()))
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;
  EXPECT_TRUE (std::isnan (result)) << "check_bounded should set return value to NaN: " << result;

  result = 0;
  x = std::numeric_limits<var>::infinity();
  EXPECT_FALSE (check_bounded (function, x, low, high, name, &result, errno_policy()))
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;
  EXPECT_TRUE (std::isnan (result)) << "check_bounded should set return value to NaN: " << result;
}
TEST(AgradErrorHandling,CheckBoundedErrnoPolicyLow) {
  using stan::agrad::var;
  using stan::math::check_bounded;
 
  const char* function = "check_bounded (%1%)";
  const char* name = "x";
  var x = 0;
  var low = -1;
  var high = 1;
  var result = 0;

  result = 0; 
  EXPECT_TRUE (check_bounded (function, x, low, high, name, &result, errno_policy())) << "check_bounded should be true x: " << x << " and bounds: " << low << ", " << high;

  result = 0;
  low = -std::numeric_limits<var>::infinity();
  EXPECT_TRUE (check_bounded (function, x, low, high, name, &result, errno_policy())) 
    << "check_bounded should be TRUE with x: " << x << " and bounds: " << low << ", " << high;

  result = 0;
  low = std::numeric_limits<var>::quiet_NaN();
  EXPECT_FALSE (check_bounded (function, x, low, high, name, &result, errno_policy())) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;
  EXPECT_TRUE (std::isnan (result)) << "check_bounded should set return value to NaN: " << result;
 
  result = 0;
  low = std::numeric_limits<var>::infinity();
  EXPECT_FALSE (check_bounded (function, x, low, high, name, &result, errno_policy())) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;
  EXPECT_TRUE (std::isnan (result)) << "check_bounded should set return value to NaN: " << result;
}
TEST(AgradErrorHandling,CheckBoundedErrnoPolicyHigh) {
  using stan::agrad::var;
  using stan::math::check_bounded;
 
  const char* function = "check_bounded (%1%)";
  const char* name = "x";
  var x = 0;
  var low = -1;
  var high = 1;
  var result = 0;

  result = 0; 
  EXPECT_TRUE (check_bounded (function, x, low, high, name, &result, errno_policy())) << "check_bounded should be true x: " << x << " and bounds: " << low << ", " << high;

  result = 0;
  high = std::numeric_limits<var>::infinity();
  EXPECT_TRUE (check_bounded (function, x, low, high, name, &result, errno_policy())) 
    << "check_bounded should be TRUE with x: " << x << " and bounds: " << low << ", " << high;
  
  result = 0;
  high = std::numeric_limits<var>::quiet_NaN();
  EXPECT_FALSE (check_bounded (function, x, low, high, name, &result, errno_policy())) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;
  EXPECT_TRUE (std::isnan (result)) << "check_bounded should set return value to NaN: " << result;

  result = 0;
  high = -std::numeric_limits<var>::infinity();
  EXPECT_FALSE (check_bounded (function, x, low, high, name, &result, errno_policy())) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;
  EXPECT_TRUE (std::isnan (result)) << "check_bounded should set return value to NaN: " << result;
}
