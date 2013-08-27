#include <stan/math/error_handling/check_bounded.hpp>
#include <stan/diff.hpp>
#include <gtest/gtest.h>

TEST(DiffErrorHandling,CheckBounded_X) {
  using stan::diff::var;
  using stan::math::check_bounded;
 
  const char* function = "check_bounded(%1%)";
  const char* name = "x";
  var x = 0;
  var low = -1;
  var high = 1;
  var result = 0;
 
  EXPECT_TRUE(check_bounded(function, x, low, high, name, &result)) 
    << "check_bounded should be TRUE with x: " << x << " and bounds: " << low << ", " << high;
  
  x = low;
  EXPECT_TRUE(check_bounded(function, x, low, high, name, &result)) 
    << "check_bounded should be TRUE with x: " << x << " equal to the lower bound: " << low;

  x = high;
  EXPECT_TRUE(check_bounded(function, x, low, high, name, &result)) 
    << "check_bounded should be TRUE with x: " << x << " equal to the lower bound: " << low;

  x = low-1;
  EXPECT_THROW(check_bounded(function, x, low, high, name, &result), 
                std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;  
  
  x = high+1;
  EXPECT_THROW(check_bounded(function, x, low, high, name, &result), 
                std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;

  x = std::numeric_limits<var>::quiet_NaN();
  EXPECT_THROW(check_bounded(function, x, low, high, name, &result), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;

  x = -std::numeric_limits<var>::infinity();
  EXPECT_THROW(check_bounded(function, x, low, high, name, &result), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;

  x = std::numeric_limits<var>::infinity();
  EXPECT_THROW(check_bounded(function, x, low, high, name, &result), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;
}

TEST(DiffErrorHandling,CheckBounded_Low) {
  using stan::diff::var;
  using stan::math::check_bounded;

  const char* function = "check_bounded(%1%)";
  const char* name = "x";
  var x = 0;
  var low = -1;
  var high = 1;
  var result = 0;
 
  EXPECT_TRUE(check_bounded(function, x, low, high, name, &result)) 
    << "check_bounded should be true x: " << x << " and bounds: " << low << ", " << high;
  
  low = -std::numeric_limits<var>::infinity();
  EXPECT_TRUE(check_bounded(function, x, low, high, name, &result)) 
    << "check_bounded should be TRUE with x: " << x << " and bounds: " << low << ", " << high;

  low = std::numeric_limits<var>::quiet_NaN();
  EXPECT_THROW(check_bounded(function, x, low, high, name, &result), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;

  low = std::numeric_limits<var>::infinity();
  EXPECT_THROW(check_bounded(function, x, low, high, name, &result), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;
}
TEST(DiffErrorHandling,CheckBounded_High) {
  using stan::diff::var;
  using stan::math::check_bounded;

  const char* function = "check_bounded(%1%)";
  const char* name = "x";
  var x = 0;
  var low = -1;
  var high = 1;
  var result = 0;
 
  EXPECT_TRUE(check_bounded(function, x, low, high, name, &result)) 
    << "check_bounded should be true x: " << x << " and bounds: " << low << ", " << high;

  high = std::numeric_limits<var>::infinity();
  EXPECT_TRUE(check_bounded(function, x, low, high, name, &result)) 
    << "check_bounded should be TRUE with x: " << x << " and bounds: " << low << ", " << high;
  
  high = std::numeric_limits<var>::quiet_NaN();
  EXPECT_THROW(check_bounded(function, x, low, high, name, &result), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;

  high = -std::numeric_limits<var>::infinity();
  EXPECT_THROW(check_bounded(function, x, low, high, name, &result), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;
}
