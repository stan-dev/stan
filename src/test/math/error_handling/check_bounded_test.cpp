#include <stan/math/error_handling/check_bounded.hpp>
#include <gtest/gtest.h>

using stan::math::check_bounded;

TEST(MathErrorHandling,CheckBounded_x) {
  const char* function = "check_bounded(%1%)";
  const char* name = "x";
  double x = 0;
  double low = -1;
  double high = 1;
  double result;
 
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
  EXPECT_THROW(check_bounded(function, x, low, high, name, &result), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;

  x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(check_bounded(function, x, low, high, name, &result), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;

  x = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_bounded(function, x, low, high, name, &result), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;

  x = std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_bounded(function, x, low, high, name, &result), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;  
}
TEST(MathErrorHandling,CheckBounded_Low) {
  const char* function = "check_bounded(%1%)";
  const char* name = "x";
  double x = 0;
  double low = -1;
  double high = 1;
  double result;
 
  EXPECT_TRUE(check_bounded(function, x, low, high, name, &result))
    << "check_bounded should be true x: " << x << " and bounds: " << low << ", " << high;
  
  low = -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_bounded(function, x, low, high, name, &result))
    << "check_bounded should be TRUE with x: " << x << " and bounds: " << low << ", " << high;

  low = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(check_bounded(function, x, low, high, name, &result), 
                std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;

  low = std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_bounded(function, x, low, high, name, &result), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;
}
TEST(MathErrorHandling,CheckBounded_High) {
  const char* function = "check_bounded(%1%)";
  const char* name = "x";
  double x = 0;
  double low = -1;
  double high = 1;
  double result;
 
  EXPECT_TRUE(check_bounded(function, x, low, high, name, &result))
    << "check_bounded should be true x: " << x << " and bounds: " << low << ", " << high;

  high = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_bounded(function, x, low, high, name, &result)) 
    << "check_bounded should be TRUE with x: " << x << " and bounds: " << low << ", " << high;
  
  high = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(check_bounded(function, x, low, high, name, &result), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;

  high = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_bounded(function, x, low, high, name, &result), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;
}

