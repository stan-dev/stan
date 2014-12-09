#include <stan/error_handling/scalar/check_bounded.hpp>
#include <stan/agrad/rev.hpp>
#include <gtest/gtest.h>

TEST(AgradRevErrorHandlingScalar,CheckBounded_X) {
  using stan::agrad::var;
  using stan::error_handling::check_bounded;
 
  const std::string function = "check_bounded";
  const std::string name = "x";
  var x = 0;
  var low = -1;
  var high = 1;
 
  EXPECT_TRUE(check_bounded(function, name, x, low, high)) 
    << "check_bounded should be TRUE with x: " << x << " and bounds: " << low << ", " << high;
  
  x = low;
  EXPECT_TRUE(check_bounded(function, name, x, low, high)) 
    << "check_bounded should be TRUE with x: " << x << " equal to the lower bound: " << low;

  x = high;
  EXPECT_TRUE(check_bounded(function, name, x, low, high)) 
    << "check_bounded should be TRUE with x: " << x << " equal to the lower bound: " << low;

  x = low-1;
  EXPECT_THROW(check_bounded(function, name, x, low, high), 
                std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;  
  
  x = high+1;
  EXPECT_THROW(check_bounded(function, name, x, low, high), 
                std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;

  x = std::numeric_limits<var>::quiet_NaN();
  EXPECT_THROW(check_bounded(function, name, x, low, high), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;

  x = -std::numeric_limits<var>::infinity();
  EXPECT_THROW(check_bounded(function, name, x, low, high), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;

  x = std::numeric_limits<var>::infinity();
  EXPECT_THROW(check_bounded(function, name, x, low, high), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;
}

TEST(AgradRevErrorHandlingScalar,CheckBounded_Low) {
  using stan::agrad::var;
  using stan::error_handling::check_bounded;

  const std::string function = "check_bounded";
  const std::string name = "x";
  var x = 0;
  var low = -1;
  var high = 1;
 
  EXPECT_TRUE(check_bounded(function, name, x, low, high)) 
    << "check_bounded should be true x: " << x << " and bounds: " << low << ", " << high;
  
  low = -std::numeric_limits<var>::infinity();
  EXPECT_TRUE(check_bounded(function, name, x, low, high)) 
    << "check_bounded should be TRUE with x: " << x << " and bounds: " << low << ", " << high;

  low = std::numeric_limits<var>::quiet_NaN();
  EXPECT_THROW(check_bounded(function, name, x, low, high), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;

  low = std::numeric_limits<var>::infinity();
  EXPECT_THROW(check_bounded(function, name, x, low, high), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;
}
TEST(AgradRevErrorHandlingScalar,CheckBounded_High) {
  using stan::agrad::var;
  using stan::error_handling::check_bounded;

  const std::string function = "check_bounded";
  const std::string name = "x";
  var x = 0;
  var low = -1;
  var high = 1;
 
  EXPECT_TRUE(check_bounded(function, name, x, low, high)) 
    << "check_bounded should be true x: " << x << " and bounds: " << low << ", " << high;

  high = std::numeric_limits<var>::infinity();
  EXPECT_TRUE(check_bounded(function, name, x, low, high)) 
    << "check_bounded should be TRUE with x: " << x << " and bounds: " << low << ", " << high;
  
  high = std::numeric_limits<var>::quiet_NaN();
  EXPECT_THROW(check_bounded(function, name, x, low, high), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;

  high = -std::numeric_limits<var>::infinity();
  EXPECT_THROW(check_bounded(function, name, x, low, high), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;
}
