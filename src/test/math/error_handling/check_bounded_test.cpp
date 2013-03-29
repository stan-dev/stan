#include <stan/math/error_handling/check_bounded.hpp>
#include <gtest/gtest.h>

using stan::math::default_policy;
typedef boost::math::policies::policy<
  boost::math::policies::domain_error<boost::math::policies::errno_on_error>
  > errno_policy;

using stan::math::check_bounded;

TEST(MathErrorHandling,CheckBoundedDefaultPolicyX) {
  const char* function = "check_bounded (%1%)";
  const char* name = "x";
  double x = 0;
  double low = -1;
  double high = 1;
  double result;
 
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

  x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW (check_bounded (function, x, low, high, name, &result, default_policy()), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;

  x = -std::numeric_limits<double>::infinity();
  EXPECT_THROW (check_bounded (function, x, low, high, name, &result, default_policy()), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;

  x = std::numeric_limits<double>::infinity();
  EXPECT_THROW (check_bounded (function, x, low, high, name, &result, default_policy()), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;
  
}
TEST(MathErrorHandling,CheckBoundedDefaultPolicyLow) {
  const char* function = "check_bounded (%1%)";
  const char* name = "x";
  double x = 0;
  double low = -1;
  double high = 1;
  double result;
 
  EXPECT_TRUE (check_bounded (function, x, low, high, name, &result, default_policy())) << "check_bounded should be true x: " << x << " and bounds: " << low << ", " << high;
  
  low = -std::numeric_limits<double>::infinity();
  EXPECT_TRUE (check_bounded (function, x, low, high, name, &result, default_policy())) 
    << "check_bounded should be TRUE with x: " << x << " and bounds: " << low << ", " << high;

  low = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW (check_bounded (function, x, low, high, name, &result, default_policy()), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;

  low = std::numeric_limits<double>::infinity();
  EXPECT_THROW (check_bounded (function, x, low, high, name, &result, default_policy()), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;
}
TEST(MathErrorHandling,CheckBoundedDefaultPolicyHigh) {
  const char* function = "check_bounded (%1%)";
  const char* name = "x";
  double x = 0;
  double low = -1;
  double high = 1;
  double result;
 
  EXPECT_TRUE (check_bounded (function, x, low, high, name, &result, default_policy())) << "check_bounded should be true x: " << x << " and bounds: " << low << ", " << high;

  high = std::numeric_limits<double>::infinity();
  EXPECT_TRUE (check_bounded (function, x, low, high, name, &result, default_policy())) 
    << "check_bounded should be TRUE with x: " << x << " and bounds: " << low << ", " << high;
  
  high = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW (check_bounded (function, x, low, high, name, &result, default_policy()), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;

  high = -std::numeric_limits<double>::infinity();
  EXPECT_THROW (check_bounded (function, x, low, high, name, &result, default_policy()), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;

}


TEST(MathErrorHandling,CheckBoundedErrnoPolicyX) {
  const char* function = "check_bounded (%1%)";
  const char* name = "x";
  double x = 0;
  double low = -1;
  double high = 1;
  double result;
 
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
  x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE (check_bounded (function, x, low, high, name, &result, errno_policy()))
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;

  result = 0;
  x = -std::numeric_limits<double>::infinity();
  EXPECT_FALSE (check_bounded (function, x, low, high, name, &result, errno_policy()))
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;
  EXPECT_TRUE (std::isnan (result)) << "check_bounded should set return value to NaN: " << result;

  result = 0;
  x = std::numeric_limits<double>::infinity();
  EXPECT_FALSE (check_bounded (function, x, low, high, name, &result, errno_policy()))
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;
  EXPECT_TRUE (std::isnan (result)) << "check_bounded should set return value to NaN: " << result;
}
TEST(MathErrorHandling,CheckBoundedErrnoPolicyLow) {
  const char* function = "check_bounded (%1%)";
  const char* name = "x";
  double x = 0;
  double low = -1;
  double high = 1;
  double result;

  result = 0; 
  EXPECT_TRUE (check_bounded (function, x, low, high, name, &result, errno_policy())) << "check_bounded should be true x: " << x << " and bounds: " << low << ", " << high;

  result = 0;
  low = -std::numeric_limits<double>::infinity();
  EXPECT_TRUE (check_bounded (function, x, low, high, name, &result, errno_policy())) 
    << "check_bounded should be TRUE with x: " << x << " and bounds: " << low << ", " << high;

  result = 0;
  low = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE (check_bounded (function, x, low, high, name, &result, errno_policy())) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;
  EXPECT_TRUE (std::isnan (result)) << "check_bounded should set return value to NaN: " << result;
 
  result = 0;
  low = std::numeric_limits<double>::infinity();
  EXPECT_FALSE (check_bounded (function, x, low, high, name, &result, errno_policy())) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;
  EXPECT_TRUE (std::isnan (result)) << "check_bounded should set return value to NaN: " << result;
}
TEST(MathErrorHandling,CheckBoundedErrnoPolicyHigh) {
  const char* function = "check_bounded (%1%)";
  const char* name = "x";
  double x = 0;
  double low = -1;
  double high = 1;
  double result;

  result = 0; 
  EXPECT_TRUE (check_bounded (function, x, low, high, name, &result, errno_policy())) << "check_bounded should be true x: " << x << " and bounds: " << low << ", " << high;

  result = 0;
  high = std::numeric_limits<double>::infinity();
  EXPECT_TRUE (check_bounded (function, x, low, high, name, &result, errno_policy())) 
    << "check_bounded should be TRUE with x: " << x << " and bounds: " << low << ", " << high;
  
  result = 0;
  high = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE (check_bounded (function, x, low, high, name, &result, errno_policy())) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;
  EXPECT_TRUE (std::isnan (result)) << "check_bounded should set return value to NaN: " << result;

  result = 0;
  high = -std::numeric_limits<double>::infinity();
  EXPECT_FALSE (check_bounded (function, x, low, high, name, &result, errno_policy())) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;
  EXPECT_TRUE (std::isnan (result)) << "check_bounded should set return value to NaN: " << result;
}
TEST(MathErrorHandling,CheckBoundedMatrixDefaultPolicy) {
  const char* function = "check_bounded(%1%)";
  double result;
  double x;
  double low;
  double high;
  Eigen::Matrix<double,Eigen::Dynamic,1> x_vec;
  Eigen::Matrix<double,Eigen::Dynamic,1> low_vec;
  Eigen::Matrix<double,Eigen::Dynamic,1> high_vec;
  x_vec.resize(3);
  low_vec.resize(3);
  high_vec.resize(3);

  // x_vec, low_vec, high_vec
  result = 0;
  x_vec    << -100, 0, 100;
  low_vec  << -200, -100, 0;
  high_vec << 0, 100, 200;
  EXPECT_TRUE(check_bounded(function, x_vec, low_vec, high_vec, "x", &result));

  result = 0;
  x_vec    << -200, 100, 100;
  low_vec  << -200, -100, 0;
  high_vec << 0, 100, 200;
  EXPECT_TRUE(check_bounded(function, x_vec, low_vec, high_vec, "x", &result));

  result = 0;
  x_vec    << -100, 0, 300;
  low_vec  << -200, -100, 0;
  high_vec << 0, 100, 200;
  EXPECT_THROW(check_bounded(function, x_vec, low_vec, high_vec, "x", &result), std::domain_error);

  // x_vec, low_vec, high
  result = 0;
  x_vec    << -100, 0, 100;
  low_vec  << -200, -100, 0;
  high     =  1000;
  EXPECT_TRUE(check_bounded(function, x_vec, low_vec, high, "x", &result));

  result = 0;
  x_vec    << -100, 0, 100;
  low_vec  << -200, -100, 0;
  high     =  50;
  EXPECT_THROW(check_bounded(function, x_vec, low_vec, high, "x", &result), std::domain_error);
    
  // x_vec, low, high_vec
  result = 0;
  x_vec    << -100, 0, 100;
  low      = -1000;
  high_vec << 0, 100, 200;
  EXPECT_TRUE(check_bounded(function, x_vec, low, high_vec, "x", &result));

  result = 0;
  x_vec    << -100, 0, 100;
  low      =  -50;
  high_vec << 0, 100, 200;
  EXPECT_THROW(check_bounded(function, x_vec, low_vec, high, "x", &result), std::domain_error);

  // x_vec, low, high
  result = 0;
  x_vec    << -100, 0, 100;
  low      = -1000;
  high     =  1000;
  EXPECT_TRUE(check_bounded(function, x_vec, low, high, "x", &result));

  result = 0;
  x_vec    << -100, 5000, 100;
  low      =  -1000;
  high     =   1000;
  EXPECT_THROW(check_bounded(function, x_vec, low_vec, high, "x", &result), std::domain_error);
  
  // x, low_vec, high_vec
  result = 0;
  x        = 0;
  low_vec  << -100, -500, -1000;
  high_vec <<  100,  500,  1000;
  EXPECT_TRUE(check_bounded(function, x, low_vec, high_vec, "x", &result));

  result = 0;
  x        = 1500;
  low_vec  << -100, -500, -1000;
  high_vec <<  100,  500,  1000;
  EXPECT_THROW(check_bounded(function, x, low_vec, high_vec, "x", &result), std::domain_error);

  // x, low_vec, high
  result = 0;
  x        = 0;
  low_vec  << -100, -500, -1000;
  high     = 1000;
  EXPECT_TRUE(check_bounded(function, x, low_vec, high, "x", &result));

  result = 0;
  x        = 1500;
  low_vec  << -100, -500, -1000;
  high     = 1000;
  EXPECT_THROW(check_bounded(function, x, low_vec, high, "x", &result), std::domain_error);  
  
  // x, low, high_vec
  result = 0;
  x        = 0;
  low      = -1000;
  high_vec << 100, 500, 1000;
  EXPECT_TRUE(check_bounded(function, x, low, high_vec, "x", &result));

  result = 0;
  x        = 1500;
  low      = -1000;
  high_vec << 100, 500, 1000;
  EXPECT_THROW(check_bounded(function, x, low, high_vec, "x", &result), std::domain_error);
}

TEST(MathErrorHandling,CheckBoundedMatrixErrnoPolicy) {
  const char* function = "check_bounded(%1%)";
  double result;
  double x;
  double low;
  double high;
  Eigen::Matrix<double,Eigen::Dynamic,1> x_vec;
  Eigen::Matrix<double,Eigen::Dynamic,1> low_vec;
  Eigen::Matrix<double,Eigen::Dynamic,1> high_vec;
  x_vec.resize(3);
  low_vec.resize(3);
  high_vec.resize(3);

  // x_vec, low_vec, high_vec
  result = 0;
  x_vec    << -100, 0, 100;
  low_vec  << -200, -100, 0;
  high_vec << 0, 100, 200;
  EXPECT_TRUE(check_bounded(function, x_vec, low_vec, high_vec, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x_vec    << -200, 100, 100;
  low_vec  << -200, -100, 0;
  high_vec << 0, 100, 200;
  EXPECT_TRUE(check_bounded(function, x_vec, low_vec, high_vec, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x_vec    << -100, 0, 300;
  low_vec  << -200, -100, 0;
  high_vec << 0, 100, 200;
  EXPECT_FALSE(check_bounded(function, x_vec, low_vec, high_vec, "x", &result, errno_policy()));
  EXPECT_FALSE(!std::isnan(result));

  // x_vec, low_vec, high
  result = 0;
  x_vec    << -100, 0, 100;
  low_vec  << -200, -100, 0;
  high     =  1000;
  EXPECT_TRUE(check_bounded(function, x_vec, low_vec, high, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x_vec    << -100, 0, 100;
  low_vec  << -200, -100, 0;
  high     =  50;
  EXPECT_FALSE(check_bounded(function, x_vec, low_vec, high, "x", &result, errno_policy()));
  EXPECT_FALSE(!std::isnan(result));

  // x_vec, low, high_vec
  result = 0;
  x_vec    << -100, 0, 100;
  low      = -1000;
  high_vec << 0, 100, 200;
  EXPECT_TRUE(check_bounded(function, x_vec, low, high_vec, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x_vec    << -100, 0, 100;
  low      =  -50;
  high_vec << 0, 100, 200;
  EXPECT_FALSE(check_bounded(function, x_vec, low_vec, high, "x", &result, errno_policy()));
  EXPECT_FALSE(!std::isnan(result));

  // x_vec, low, high
  result = 0;
  x_vec    << -100, 0, 100;
  low      = -1000;
  high     =  1000;
  EXPECT_TRUE(check_bounded(function, x_vec, low, high, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x_vec    << -100, 5000, 100;
  low      =  -1000;
  high     =   1000;
  EXPECT_FALSE(check_bounded(function, x_vec, low_vec, high, "x", &result, errno_policy()));
  EXPECT_FALSE(!std::isnan(result));

  // x, low_vec, high_vec
  result = 0;
  x        = 0;
  low_vec  << -100, -500, -1000;
  high_vec <<  100,  500,  1000;
  EXPECT_TRUE(check_bounded(function, x, low_vec, high_vec, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x        = 1500;
  low_vec  << -100, -500, -1000;
  high_vec <<  100,  500,  1000;
  EXPECT_FALSE(check_bounded(function, x, low_vec, high_vec, "x", &result, errno_policy()));
  EXPECT_FALSE(!std::isnan(result));

  // x, low_vec, high
  result = 0;
  x        = 0;
  low_vec  << -100, -500, -1000;
  high     = 1000;
  EXPECT_TRUE(check_bounded(function, x, low_vec, high, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x        = 1500;
  low_vec  << -100, -500, -1000;
  high     = 1000;
  EXPECT_FALSE(check_bounded(function, x, low_vec, high, "x", &result, errno_policy()));
  EXPECT_FALSE(!std::isnan(result));

  // x, low, high_vec
  result = 0;
  x        = 0;
  low      = -1000;
  high_vec << 100, 500, 1000;
  EXPECT_TRUE(check_bounded(function, x, low, high_vec, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x        = 1500;
  low      = -1000;
  high_vec << 100, 500, 1000;
  EXPECT_FALSE(check_bounded(function, x, low, high_vec, "x", &result, errno_policy()));
  EXPECT_FALSE(!std::isnan(result));
}
