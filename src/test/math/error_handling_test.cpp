#include <gtest/gtest.h>
#include <stan/math/error_handling.hpp>

typedef boost::math::policies::policy<
  boost::math::policies::domain_error<boost::math::policies::errno_on_error>
  > errno_policy;
using namespace stan::math;

//---------- check_not_nan tests ----------
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
  double result;

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

//---------- check_finite tests ----------
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


// ---------- check_finite: vector tests ----------
TEST(MathErrorHandling,CheckFiniteVectorDefaultPolicy) {
  const char* function = "check_finite (%1%)";
  double result;
  std::vector<double> x;
  
  x.clear();
  x.push_back (-1);
  x.push_back (0);
  x.push_back (1);
  ASSERT_TRUE (check_finite (function, x, "x", &result, default_policy())) << "check_finite should be true with finite x";

  x.clear();
  x.push_back(-1);
  x.push_back(0);
  x.push_back(std::numeric_limits<double>::infinity());
  EXPECT_THROW (check_finite (function, x, "x", &result, default_policy()), std::domain_error) << "check_finite should throw exception on Inf";

  x.clear();
  x.push_back(-1);
  x.push_back(0);
  x.push_back(-std::numeric_limits<double>::infinity());
  EXPECT_THROW (check_finite (function, x, "x", &result, default_policy()), std::domain_error) << "check_finite should throw exception on -Inf";
  
  x.clear();
  x.push_back(-1);
  x.push_back(0);
  x.push_back(std::numeric_limits<double>::quiet_NaN());
  EXPECT_THROW (check_finite (function, x, "x", &result, default_policy()), std::domain_error) << "check_finite should throw exception on NaN";
}

TEST(MathErrorHandling,CheckFiniteVectorErrnoPolicy) {
  const char* function = "check_finite (%1%)";
  std::vector<double> x;
  x.push_back (-1);
  x.push_back (0);
  x.push_back (1);
  double result;
 
  result = 0;
  EXPECT_TRUE (check_finite (function, x, "x", &result, errno_policy())) << "check_finite should be true with finite x";

  result = 0;
  x.clear();
  x.push_back(-1);
  x.push_back(0);
  x.push_back(std::numeric_limits<double>::infinity());
  EXPECT_FALSE (check_finite (function, x, "x", &result, errno_policy())) << "check_finite should return FALSE on Inf";  
  EXPECT_TRUE (std::isnan (result)) << "check_finite should have returned NaN";


  result = 0;
  x.clear();
  x.push_back(-1);
  x.push_back(0);
  x.push_back(-std::numeric_limits<double>::infinity());
  EXPECT_FALSE (check_finite (function, x, "x", &result, errno_policy())) << "check_finite should return FALSE on -Inf";
  EXPECT_TRUE (std::isnan (result)) << "check_finite should have returned NaN";


  result = 0;
  x.clear();
  x.push_back(-1);
  x.push_back(0);
  x.push_back(std::numeric_limits<double>::quiet_NaN());
  EXPECT_FALSE (check_finite (function, x, "x", &result, errno_policy())) << "check_finite should return FALSE on NaN";
  EXPECT_TRUE (std::isnan (result)) << "check_finite should have returned NaN";
}

// ---------- check_finite: matrix tests ----------
TEST(MathErrorHandling,CheckFiniteMatrixDefaultPolicy) {
  const char* function = "check_finite (%1%)";
  double result;
  Eigen::Matrix<double,Eigen::Dynamic,1> x;
  
  result = 0;
  x.resize(3);
  x << -1, 0, 1;
  ASSERT_TRUE (check_finite (function, x, "x", &result, default_policy())) << "check_finite should be true with finite x";

  result = 0;
  x.resize(3);
  x << -1, 0, std::numeric_limits<double>::infinity();
  EXPECT_THROW (check_finite (function, x, "x", &result, default_policy()), std::domain_error) << "check_finite should throw exception on Inf";

  result = 0;
  x.resize(3);
  x << -1, 0, -std::numeric_limits<double>::infinity();
  EXPECT_THROW (check_finite (function, x, "x", &result, default_policy()), std::domain_error) << "check_finite should throw exception on -Inf";
  
  result = 0;
  x.resize(3);
  x << -1, 0, std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW (check_finite (function, x, "x", &result, default_policy()), std::domain_error) << "check_finite should throw exception on NaN";
}

TEST(MathErrorHandling,CheckFiniteMatrixErrnoPolicy) {
  const char* function = "check_finite (%1%)";
  double result;
  Eigen::Matrix<double,Eigen::Dynamic,1> x;
  
  result = 0;
  x.resize(3);
  x << -1, 0, 1;
  EXPECT_TRUE (check_finite (function, x, "x", &result, errno_policy())) << "check_finite should be true with finite x";

  result = 0;
  x.resize(3);
  x << -1, 0, std::numeric_limits<double>::infinity();
  EXPECT_FALSE (check_finite (function, x, "x", &result, errno_policy())) << "check_finite should return FALSE on Inf";
  EXPECT_TRUE (std::isnan (result)) << "check_finite should have returned NaN";

  result = 0;
  x.resize(3);
  x << -1, 0, -std::numeric_limits<double>::infinity();
  EXPECT_FALSE (check_finite (function, x, "x", &result, errno_policy())) << "check_finite should return FALSE on -Inf";
  EXPECT_TRUE (std::isnan (result)) << "check_finite should have returned NaN";

  result = 0;
  x.resize(3);
  x << -1, 0, std::numeric_limits<double>::quiet_NaN(); 
  EXPECT_FALSE (check_finite (function, x, "x", &result, errno_policy())) << "check_finite should return FALSE on NaN";
  EXPECT_TRUE (std::isnan (result)) << "check_finite should have returned NaN";
  }

// ---------- check_bounded tests ----------
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

// ----------  ----------
TEST(MathErrorHandling,CheckGreaterDefaultPolicy) {
  const char* function = "check_greater(%1%)";
  double x = 10.0;
  double lb = 0.0;
  double result;
 
  EXPECT_TRUE(check_greater(function, x, lb, "x", &result, default_policy())) 
    << "check_greater should be true with x > lb";
  
  x = -1.0;
  EXPECT_THROW(check_greater(function, x, lb, "x", &result, default_policy()), std::domain_error)
    << "check_greater should throw an exception with x < lb";

  x = lb;
  EXPECT_THROW(check_greater(function, x, lb, "x", &result, default_policy()), std::domain_error)
    << "check_greater should throw an exception with x == lb";

  x = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_greater(function, x, lb, "x", &result, default_policy()))
    << "check_greater should be true with x == Inf and lb = 0.0";

  x = 10.0;
  lb = std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_greater(function, x, lb, "x", &result, default_policy()), std::domain_error)
    << "check_greater should throw an exception with x == 10.0 and lb == Inf";

  x = std::numeric_limits<double>::infinity();
  lb = std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_greater(function, x, lb, "x", &result, default_policy()), std::domain_error)
    << "check_greater should throw an exception with x == Inf and lb == Inf";
}
TEST(MathErrorHandling,CheckGreaterErrnoPolicy) {
  const char* function = "check_greater(%1%)";
  double x = 10.0;
  double lb = 0.0;
  double result;
 
  result = 0;
  EXPECT_TRUE(check_greater(function, x, lb, "x", &result, errno_policy())) 
    << "check_greater should return true with x > lb";
  EXPECT_FALSE(std::isnan(result));

  result = 0;  
  x = -1.0;
  EXPECT_FALSE(check_greater(function, x, lb, "x", &result, errno_policy()))
    << "check_greater should return false with x < lb";
  EXPECT_TRUE(std::isnan(result));

  result = 0;
  x = lb;
  EXPECT_FALSE(check_greater(function, x, lb, "x", &result, errno_policy()))
    << "check_greater should return false with x == lb";
  EXPECT_TRUE(std::isnan(result));

  result = 0;
  x = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_greater(function, x, lb, "x", &result, errno_policy()))
    << "check_greater should return true with x == Inf and lb = 0.0";
  EXPECT_FALSE(std::isnan(result));

  result = 0;
  x = 10.0;
  lb = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(check_greater(function, x, lb, "x", &result, errno_policy()))
    << "check_greater should return false with x == 10.0 and lb == Inf";
  EXPECT_TRUE(std::isnan(result));

  result = 0;
  x = std::numeric_limits<double>::infinity();
  lb = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(check_greater(function, x, lb, "x", &result, errno_policy()))
    << "check_greater should return false with x == Inf and lb == Inf";
  EXPECT_TRUE(std::isnan(result));
}

TEST(MathErrorHandling,CheckGreaterMatrixDefaultPolicy) {
  const char* function = "check_greater(%1%)";
  double result;
  double x;
  double low;
  Eigen::Matrix<double,Eigen::Dynamic,1> x_vec;
  Eigen::Matrix<double,Eigen::Dynamic,1> low_vec;
  x_vec.resize(3);
  low_vec.resize(3);

  // x_vec, low_vec
  result = 0;
  x_vec   << -1, 0, 1;
  low_vec << -2, -1, 0;
  EXPECT_TRUE(check_greater(function, x_vec, low_vec, "x", &result)) 
    << "check_greater: matrix<3,1>, matrix<3,1>";

  x_vec   <<   -1,    0,   1;
  low_vec << -1.1, -0.1, 0.9;
  EXPECT_TRUE(check_greater(function, x_vec, low_vec, "x", &result)) 
    << "check_greater: matrix<3,1>, matrix<3,1>";


  x_vec   << -1, 0, std::numeric_limits<double>::infinity();
  low_vec << -2, -1, 0;
  EXPECT_TRUE(check_greater(function, x_vec, low_vec, "x", &result)) 
    << "check_greater: matrix<3,1>, matrix<3,1>, y has infinity";
  
  x_vec   << -1, 0, 1;
  low_vec << -2, 0, 0;
  EXPECT_THROW(check_greater(function, x_vec, low_vec, "x", &result), std::domain_error) 
    << "check_greater: matrix<3,1>, matrix<3,1>, should fail for index 1";
  
  x_vec   << -1, 0,  1;
  low_vec << -2, -1, std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_greater(function, x_vec, low_vec, "x", &result), std::domain_error) 
    << "check_greater: matrix<3,1>, matrix<3,1>, should fail with infinity";
  
  x_vec   << -1, 0,  std::numeric_limits<double>::infinity();
  low_vec << -2, -1, std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_greater(function, x_vec, low_vec, "x", &result), std::domain_error) 
    << "check_greater: matrix<3,1>, matrix<3,1>, should fail with infinity";
  
  x_vec   << -1, 0,  1;
  low_vec << -2, -1, -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_greater(function, x_vec, low_vec, "x", &result))
  << "check_greater: matrix<3,1>, matrix<3,1>, should pass with -infinity";

  // x_vec, low
  result = 0;
  x_vec   << -1, 0, 1;
  low = -2;
  EXPECT_TRUE(check_greater(function, x_vec, low, "x", &result)) 
    << "check_greater: matrix<3,1>, double";

  x_vec   <<   -1,    0,   1;
  low = -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_greater(function, x_vec, low, "x", &result)) 
    << "check_greater: matrix<3,1>, double";

  x_vec   << -1, 0, 1;
  low = 0;
  EXPECT_THROW(check_greater(function, x_vec, low, "x", &result), std::domain_error) 
    << "check_greater: matrix<3,1>, double, should fail for index 1/2";
  
  x_vec   << -1, 0,  1;
  low = std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_greater(function, x, low, "x", &result), std::domain_error) 
    << "check_greater: matrix<3,1>, double, should fail with infinity";
  
  // x, low_vec
  result = 0;
  x = 2;
  low_vec << -1, 0, 1;
  EXPECT_TRUE(check_greater(function, x, low_vec, "x", &result)) 
    << "check_greater: double, matrix<3,1>";

  x = 10;
  low_vec << -1, 0, -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_greater(function, x, low_vec, "x", &result)) 
    << "check_greater: double, matrix<3,1>, low has -inf";

  x = 10;
  low_vec << -1, 0, std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_greater(function, x, low_vec, "x", &result), std::domain_error) 
    << "check_greater: double, matrix<3,1>, low has inf";
  
  x = std::numeric_limits<double>::infinity();
  low_vec << -1, 0, std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_greater(function, x, low_vec, "x", &result), std::domain_error) 
    << "check_greater: double, matrix<3,1>, x is inf, low has inf";
  
  x = std::numeric_limits<double>::infinity();
  low_vec << -1, 0, 1;
  EXPECT_TRUE(check_greater(function, x, low_vec, "x", &result)) 
    << "check_greater: double, matrix<3,1>, x is inf";

  x = 1.1;
  low_vec << -1, 0, 1;
  EXPECT_TRUE(check_greater(function, x, low_vec, "x", &result)) 
    << "check_greater: double, matrix<3,1>";
  
  x = 0.9;
  low_vec << -1, 0, 1;
  EXPECT_THROW(check_greater(function, x, low_vec, "x", &result), std::domain_error) 
    << "check_greater: double, matrix<3,1>";
}

TEST(MathErrorHandling,CheckGreaterMatrixErrnoPolicy) {
  const char* function = "check_greater(%1%)";
  double result;
  double x;
  double low;
  Eigen::Matrix<double,Eigen::Dynamic,1> x_vec;
  Eigen::Matrix<double,Eigen::Dynamic,1> low_vec;
  x_vec.resize(3);
  low_vec.resize(3);

  // x_vec, low_vec
  result = 0;
  x_vec   << -1, 0, 1;
  low_vec << -2, -1, 0;
  EXPECT_TRUE(check_greater(function, x_vec, low_vec, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result))
  << "check_greater: matrix<3,1>, matrix<3,1>";

  result = 0;
  x_vec   <<   -1,    0,   1;
  low_vec << -1.1, -0.1, 0.9;
  EXPECT_TRUE(check_greater(function, x_vec, low_vec, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result))
  << "check_greater: matrix<3,1>, matrix<3,1>";

  result = 0;
  x_vec   << -1, 0, std::numeric_limits<double>::infinity();
  low_vec << -2, -1, 0;
  EXPECT_TRUE(check_greater(function, x_vec, low_vec, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result))
    << "check_greater: matrix<3,1>, matrix<3,1>, y has infinity";
  
  result = 0;
  x_vec   << -1, 0, 1;
  low_vec << -2, 0, 0;
  EXPECT_FALSE(check_greater(function, x_vec, low_vec, "x", &result, errno_policy()));
  EXPECT_FALSE(!std::isnan(result))
    << "check_greater: matrix<3,1>, matrix<3,1>, should fail for index 1";
  
  result = 0;
  x_vec   << -1, 0,  1;
  low_vec << -2, -1, std::numeric_limits<double>::infinity();
  EXPECT_FALSE(check_greater(function, x_vec, low_vec, "x", &result, errno_policy()));
  EXPECT_FALSE(!std::isnan(result))
    << "check_greater: matrix<3,1>, matrix<3,1>, should fail with infinity";
  
  result = 0;
  x_vec   << -1, 0,  std::numeric_limits<double>::infinity();
  low_vec << -2, -1, std::numeric_limits<double>::infinity();
  EXPECT_FALSE(check_greater(function, x_vec, low_vec, "x", &result, errno_policy()));
  EXPECT_FALSE(!std::isnan(result))
    << "check_greater: matrix<3,1>, matrix<3,1>, should fail with infinity";
  
  result = 0;
  x_vec   << -1, 0,  1;
  low_vec << -2, -1, -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_greater(function, x_vec, low_vec, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result))
    << "check_greater: matrix<3,1>, matrix<3,1>, should pass with -infinity";

  // x_vec, low
  result = 0;
  x_vec   << -1, 0, 1;
  low = -2;
  EXPECT_TRUE(check_greater(function, x_vec, low, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result))
    << "check_greater: matrix<3,1>, double";

  result = 0;
  x_vec   <<   -1,    0,   1;
  low = -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_greater(function, x_vec, low, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result))
    << "check_greater: matrix<3,1>, double";

  result = 0;
  x_vec   << -1, 0, 1;
  low = 0;
  EXPECT_FALSE(check_greater(function, x_vec, low, "x", &result, errno_policy()));
  EXPECT_FALSE(!std::isnan(result))
    << "check_greater: matrix<3,1>, double, should fail for index 1/2";
  
  result = 0;
  x_vec   << -1, 0,  1;
  low = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(check_greater(function, x, low, "x", &result, errno_policy()));
  EXPECT_FALSE(!std::isnan(result))
    << "check_greater: matrix<3,1>, double, should fail with infinity";
  

  // x, low_vec
  result = 0;
  x = 2;
  low_vec << -1, 0, 1;
  EXPECT_TRUE(check_greater(function, x, low_vec, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result))
    << "check_greater: double, matrix<3,1>";

  result = 0;
  x = 10;
  low_vec << -1, 0, -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(result = check_greater(function, x, low_vec, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result)) 
    << "check_greater: double, matrix<3,1>, low has -inf";

  result = 0;
  x = 10;
  low_vec << -1, 0, std::numeric_limits<double>::infinity();
  EXPECT_FALSE(check_greater(function, x, low_vec, "x", &result, errno_policy()));
  EXPECT_FALSE(!std::isnan(result))
    << "check_greater: double, matrix<3,1>, low has inf";
  
  result = 0;
  x = std::numeric_limits<double>::infinity();
  low_vec << -1, 0, std::numeric_limits<double>::infinity();
  EXPECT_FALSE(check_greater(function, x, low_vec, "x", &result, errno_policy()));
  EXPECT_FALSE(!std::isnan(result))
    << "check_greater: double, matrix<3,1>, x is inf, low has inf";
  
  result = 0;
  x = std::numeric_limits<double>::infinity();
  low_vec << -1, 0, 1;
  EXPECT_TRUE(check_greater(function, x, low_vec, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result))
    << "check_greater: double, matrix<3,1>, x is inf";

  result = 0;
  x = 1.1;
  low_vec << -1, 0, 1;
  EXPECT_TRUE(check_greater(function, x, low_vec, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result))
    << "check_greater: double, matrix<3,1>";
  
  result = 0;
  x = 0.9;
  low_vec << -1, 0, 1;
  EXPECT_FALSE(check_greater(function, x, low_vec, "x", &result, errno_policy()));
  EXPECT_FALSE(!std::isnan(result)) 
    << "check_greater: double, matrix<3,1>";
}





TEST(MathErrorHandling,CheckGreaterOrEqualDefaultPolicy) {
  const char* function = "check_greater_or_equal(%1%)";
  double x = 10.0;
  double lb = 0.0;
  double result;
 
  EXPECT_TRUE(check_greater_or_equal(function, x, lb, "x", &result, default_policy())) 
    << "check_greater_or_equal should be true with x > lb";
  
  x = -1.0;
  EXPECT_THROW(check_greater_or_equal(function, x, lb, "x", &result, default_policy()), std::domain_error)
    << "check_greater_or_equal should throw an exception with x < lb";

  x = lb;
  EXPECT_NO_THROW(check_greater_or_equal(function, x, lb, "x", &result, default_policy()))
    << "check_greater_or_equal should not throw an exception with x == lb";

  x = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_greater_or_equal(function, x, lb, "x", &result, default_policy()))
    << "check_greater should be true with x == Inf and lb = 0.0";

  x = 10.0;
  lb = std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_greater_or_equal(function, x, lb, "x", &result, default_policy()), std::domain_error)
    << "check_greater should throw an exception with x == 10.0 and lb == Inf";

  x = std::numeric_limits<double>::infinity();
  lb = std::numeric_limits<double>::infinity();
  EXPECT_NO_THROW(check_greater_or_equal(function, x, lb, "x", &result, default_policy()))
    << "check_greater should not throw an exception with x == Inf and lb == Inf";
}
TEST(MathErrorHandling,CheckGreaterOrEqualErrnoPolicy) {
  const char* function = "check_greater_or_equal(%1%)";
  double x = 10.0;
  double lb = 0.0;
  double result;
 
  result = 0;
  EXPECT_TRUE(check_greater_or_equal(function, x, lb, "x", &result, errno_policy())) 
    << "check_greater should return true with x > lb";
  EXPECT_FALSE(std::isnan(result));

  result = 0;  
  x = -1.0;
  EXPECT_FALSE(check_greater_or_equal(function, x, lb, "x", &result, errno_policy()))
    << "check_greater should return false with x < lb";
  EXPECT_TRUE(std::isnan(result));

  result = 0;
  x = lb;
  EXPECT_TRUE(check_greater_or_equal(function, x, lb, "x", &result, errno_policy()))
    << "check_greater should return true with x == lb";
  EXPECT_FALSE(std::isnan(result));

  result = 0;
  x = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_greater_or_equal(function, x, lb, "x", &result, errno_policy()))
    << "check_greater should return true with x == Inf and lb = 0.0";
  EXPECT_FALSE(std::isnan(result));

  result = 0;
  x = 10.0;
  lb = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(check_greater_or_equal(function, x, lb, "x", &result, errno_policy()))
    << "check_greater should return false with x == 10.0 and lb == Inf";
  EXPECT_TRUE(std::isnan(result));

  result = 0;
  x = std::numeric_limits<double>::infinity();
  lb = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_greater_or_equal(function, x, lb, "x", &result, errno_policy()))
    << "check_greater should return false with x == Inf and lb == Inf";
  EXPECT_FALSE(std::isnan(result));
}


TEST(MathErrorHandling,CheckLessDefaultPolicy) {
  const char* function = "check_less(%1%)";
  double x = -10.0;
  double lb = 0.0;
  double result;
 
  EXPECT_TRUE(check_less(function, x, lb, "x", &result, default_policy())) 
    << "check_less should be true with x < lb";
  
  x = 1.0;
  EXPECT_THROW(check_less(function, x, lb, "x", &result, default_policy()), std::domain_error)
    << "check_less should throw an exception with x > lb";

  x = lb;
  EXPECT_THROW(check_less(function, x, lb, "x", &result, default_policy()), std::domain_error)
    << "check_less should throw an exception with x == lb";

  x = -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_less(function, x, lb, "x", &result, default_policy()))
    << "check_less should be true with x == -Inf and lb = 0.0";

  x = -10.0;
  lb = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_less(function, x, lb, "x", &result, default_policy()), std::domain_error)
    << "check_less should throw an exception with x == -10.0 and lb == -Inf";

  x = -std::numeric_limits<double>::infinity();
  lb = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_less(function, x, lb, "x", &result, default_policy()), std::domain_error)
    << "check_less should throw an exception with x == -Inf and lb == -Inf";
}
TEST(MathErrorHandling,CheckLessErrnoPolicy) {
  const char* function = "check_less(%1%)";
  double x = -10.0;
  double lb = 0.0;
  double result;
 
  result = 0;
  EXPECT_TRUE(check_less(function, x, lb, "x", &result, errno_policy())) 
    << "check_less should return true with x < lb";
  EXPECT_FALSE(std::isnan(result));

  result = 0;  
  x = 1.0;
  EXPECT_FALSE(check_less(function, x, lb, "x", &result, errno_policy()))
    << "check_less should return false with x > lb";
  EXPECT_TRUE(std::isnan(result));

  result = 0;
  x = lb;
  EXPECT_FALSE(check_less(function, x, lb, "x", &result, errno_policy()))
    << "check_less should return false with x == lb";
  EXPECT_TRUE(std::isnan(result));

  result = 0;
  x = -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_less(function, x, lb, "x", &result, errno_policy()))
    << "check_less should return true with x == -Inf and lb = 0.0";
  EXPECT_FALSE(std::isnan(result));

  result = 0;
  x = -10.0;
  lb = -std::numeric_limits<double>::infinity();
  EXPECT_FALSE(check_less(function, x, lb, "x", &result, errno_policy()))
    << "check_less should return false with x == -10.0 and lb == -Inf";
  EXPECT_TRUE(std::isnan(result));

  result = 0;
  x = -std::numeric_limits<double>::infinity();
  lb = -std::numeric_limits<double>::infinity();
  EXPECT_FALSE(check_less(function, x, lb, "x", &result, errno_policy()))
    << "check_less should return false with x == -Inf and lb == -Inf";
  EXPECT_TRUE(std::isnan(result));
}


TEST(MathErrorHandling,CheckLessOrEqualDefaultPolicy) {
  const char* function = "check_less_or_equal(%1%)";
  double x = -10.0;
  double lb = 0.0;
  double result;
 
  EXPECT_TRUE(check_less_or_equal(function, x, lb, "x", &result, default_policy())) 
    << "check_less_or_equal should be true with x < lb";
  
  x = 1.0;
  EXPECT_THROW(check_less_or_equal(function, x, lb, "x", &result, default_policy()), std::domain_error)
    << "check_less_or_equal should throw an exception with x > lb";

  x = lb;
  EXPECT_NO_THROW(check_less_or_equal(function, x, lb, "x", &result, default_policy()))
    << "check_less_or_equal should not throw an exception with x == lb";

  x = -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_less_or_equal(function, x, lb, "x", &result, default_policy()))
    << "check_less should be true with x == -Inf and lb = 0.0";

  x = -10.0;
  lb = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_less_or_equal(function, x, lb, "x", &result, default_policy()), std::domain_error)
    << "check_less should throw an exception with x == -10.0 and lb == -Inf";

  x = -std::numeric_limits<double>::infinity();
  lb = -std::numeric_limits<double>::infinity();
  EXPECT_NO_THROW(check_less_or_equal(function, x, lb, "x", &result, default_policy()))
    << "check_less should not throw an exception with x == -Inf and lb == -Inf";
}
TEST(MathErrorHandling,CheckLessOrEqualErrnoPolicy) {
  const char* function = "check_less_or_equal(%1%)";
  double x = -10.0;
  double lb = 0.0;
  double result;
 
  result = 0;
  EXPECT_TRUE(check_less_or_equal(function, x, lb, "x", &result, errno_policy())) 
    << "check_less_or_equal should return true with x < lb";
  EXPECT_FALSE(std::isnan(result));

  result = 0;  
  x = 1.0;
  EXPECT_FALSE(check_less_or_equal(function, x, lb, "x", &result, errno_policy()))
    << "check_less_or_equal should return false with x > lb";
  EXPECT_TRUE(std::isnan(result));

  result = 0;
  x = lb;
  EXPECT_TRUE(check_less_or_equal(function, x, lb, "x", &result, errno_policy()))
    << "check_less_or_equal should return false with x == lb";
  EXPECT_FALSE(std::isnan(result));

  result = 0;
  x = -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_less_or_equal(function, x, lb, "x", &result, errno_policy()))
    << "check_less_or_equal should return true with x == -Inf and lb = 0.0";
  EXPECT_FALSE(std::isnan(result));

  result = 0;
  x = -10.0;
  lb = -std::numeric_limits<double>::infinity();
  EXPECT_FALSE(check_less_or_equal(function, x, lb, "x", &result, errno_policy()))
    << "check_less_or_equal should return false with x == -10.0 and lb == -Inf";
  EXPECT_TRUE(std::isnan(result));

  result = 0;
  x = -std::numeric_limits<double>::infinity();
  lb = -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_less_or_equal(function, x, lb, "x", &result, errno_policy()))
    << "check_less_or_equal should return true with x == -Inf and lb == -Inf";
  EXPECT_FALSE(std::isnan(result));
}

