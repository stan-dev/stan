#include <gtest/gtest.h>
#include <stan/maths/error_handling.hpp>

typedef boost::math::policies::policy<
  boost::math::policies::domain_error<boost::math::policies::errno_on_error>
  > errno_policy;
using namespace stan::maths;

//---------- check_not_nan tests ----------
TEST(ProbDistributionsErrorHandling,CheckNotNanDefaultPolicy) {
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

TEST(ProbDistributionsErrorHandling,CheckNotNanErrnoPolicy) {
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


//---------- check_finite tests ----------
TEST(ProbDistributionsErrorHandling,CheckFiniteDefaultPolicy) {
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

TEST(ProbDistributionsErrorHandling,CheckFiniteErrnoPolicy) {
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
TEST(ProbDistributionsErrorHandling,CheckFiniteVectorDefaultPolicy) {
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

TEST(ProbDistributionsErrorHandling,CheckFiniteVectorErrnoPolicy) {
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

// // ---------- check_finite: matrix tests ----------
// TEST(ProbDistributionsErrorHandling,CheckFiniteMatrixDefaultPolicy) {
//   const char* function = "check_finite (%1%)";
//   double result;
//   Eigen::Matrix<double,Eigen::Dynamic,1> x;
  
//   result = 0;
//   x.resize(3);
//   x << -1, 0, 1;
//   ASSERT_TRUE (check_finite (function, x, "x", &result, default_policy())) << "check_finite should be true with finite x";

//   result = 0;
//   x.resize(3);
//   x << -1, 0, std::numeric_limits<double>::infinity();
//   EXPECT_THROW (check_finite (function, x, "x", &result, default_policy()), std::domain_error) << "check_finite should throw exception on Inf";

//   result = 0;
//   x.resize(3);
//   x << -1, 0, -std::numeric_limits<double>::infinity();
//   EXPECT_THROW (check_finite (function, x, "x", &result, default_policy()), std::domain_error) << "check_finite should throw exception on -Inf";
  
//   result = 0;
//   x.resize(3);
//   x << -1, 0, std::numeric_limits<double>::quiet_NaN();
//   EXPECT_THROW (check_finite (function, x, "x", &result, default_policy()), std::domain_error) << "check_finite should throw exception on NaN";
// }

// TEST(ProbDistributionsErrorHandling,CheckFiniteMatrixErrnoPolicy) {
//   const char* function = "check_finite (%1%)";
//   double result;
//   Eigen::Matrix<double,Eigen::Dynamic,1> x;
  
//   result = 0;
//   x.resize(3);
//   x << -1, 0, 1;
//   EXPECT_TRUE (check_finite (function, x, "x", &result, errno_policy())) << "check_finite should be true with finite x";

//   result = 0;
//   x.resize(3);
//   x << -1, 0, std::numeric_limits<double>::infinity();
//   EXPECT_FALSE (check_finite (function, x, "x", &result, errno_policy())) << "check_finite should return FALSE on Inf";
//   EXPECT_TRUE (std::isnan (result)) << "check_finite should have returned NaN";

//   result = 0;
//   x.resize(3);
//   x << -1, 0, -std::numeric_limits<double>::infinity();
//   EXPECT_FALSE (check_finite (function, x, "x", &result, errno_policy())) << "check_finite should return FALSE on -Inf";
//   EXPECT_TRUE (std::isnan (result)) << "check_finite should have returned NaN";

//   result = 0;
//   x.resize(3);
//   x << -1, 0, std::numeric_limits<double>::quiet_NaN(); 
//   EXPECT_FALSE (check_finite (function, x, "x", &result, errno_policy())) << "check_finite should return FALSE on NaN";
//   EXPECT_TRUE (std::isnan (result)) << "check_finite should have returned NaN";
//   }

// ---------- check_bounded tests ----------
TEST(ProbDistributionsErrorHandling,CheckBoundedDefaultPolicyX) {
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
TEST(ProbDistributionsErrorHandling,CheckBoundedDefaultPolicyLow) {
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
TEST(ProbDistributionsErrorHandling,CheckBoundedDefaultPolicyHigh) {
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


TEST(ProbDistributionsErrorHandling,CheckBoundedErrnoPolicyX) {
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
TEST(ProbDistributionsErrorHandling,CheckBoundedErrnoPolicyLow) {
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
TEST(ProbDistributionsErrorHandling,CheckBoundedErrnoPolicyHigh) {
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
//TEST(ProbDistributionsErrorHandling,)
