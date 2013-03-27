#include <stan/math/error_handling/check_finite.hpp>
#include <gtest/gtest.h>

using stan::math::default_policy;
typedef boost::math::policies::policy<
  boost::math::policies::domain_error<boost::math::policies::errno_on_error>
  > errno_policy;

using stan::math::check_finite;

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
