#include <gtest/gtest.h>
#include "stan/prob/distributions_error_handling.hpp"
#include <limits>

typedef boost::math::policies::policy<
  boost::math::policies::domain_error<boost::math::policies::errno_on_error>, 
  boost::math::policies::pole_error<boost::math::policies::errno_on_error>,
  boost::math::policies::overflow_error<boost::math::policies::errno_on_error>,
  boost::math::policies::evaluation_error<boost::math::policies::errno_on_error> 
  > errno_policy;
typedef boost::math::policies::policy<> default_policy;

using namespace stan::prob;

//---------- convert: double tests ----------
TEST(ProbDistributionsErrorHandling,ConvertDouble) {
  double x = 100.0;
  EXPECT_FLOAT_EQ (x, convert(x)) << "Expect the same number back";
}
TEST(ProbDistributionsErrorHandling,ConvertDoubleMax) {
  double x = std::numeric_limits<double>::max();
  EXPECT_FLOAT_EQ (x, convert(x)) << "Check for std_numeric_limits<double>::max: " << x;
}
TEST(ProbDistributionsErrorHandling,ConvertDoubleMinusMax) {
  double x = -std::numeric_limits<double>::max();
  EXPECT_FLOAT_EQ (x, convert(x)) << "Check for -std_numeric_limits<double>::max: " << x;
}
TEST(ProbDistributionsErrorHandling,ConvertDoubleQuietNaN) {
  double x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE (std::isnan (convert(x))) << "Check for std_numeric_limits<double>::quiet_NaN: " << x;
}

//---------- convert: var tests ----------
TEST(ProbDistributionsErrorHandling,ConvertVar) {
  stan::agrad::var x(100.0);
  EXPECT_FLOAT_EQ (100.0, convert(x)) << "Expect the same number back";
}
TEST(ProbDistributionsErrorHandling,ConvertVarMax) {
  stan::agrad::var x(std::numeric_limits<double>::max());
  EXPECT_FLOAT_EQ (std::numeric_limits<double>::max(), convert(x)) << "Check for std_numeric_limits<double>::max: " << x;
  x = std::numeric_limits<stan::agrad::var>::max();
  EXPECT_FLOAT_EQ (std::numeric_limits<double>::max(), convert(x)) << "Check for std_numeric_limits<stan::agrad::var>::max: " << x;
}
TEST(ProbDistributionsErrorHandling,ConvertVarMinusMax) {
  stan::agrad::var x = -std::numeric_limits<double>::max();
  EXPECT_FLOAT_EQ (-std::numeric_limits<double>::max(), convert(x)) << "Check for -std_numeric_limits<double>::max: " << x;
  x = -std::numeric_limits<stan::agrad::var>::max();
  EXPECT_FLOAT_EQ (-std::numeric_limits<double>::max(), convert(x)) << "Check for -std_numeric_limits<stan::agrad::var>::max: " << x;
}
TEST(ProbDistributionsErrorHandling,ConvertVarQuietNaN) {
  stan::agrad::var x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE (std::isnan (convert(x))) << "Check for std_numeric_limits<double>::quiet_NaN: " << x;
  x = std::numeric_limits<stan::agrad::var>::quiet_NaN();
  EXPECT_TRUE (std::isnan (convert(x))) << "Check for std_numeric_limits<stan::agrad::var>::quiet_NaN: " << x;
}

//---------- check_x tests ----------
TEST(ProbDistributionsErrorHandling,CheckXDefaultPolicy) {
  const char* function = "function %1%";
  double x = 0;
  double result;
 
  EXPECT_TRUE (check_x (function, x, &result, default_policy())) << "check_x should be true with finite x:" << x;
  x = std::numeric_limits<double>::max();
  EXPECT_TRUE (check_x (function, x, &result, default_policy())) << "check_x should return TRUE on Inf: " << x;
  x = -std::numeric_limits<double>::max();
  EXPECT_TRUE (check_x (function, x, &result, default_policy())) << "check_x should return TRUE on -Inf: " << x;

  x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW (check_x (function, x, &result, default_policy()), std::domain_error) << "check_x should throw exception on NaN: " << x;
}

TEST(ProbDistributionsErrorHandling,CheckXErrnoPolicy) {
  const char* function = "function %1%";
  double x = 0;
  double result;
 
  EXPECT_TRUE (check_x (function, x, &result, errno_policy())) << "check_x should be true with finite x:" << x;
  x = std::numeric_limits<double>::max();
  EXPECT_TRUE (check_x (function, x, &result, errno_policy())) << "check_x should return TRUE on Inf: " << x;
  x = -std::numeric_limits<double>::max();
  EXPECT_TRUE (check_x (function, x, &result, errno_policy())) << "check_x should return TRUE on -Inf: " << x;

  x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE (check_x (function, x, &result, errno_policy())) << "check_x should return FALSE on NaN: " << x;
  EXPECT_TRUE (std::isnan (result)) << "check_x should have returned NaN: " << x;
}


// ---------- check_x: vector tests ----------
TEST(ProbDistributionsErrorHandling,CheckXVectorDefaultPolicy) {
  const char* function = "function %1%";
  double result;
  std::vector<double> x;
  
  x.clear();
  x.push_back (-1);
  x.push_back (0);
  x.push_back (1);
  ASSERT_TRUE (check_x (function, x, &result, default_policy())) << "check_x should be true with finite x";

  x.clear();
  x.push_back(-1);
  x.push_back(0);
  x.push_back(std::numeric_limits<double>::max());
  EXPECT_TRUE (check_x (function, x, &result, default_policy())) << "check_x should return TRUE on Inf";

  x.clear();
  x.push_back(-1);
  x.push_back(0);
  x.push_back(-std::numeric_limits<double>::max());
  EXPECT_TRUE (check_x (function, x, &result, default_policy())) << "check_x should return TRUE on -Inf";
  
  x.clear();
  x.push_back(-1);
  x.push_back(0);
  x.push_back(std::numeric_limits<double>::quiet_NaN());
  EXPECT_THROW (check_x (function, x, &result, default_policy()), std::domain_error) << "check_x should throw exception on NaN";
}

TEST(ProbDistributionsErrorHandling,CheckXVectorErrnoPolicy) {
  const char* function = "function %1%";
  std::vector<double> x;
  x.push_back (-1);
  x.push_back (0);
  x.push_back (1);
  double result;
 
  EXPECT_TRUE (check_x (function, x, &result, errno_policy())) << "check_x should be true with finite x";

  x.clear();
  x.push_back(-1);
  x.push_back(0);
  x.push_back(std::numeric_limits<double>::max());
  EXPECT_TRUE (check_x (function, x, &result, errno_policy())) << "check_x should return TRUE on Inf";


  x.clear();
  x.push_back(-1);
  x.push_back(0);
  x.push_back(-std::numeric_limits<double>::max());
  EXPECT_TRUE (check_x (function, x, &result, errno_policy())) << "check_x should return TRUE on -Inf";

  x.clear();
  x.push_back(-1);
  x.push_back(0);
  x.push_back(std::numeric_limits<double>::quiet_NaN());
  EXPECT_FALSE (check_x (function, x, &result, errno_policy())) << "check_x should return FALSE on NaN";
  EXPECT_TRUE (std::isnan (result)) << "check_x should have returned NaN";
}

// ---------- check_x: matrix tests ----------
TEST(ProbDistributionsErrorHandling,CheckXMatrixDefaultPolicy) {
  const char* function = "function %1%";
  double result;
  Eigen::Matrix<double,Eigen::Dynamic,1> x;
  
  x.resize(3);
  x << -1, 0, 1;
  ASSERT_TRUE (check_x (function, x, &result, default_policy())) << "check_x should be true with finite x";

  x.resize(3);
  x << -1, 0, std::numeric_limits<double>::max();
  EXPECT_TRUE (check_x (function, x, &result, default_policy())) << "check_x should return TRUE on Inf";

  x.resize(3);
  x << -1, 0, -std::numeric_limits<double>::max();
  EXPECT_TRUE (check_x (function, x, &result, default_policy())) << "check_x should return TRUE on -Inf";
  
  x.resize(3);
  x << -1, 0, std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW (check_x (function, x, &result, default_policy()), std::domain_error) << "check_x should throw exception on NaN";
}

TEST(ProbDistributionsErrorHandling,CheckXMatrixErrnoPolicy) {
  const char* function = "function %1%";
  double result;
  Eigen::Matrix<double,Eigen::Dynamic,1> x;
  
  x.resize(3);
  x << -1, 0, 1;
 
  EXPECT_TRUE (check_x (function, x, &result, errno_policy())) << "check_x should be true with finite x";

  x.resize(3);
  x << -1, 0, std::numeric_limits<double>::max();
  EXPECT_TRUE (check_x (function, x, &result, errno_policy())) << "check_x should return TRUE on Inf";


  x.resize(3);
  x << -1, 0, -std::numeric_limits<double>::max();
  EXPECT_TRUE (check_x (function, x, &result, errno_policy())) << "check_x should return TRUE on -Inf";

  x.resize(3);
  x << -1, 0, std::numeric_limits<double>::quiet_NaN();  EXPECT_FALSE (check_x (function, x, &result, errno_policy())) << "check_x should return FALSE on NaN";
  EXPECT_TRUE (std::isnan (result)) << "check_x should have returned NaN";
  }



// ----------  ----------
//TEST(ProbDistributionsErrorHandling,)
