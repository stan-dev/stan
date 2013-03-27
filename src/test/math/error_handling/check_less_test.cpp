#include <stan/math/error_handling/check_less.hpp>
#include <gtest/gtest.h>

using stan::math::default_policy;
typedef boost::math::policies::policy<
  boost::math::policies::domain_error<boost::math::policies::errno_on_error>
  > errno_policy;

using stan::math::check_less;

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

TEST(MathErrorHandling,CheckLessMatrixDefaultPolicy) {
  const char* function = "check_less(%1%)";
  double result;
  double x;
  double high;
  Eigen::Matrix<double,Eigen::Dynamic,1> x_vec;
  Eigen::Matrix<double,Eigen::Dynamic,1> high_vec;
  x_vec.resize(3);
  high_vec.resize(3);
  
  
  // x_vec, high
  result = 0;
  x_vec << -5, 0, 5;
  high = 10;
  EXPECT_TRUE(check_less(function, x_vec, high, "x", &result));

  result = 0;
  x_vec << -5, 0, 5;
  high = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_less(function, x_vec, high, "x", &result));

  result = 0;
  x_vec << -5, 0, 5;
  high = 5;
  EXPECT_THROW(check_less(function, x_vec, high, "x", &result), std::domain_error);
  
  result = 0;
  x_vec << -5, 0, std::numeric_limits<double>::infinity();
  high = 5;
  EXPECT_THROW(check_less(function, x_vec, high, "x", &result), std::domain_error);

  result = 0;
  x_vec << -5, 0, std::numeric_limits<double>::infinity();
  high = std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_less(function, x_vec, high, "x", &result), std::domain_error);
  
  // x_vec, high_vec
  result = 0;
  x_vec << -5, 0, 5;
  high_vec << 0, 5, 10;
  EXPECT_TRUE(check_less(function, x_vec, high_vec, "x", &result));

  result = 0;
  x_vec << -5, 0, 5;
  high_vec << std::numeric_limits<double>::infinity(), 10, 10;
  EXPECT_TRUE(check_less(function, x_vec, high_vec, "x", &result));

  result = 0;
  x_vec << -5, 0, 5;
  high_vec << 10, 10, 5;
  EXPECT_THROW(check_less(function, x_vec, high_vec, "x", &result), std::domain_error);
  
  result = 0;
  x_vec << -5, 0, std::numeric_limits<double>::infinity();
  high_vec << 10, 10, 10;
  EXPECT_THROW(check_less(function, x_vec, high_vec, "x", &result), std::domain_error);

  result = 0;
  x_vec << -5, 0, std::numeric_limits<double>::infinity();
  high_vec << 10, 10, std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_less(function, x_vec, high_vec, "x", &result), std::domain_error);

  
  // x, high_vec
  result = 0;
  x = -100;
  high_vec << 0, 5, 10;
  EXPECT_TRUE(check_less(function, x, high_vec, "x", &result));

  result = 0;
  x = 10;
  high_vec << 100, 200, std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_less(function, x, high_vec, "x", &result));

  result = 0;
  x = 5;
  high_vec << 100, 200, 5;
  EXPECT_THROW(check_less(function, x, high_vec, "x", &result), std::domain_error);
  
  result = 0;
  x = std::numeric_limits<double>::infinity();
  high_vec << 10, 20, 30;
  EXPECT_THROW(check_less(function, x, high_vec, "x", &result), std::domain_error);

  result = 0;
  x = std::numeric_limits<double>::infinity();
  high_vec << std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_less(function, x, high_vec, "x", &result), std::domain_error);
}

TEST(MathErrorHandling,CheckLessMatrixErrnoPolicy) {
  const char* function = "check_less(%1%)";
  double result;
  double x;
  double high;
  Eigen::Matrix<double,Eigen::Dynamic,1> x_vec;
  Eigen::Matrix<double,Eigen::Dynamic,1> high_vec;
  x_vec.resize(3);
  high_vec.resize(3);
  
  
  // x_vec, high
  result = 0;
  x_vec << -5, 0, 5;
  high = 10;
  EXPECT_TRUE(check_less(function, x_vec, high, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x_vec << -5, 0, 5;
  high = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_less(function, x_vec, high, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x_vec << -5, 0, 5;
  high = 5;
  EXPECT_FALSE(check_less(function, x_vec, high, "x", &result, errno_policy()));
  EXPECT_FALSE(!std::isnan(result));
  
  result = 0;
  x_vec << -5, 0, std::numeric_limits<double>::infinity();
  high = 5;
  EXPECT_FALSE(check_less(function, x_vec, high, "x", &result, errno_policy()));
  EXPECT_FALSE(!std::isnan(result));
  
  result = 0;
  x_vec << -5, 0, std::numeric_limits<double>::infinity();
  high = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(check_less(function, x_vec, high, "x", &result, errno_policy()));
  EXPECT_FALSE(!std::isnan(result));
  
  // x_vec, high_vec
  result = 0;
  x_vec << -5, 0, 5;
  high_vec << 0, 5, 10;
  EXPECT_TRUE(check_less(function, x_vec, high_vec, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x_vec << -5, 0, 5;
  high_vec << std::numeric_limits<double>::infinity(), 10, 10;
  EXPECT_TRUE(check_less(function, x_vec, high_vec, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x_vec << -5, 0, 5;
  high_vec << 10, 10, 5;
  EXPECT_FALSE(check_less(function, x_vec, high_vec, "x", &result, errno_policy()));
  EXPECT_FALSE(!std::isnan(result));

  result = 0;
  x_vec << -5, 0, std::numeric_limits<double>::infinity();
  high_vec << 10, 10, 10;
  EXPECT_FALSE(check_less(function, x_vec, high_vec, "x", &result, errno_policy()));
  EXPECT_FALSE(!std::isnan(result));

  result = 0;
  x_vec << -5, 0, std::numeric_limits<double>::infinity();
  high_vec << 10, 10, std::numeric_limits<double>::infinity();
  EXPECT_FALSE(check_less(function, x_vec, high_vec, "x", &result, errno_policy()));
  EXPECT_FALSE(!std::isnan(result));
  
  // x, high_vec
  result = 0;
  x = -100;
  high_vec << 0, 5, 10;
  EXPECT_TRUE(check_less(function, x, high_vec, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x = 10;
  high_vec << 100, 200, std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_less(function, x, high_vec, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x = 5;
  high_vec << 100, 200, 5;
  EXPECT_FALSE(check_less(function, x, high_vec, "x", &result, errno_policy()));
  EXPECT_FALSE(!std::isnan(result));

  result = 0;
  x = std::numeric_limits<double>::infinity();
  high_vec << 10, 20, 30;
  EXPECT_FALSE(check_less(function, x, high_vec, "x", &result, errno_policy()));
  EXPECT_FALSE(!std::isnan(result));

  result = 0;
  x = std::numeric_limits<double>::infinity();
  high_vec << std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity();
  EXPECT_FALSE(check_less(function, x, high_vec, "x", &result, errno_policy()));
  EXPECT_FALSE(!std::isnan(result));
}
