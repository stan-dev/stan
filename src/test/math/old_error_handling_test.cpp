#include <gtest/gtest.h>
#include <stan/math/error_handling.hpp>

typedef boost::math::policies::policy<
  boost::math::policies::domain_error<boost::math::policies::errno_on_error>
  > errno_policy;
using namespace stan::math;


// ---------- check_bounded tests ----------


// ----------  ----------

TEST(MathErrorHandling,CheckGreaterOrEqualMatrixDefaultPolicy) {
  const char* function = "check_greater_or_equal(%1%)";
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
  EXPECT_TRUE(check_greater_or_equal(function, x_vec, low_vec, "x", &result)) 
    << "check_greater_or_equal: matrix<3,1>, matrix<3,1>";

  x_vec   <<   -1,    0,   1;
  low_vec << -1.1, -0.1, 0.9;
  EXPECT_TRUE(check_greater_or_equal(function, x_vec, low_vec, "x", &result)) 
    << "check_greater_or_equal: matrix<3,1>, matrix<3,1>";


  x_vec   << -1, 0, std::numeric_limits<double>::infinity();
  low_vec << -2, -1, 0;
  EXPECT_TRUE(check_greater_or_equal(function, x_vec, low_vec, "x", &result)) 
    << "check_greater_or_equal: matrix<3,1>, matrix<3,1>, y has infinity";
  
  x_vec   << -1, 0, 1;
  low_vec << -2, 0, 0;
  EXPECT_TRUE(check_greater_or_equal(function, x_vec, low_vec, "x", &result))
    << "check_greater_or_equal: matrix<3,1>, matrix<3,1>, should pass for index 1";
  
  x_vec   << -1, 0,  1;
  low_vec << -2, -1, std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_greater_or_equal(function, x_vec, low_vec, "x", &result), std::domain_error) 
    << "check_greater_or_equal: matrix<3,1>, matrix<3,1>, should fail with infinity";
  
  x_vec   << -1, 0,  std::numeric_limits<double>::infinity();
  low_vec << -2, -1, std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_greater_or_equal(function, x_vec, low_vec, "x", &result))
    << "check_greater_or_equal: matrix<3,1>, matrix<3,1>, both bound and value infinity";
  
  x_vec   << -1, 0,  1;
  low_vec << -2, -1, -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_greater_or_equal(function, x_vec, low_vec, "x", &result))
  << "check_greater_or_equal: matrix<3,1>, matrix<3,1>, should pass with -infinity";

  // x_vec, low
  result = 0;
  x_vec   << -1, 0, 1;
  low = -2;
  EXPECT_TRUE(check_greater_or_equal(function, x_vec, low, "x", &result)) 
    << "check_greater_or_equal: matrix<3,1>, double";

  x_vec   <<   -1,    0,   1;
  low = -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_greater_or_equal(function, x_vec, low, "x", &result)) 
    << "check_greater_or_equal: matrix<3,1>, double";

  x_vec   << -1, 0, 1;
  low = 0;
  EXPECT_THROW(check_greater_or_equal(function, x_vec, low, "x", &result), std::domain_error) 
    << "check_greater_or_equal: matrix<3,1>, double, should fail for index 1/2";
  
  x_vec   << -1, 0,  1;
  low = std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_greater_or_equal(function, x_vec, low, "x", &result), std::domain_error) 
    << "check_greater_or_equal: matrix<3,1>, double, should fail with infinity";
  
  // x, low_vec
  result = 0;
  x = 2;
  low_vec << -1, 0, 1;
  EXPECT_TRUE(check_greater_or_equal(function, x, low_vec, "x", &result)) 
    << "check_greater_or_equal: double, matrix<3,1>";

  x = 10;
  low_vec << -1, 0, -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_greater_or_equal(function, x, low_vec, "x", &result)) 
    << "check_greater_or_equal: double, matrix<3,1>, low has -inf";

  x = 10;
  low_vec << -1, 0, std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_greater_or_equal(function, x, low_vec, "x", &result), std::domain_error) 
    << "check_greater_or_equal: double, matrix<3,1>, low has inf";
  
  x = std::numeric_limits<double>::infinity();
  low_vec << -1, 0, std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_greater_or_equal(function, x, low_vec, "x", &result))
    << "check_greater_or_equal: double, matrix<3,1>, x is inf, low has inf";
  
  x = std::numeric_limits<double>::infinity();
  low_vec << -1, 0, 1;
  EXPECT_TRUE(check_greater_or_equal(function, x, low_vec, "x", &result)) 
    << "check_greater_or_equal: double, matrix<3,1>, x is inf";

  x = 1.1;
  low_vec << -1, 0, 1;
  EXPECT_TRUE(check_greater_or_equal(function, x, low_vec, "x", &result)) 
    << "check_greater_or_equal: double, matrix<3,1>";
  
  x = 0.9;
  low_vec << -1, 0, 1;
  EXPECT_THROW(check_greater_or_equal(function, x, low_vec, "x", &result), std::domain_error) 
    << "check_greater_or_equal: double, matrix<3,1>";
}
TEST(MathErrorHandling,CheckGreaterOrEqualMatrixErrnoPolicy) {
  const char* function = "check_greater_or_equal(%1%)";
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
  EXPECT_TRUE(check_greater_or_equal(function, x_vec, low_vec, "x", &result, errno_policy())) 
    << "check_greater_or_equal: matrix<3,1>, matrix<3,1>";
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x_vec   <<   -1,    0,   1;
  low_vec << -1.1, -0.1, 0.9;
  EXPECT_TRUE(check_greater_or_equal(function, x_vec, low_vec, "x", &result, errno_policy())) 
    << "check_greater_or_equal: matrix<3,1>, matrix<3,1>";
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x_vec   << -1, 0, std::numeric_limits<double>::infinity();
  low_vec << -2, -1, 0;
  EXPECT_TRUE(check_greater_or_equal(function, x_vec, low_vec, "x", &result, errno_policy())) 
    << "check_greater_or_equal: matrix<3,1>, matrix<3,1>, y has infinity";
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x_vec   << -1, 0, 1;
  low_vec << -2, 0, 0;
  EXPECT_TRUE(check_greater_or_equal(function, x_vec, low_vec, "x", &result, errno_policy()))
    << "check_greater_or_equal: matrix<3,1>, matrix<3,1>, should pass for index 1";
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x_vec   << -1, 0,  1;
  low_vec << -2, -1, std::numeric_limits<double>::infinity();
  EXPECT_FALSE(check_greater_or_equal(function, x_vec, low_vec, "x", &result, errno_policy()))
    << "check_greater_or_equal: matrix<3,1>, matrix<3,1>, should fail with infinity";
  EXPECT_FALSE(!std::isnan(result));

  result = 0;
  x_vec   << -1, 0,  std::numeric_limits<double>::infinity();
  low_vec << -2, -1, std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_greater_or_equal(function, x_vec, low_vec, "x", &result, errno_policy()))
    << "check_greater_or_equal: matrix<3,1>, matrix<3,1>, both bound and value infinity";
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x_vec   << -1, 0,  1;
  low_vec << -2, -1, -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_greater_or_equal(function, x_vec, low_vec, "x", &result, errno_policy()))
    << "check_greater_or_equal: matrix<3,1>, matrix<3,1>, should pass with -infinity";
  EXPECT_TRUE(!std::isnan(result));

  // x_vec, low
  result = 0;
  x_vec   << -1, 0, 1;
  low = -2;
  EXPECT_TRUE(check_greater_or_equal(function, x_vec, low, "x", &result, errno_policy())) 
    << "check_greater_or_equal: matrix<3,1>, double";
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x_vec   <<   -1,    0,   1;
  low = -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_greater_or_equal(function, x_vec, low, "x", &result, errno_policy())) 
    << "check_greater_or_equal: matrix<3,1>, double";
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x_vec   << -1, 0, 1;
  low = 0;
  EXPECT_FALSE(check_greater_or_equal(function, x_vec, low, "x", &result, errno_policy()))
    << "check_greater_or_equal: matrix<3,1>, double, should fail for index 1/2";
  EXPECT_FALSE(!std::isnan(result));
  
  result = 0;
  x_vec   << -1, 0,  1;
  low = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(check_greater_or_equal(function, x_vec, low, "x", &result, errno_policy()))
    << "check_greater_or_equal: matrix<3,1>, double, should fail with infinity";
  EXPECT_FALSE(!std::isnan(result));
  
  // x, low_vec
  result = 0;
  x = 2;
  low_vec << -1, 0, 1;
  EXPECT_TRUE(check_greater_or_equal(function, x, low_vec, "x", &result, errno_policy())) 
    << "check_greater_or_equal: double, matrix<3,1>";
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x = 10;
  low_vec << -1, 0, -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_greater_or_equal(function, x, low_vec, "x", &result, errno_policy())) 
    << "check_greater_or_equal: double, matrix<3,1>, low has -inf";
  EXPECT_TRUE(!std::isnan(result));
  
  result = 0;
  x = 10;
  low_vec << -1, 0, std::numeric_limits<double>::infinity();
  EXPECT_FALSE(check_greater_or_equal(function, x, low_vec, "x", &result, errno_policy()))
    << "check_greater_or_equal: double, matrix<3,1>, low has inf";
  EXPECT_FALSE(!std::isnan(result));

  result = 0;
  x = std::numeric_limits<double>::infinity();
  low_vec << -1, 0, std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_greater_or_equal(function, x, low_vec, "x", &result, errno_policy()))
    << "check_greater_or_equal: double, matrix<3,1>, x is inf, low has inf";
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x = std::numeric_limits<double>::infinity();
  low_vec << -1, 0, 1;
  EXPECT_TRUE(check_greater_or_equal(function, x, low_vec, "x", &result, errno_policy())) 
    << "check_greater_or_equal: double, matrix<3,1>, x is inf";
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x = 1.1;
  low_vec << -1, 0, 1;
  EXPECT_TRUE(check_greater_or_equal(function, x, low_vec, "x", &result, errno_policy())) 
    << "check_greater_or_equal: double, matrix<3,1>";
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x = 0.9;
  low_vec << -1, 0, 1;
  EXPECT_FALSE(check_greater_or_equal(function, x, low_vec, "x", &result, errno_policy()))
    << "check_greater_or_equal: double, matrix<3,1>";
  EXPECT_FALSE(!std::isnan(result));
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

TEST(MathErrorHandling,CheckLessOrEqualMatrixDefaultPolicy) {
  const char* function = "check_less_or_equal(%1%)";
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
  EXPECT_TRUE(check_less_or_equal(function, x_vec, high, "x", &result));

  result = 0;
  x_vec << -5, 0, 5;
  high = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_less_or_equal(function, x_vec, high, "x", &result));

  result = 0;
  x_vec << -5, 0, 5;
  high = 5;
  EXPECT_TRUE(check_less_or_equal(function, x_vec, high, "x", &result));
  
  result = 0;
  x_vec << -5, 0, std::numeric_limits<double>::infinity();
  high = 5;
  EXPECT_THROW(check_less_or_equal(function, x_vec, high, "x", &result), std::domain_error);

  result = 0;
  x_vec << -5, 0, std::numeric_limits<double>::infinity();
  high = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_less_or_equal(function, x_vec, high, "x", &result));
  
  // x_vec, high_vec
  result = 0;
  x_vec << -5, 0, 5;
  high_vec << 0, 5, 10;
  EXPECT_TRUE(check_less_or_equal(function, x_vec, high_vec, "x", &result));

  result = 0;
  x_vec << -5, 0, 5;
  high_vec << std::numeric_limits<double>::infinity(), 10, 10;
  EXPECT_TRUE(check_less_or_equal(function, x_vec, high_vec, "x", &result));

  result = 0;
  x_vec << -5, 0, 5;
  high_vec << 10, 10, 5;
  EXPECT_TRUE(check_less_or_equal(function, x_vec, high_vec, "x", &result));
  
  result = 0;
  x_vec << -5, 0, std::numeric_limits<double>::infinity();
  high_vec << 10, 10, 10;
  EXPECT_THROW(check_less_or_equal(function, x_vec, high_vec, "x", &result), std::domain_error);

  result = 0;
  x_vec << -5, 0, std::numeric_limits<double>::infinity();
  high_vec << 10, 10, std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_less_or_equal(function, x_vec, high_vec, "x", &result));

  
  // x, high_vec
  result = 0;
  x = -100;
  high_vec << 0, 5, 10;
  EXPECT_TRUE(check_less_or_equal(function, x, high_vec, "x", &result));

  result = 0;
  x = 10;
  high_vec << 100, 200, std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_less_or_equal(function, x, high_vec, "x", &result));

  result = 0;
  x = 5;
  high_vec << 100, 200, 5;
  EXPECT_TRUE(check_less_or_equal(function, x, high_vec, "x", &result));
  
  result = 0;
  x = std::numeric_limits<double>::infinity();
  high_vec << 10, 20, 30;
  EXPECT_THROW(check_less_or_equal(function, x, high_vec, "x", &result), std::domain_error);

  result = 0;
  x = std::numeric_limits<double>::infinity();
  high_vec << std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_less_or_equal(function, x, high_vec, "x", &result));
}

TEST(MathErrorHandling,CheckLessOrEqualMatrixErrnoPolicy) {
  const char* function = "check_less_or_equal(%1%)";
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
  EXPECT_TRUE(check_less_or_equal(function, x_vec, high, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x_vec << -5, 0, 5;
  high = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_less_or_equal(function, x_vec, high, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x_vec << -5, 0, 5;
  high = 5;
  EXPECT_TRUE(check_less_or_equal(function, x_vec, high, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result));
  
  result = 0;
  x_vec << -5, 0, std::numeric_limits<double>::infinity();
  high = 5;
  EXPECT_FALSE(check_less_or_equal(function, x_vec, high, "x", &result, errno_policy()));
  EXPECT_FALSE(!std::isnan(result));
  
  result = 0;
  x_vec << -5, 0, std::numeric_limits<double>::infinity();
  high = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_less_or_equal(function, x_vec, high, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result));
  
  // x_vec, high_vec
  result = 0;
  x_vec << -5, 0, 5;
  high_vec << 0, 5, 10;
  EXPECT_TRUE(check_less_or_equal(function, x_vec, high_vec, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x_vec << -5, 0, 5;
  high_vec << std::numeric_limits<double>::infinity(), 10, 10;
  EXPECT_TRUE(check_less_or_equal(function, x_vec, high_vec, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x_vec << -5, 0, 5;
  high_vec << 10, 10, 5;
  EXPECT_TRUE(check_less_or_equal(function, x_vec, high_vec, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x_vec << -5, 0, std::numeric_limits<double>::infinity();
  high_vec << 10, 10, 10;
  EXPECT_FALSE(check_less_or_equal(function, x_vec, high_vec, "x", &result, errno_policy()));
  EXPECT_FALSE(!std::isnan(result));

  result = 0;
  x_vec << -5, 0, std::numeric_limits<double>::infinity();
  high_vec << 10, 10, std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_less_or_equal(function, x_vec, high_vec, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result));
  
  // x, high_vec
  result = 0;
  x = -100;
  high_vec << 0, 5, 10;
  EXPECT_TRUE(check_less_or_equal(function, x, high_vec, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x = 10;
  high_vec << 100, 200, std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_less_or_equal(function, x, high_vec, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x = 5;
  high_vec << 100, 200, 5;
  EXPECT_TRUE(check_less_or_equal(function, x, high_vec, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result));

  result = 0;
  x = std::numeric_limits<double>::infinity();
  high_vec << 10, 20, 30;
  EXPECT_FALSE(check_less_or_equal(function, x, high_vec, "x", &result, errno_policy()));
  EXPECT_FALSE(!std::isnan(result));

  result = 0;
  x = std::numeric_limits<double>::infinity();
  high_vec << std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_less_or_equal(function, x, high_vec, "x", &result, errno_policy()));
  EXPECT_TRUE(!std::isnan(result));
}
