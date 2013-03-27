#include <stan/math/error_handling/check_greater.hpp>
#include <gtest/gtest.h>
#include <stan/math/matrix/Eigen.hpp>

using stan::math::default_policy;
typedef boost::math::policies::policy<
  boost::math::policies::domain_error<boost::math::policies::errno_on_error>
  > errno_policy;

using stan::math::check_greater;


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
  EXPECT_TRUE(check_greater(function, x, low_vec, "x", &result, errno_policy()));
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
