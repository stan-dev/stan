#include <stan/math/error_handling/check_greater.hpp>
#include <gtest/gtest.h>
#include <stan/math/matrix/Eigen.hpp>

using stan::math::check_greater;

TEST(MathErrorHandling,CheckGreater) {
  const char* function = "check_greater(%1%)";
  double x = 10.0;
  double lb = 0.0;
  double result;
 
  EXPECT_TRUE(check_greater(function, x, lb, "x", &result)) 
    << "check_greater should be true with x > lb";
  
  x = -1.0;
  EXPECT_THROW(check_greater(function, x, lb, "x", &result), std::domain_error)
    << "check_greater should throw an exception with x < lb";

  x = lb;
  EXPECT_THROW(check_greater(function, x, lb, "x", &result), std::domain_error)
    << "check_greater should throw an exception with x == lb";

  x = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_greater(function, x, lb, "x", &result))
    << "check_greater should be true with x == Inf and lb = 0.0";

  x = 10.0;
  lb = std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_greater(function, x, lb, "x", &result), std::domain_error)
    << "check_greater should throw an exception with x == 10.0 and lb == Inf";

  x = std::numeric_limits<double>::infinity();
  lb = std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_greater(function, x, lb, "x", &result), std::domain_error)
    << "check_greater should throw an exception with x == Inf and lb == Inf";
}

TEST(MathErrorHandling,CheckGreater_Matrix) {
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
