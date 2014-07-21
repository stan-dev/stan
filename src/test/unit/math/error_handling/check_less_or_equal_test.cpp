#include <stan/math/error_handling/check_less_or_equal.hpp>
#include <gtest/gtest.h>

using stan::math::check_less_or_equal;

TEST(MathErrorHandling,CheckLessOrEqual) {
  const char* function = "check_less_or_equal(%1%)";
  double x = -10.0;
  double lb = 0.0;
  double result;
 
  EXPECT_TRUE(check_less_or_equal(function, x, lb, "x", &result)) 
    << "check_less_or_equal should be true with x < lb";
  
  x = 1.0;
  EXPECT_THROW(check_less_or_equal(function, x, lb, "x", &result), 
               std::domain_error)
    << "check_less_or_equal should throw an exception with x > lb";

  x = lb;
  EXPECT_NO_THROW(check_less_or_equal(function, x, lb, "x", &result))
    << "check_less_or_equal should not throw an exception with x == lb";

  x = -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_less_or_equal(function, x, lb, "x", &result))
    << "check_less should be true with x == -Inf and lb = 0.0";

  x = -10.0;
  lb = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_less_or_equal(function, x, lb, "x", &result), 
               std::domain_error)
    << "check_less should throw an exception with x == -10.0 and lb == -Inf";

  x = -std::numeric_limits<double>::infinity();
  lb = -std::numeric_limits<double>::infinity();
  EXPECT_NO_THROW(check_less_or_equal(function, x, lb, "x", &result))
    << "check_less should not throw an exception with x == -Inf and lb == -Inf";
}

TEST(MathErrorHandling,CheckLessOrEqual_Matrix) {
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
  EXPECT_THROW(check_less_or_equal(function, x_vec, high, "x", &result),
               std::domain_error);

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
  EXPECT_THROW(check_less_or_equal(function, x_vec, high_vec, "x", &result),
               std::domain_error);

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
  EXPECT_THROW(check_less_or_equal(function, x, high_vec, "x", &result), 
               std::domain_error);

  result = 0;
  x = std::numeric_limits<double>::infinity();
  high_vec << std::numeric_limits<double>::infinity(), 
    std::numeric_limits<double>::infinity(), 
    std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_less_or_equal(function, x, high_vec, "x", &result));
}

TEST(MathErrorHandling,CheckLessOrEqual_Matrix_one_indexed_message) {
  const char* function = "check_less(%1%)";
  double result;
  double x;
  double high;
  Eigen::Matrix<double,Eigen::Dynamic,1> x_vec;
  Eigen::Matrix<double,Eigen::Dynamic,1> high_vec;
  x_vec.resize(3);
  high_vec.resize(3);
  std::string message;

  // x_vec, high
  result = 0;
  x_vec << -5, 0, 5;
  high = 4;

  try {
    check_less_or_equal(function, x_vec, high, "x", &result);
    FAIL() << "should have thrown";
  } catch (std::domain_error& e) {
    message = e.what();
  } catch (...) {
    FAIL() << "threw the wrong error";
  }

  EXPECT_NE(std::string::npos, message.find("[3]"))
    << message;

  // x_vec, high_vec
  result = 0;
  x_vec << -5, 6, 0;
  high_vec << 10, 5, 10;

  try {
    check_less_or_equal(function, x_vec, high_vec, "x", &result);
    FAIL() << "should have thrown";
  } catch (std::domain_error& e) {
    message = e.what();
  } catch (...) {
    FAIL() << "threw the wrong error";
  }

  EXPECT_NE(std::string::npos, message.find("[2]"))
    << message;


  // x, high_vec
  result = 0;
  x = 30;
  high_vec << 10, 20, 30;

  try {
    check_less_or_equal(function, x, high_vec, "x", &result);
    FAIL() << "should have thrown";
  } catch (std::domain_error& e) {
    message = e.what();
  } catch (...) {
    FAIL() << "threw the wrong error";
  }

  EXPECT_EQ(std::string::npos, message.find("["))
    << "no index provided" << std::endl
    << message;
}
