#include <stan/math/error_handling/check_nonnegative.hpp>
#include <gtest/gtest.h>

using stan::math::check_nonnegative;

TEST(MathErrorHandling,CheckNonnegative) {
  const char* function = "check_nonnegative(%1%)";
  double x = 0;
  double result;

  EXPECT_TRUE(check_nonnegative(function, x, "x",&result)) 
    << "check_nonnegative should be true with finite x: " << x;

  x = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_nonnegative(function, x, "x",&result)) 
    << "check_nonnegative should be true with x = Inf: " << x;

  x = -0.01;
  EXPECT_THROW(check_nonnegative(function, x, "x", &result), std::domain_error) 
    << "check_nonnegative should throw exception with x = " << x;

  x = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_nonnegative(function, x, "x",&result), std::domain_error) 
    << "check_nonnegative should throw exception with x = -Inf: " << x;

  x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(check_nonnegative(function, x, "x",&result), std::domain_error) 
    << "check_nonnegative should throw exception on NaN: " << x;
}


TEST(MathErrorHandling,CheckNonnegativeVectorized) {
  int N = 5;
  const char* function = "check_nonnegative(%1%)";
  std::vector<double> x(N);
  double result;

  x.assign(N, 0);
  EXPECT_TRUE(check_nonnegative(function, x, "x",&result)) 
    << "check_nonnegative(vector) should be true with finite x: " << x[0];

  x.assign(N, std::numeric_limits<double>::infinity());
  EXPECT_TRUE(check_nonnegative(function, x, "x",&result)) 
    << "check_nonnegative(vector) should be true with x = Inf: " << x[0];

  x.assign(N, -0.01);
  EXPECT_THROW(check_nonnegative(function, x, "x", &result), std::domain_error) 
    << "check_nonnegative should throw exception with x = " << x[0];


  x.assign(N, -std::numeric_limits<double>::infinity());
  EXPECT_THROW(check_nonnegative(function, x, "x",&result), std::domain_error) 
    << "check_nonnegative(vector) should throw an exception with x = -Inf: " << x[0];

  x.assign(N, std::numeric_limits<double>::quiet_NaN());
  EXPECT_THROW(check_nonnegative(function, x, "x",&result), std::domain_error) 
    << "check_nonnegative(vector) should throw exception on NaN: " << x[0];
}

TEST(MathErrorHandling, CheckNonnegativeVectorized_one_indexed_message) {
  int N = 5;
  const char* function = "check_nonnegative(%1%)";
  std::vector<double> x(N);
  double result;
  std::string message;

  x.assign(N, 0);
  x[2] = -1;
  try {
    check_nonnegative(function, x, "x", &result);
    FAIL() << "should have thrown";
  } catch (std::domain_error& e) {
    message = e.what();
  } catch (...) {
    FAIL() << "threw the wrong error";
  }

  EXPECT_NE(std::string::npos, message.find("[3]"));
}
