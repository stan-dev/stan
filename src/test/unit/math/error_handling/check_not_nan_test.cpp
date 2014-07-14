#include <stan/math/error_handling/check_not_nan.hpp>
#include <gtest/gtest.h>

using stan::math::check_not_nan;

TEST(MathErrorHandling,CheckNotNan) {
  const char* function = "check_not_nan(%1%)";
  double x = 0;
  double result;

  EXPECT_TRUE(check_not_nan(function, x, "x",&result)) 
    << "check_not_nan should be true with finite x: " << x;

  x = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_not_nan(function, x, "x",&result)) 
    << "check_not_nan should be true with x = Inf: " << x;

  x = -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_not_nan(function, x, "x",&result)) 
    << "check_not_nan should be true with x = -Inf: " << x;

  x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(check_not_nan(function, x, "x",&result), std::domain_error) 
    << "check_not_nan should throw exception on NaN: " << x;
}


TEST(MathErrorHandling,CheckNotNanVectorized) {
  int N = 5;
  const char* function = "check_not_nan(%1%)";
  std::vector<double> x(N);
  double result;

  x.assign(N, 0);
  EXPECT_TRUE(check_not_nan(function, x, "x",&result)) 
    << "check_not_nan(vector) should be true with finite x: " << x[0];

  x.assign(N, std::numeric_limits<double>::infinity());
  EXPECT_TRUE(check_not_nan(function, x, "x",&result)) 
    << "check_not_nan(vector) should be true with x = Inf: " << x[0];

  x.assign(N, -std::numeric_limits<double>::infinity());
  EXPECT_TRUE(check_not_nan(function, x, "x",&result)) 
    << "check_not_nan(vector) should be true with x = -Inf: " << x[0];

  x.assign(N, std::numeric_limits<double>::quiet_NaN());
  EXPECT_THROW(check_not_nan(function, x, "x",&result), std::domain_error) 
    << "check_not_nan(vector) should throw exception on NaN: " << x[0];
}

TEST(MathErrorHandling, CheckNotNanVectorized_one_indexed_message) {
  int N = 5;
  const char* function = "check_not_nan(%1%)";
  std::vector<double> x(N);
  double result;
  std::string message;

  x.assign(N, 0);
  x[2] = std::numeric_limits<double>::quiet_NaN();
  try {
    check_not_nan(function, x, "x", &result);
    FAIL() << "should have thrown";
  } catch (std::domain_error& e) {
    message = e.what();
  } catch (...) {
    FAIL() << "threw the wrong error";
  }

  EXPECT_NE(std::string::npos, message.find("[3]"))
    << message;
}
