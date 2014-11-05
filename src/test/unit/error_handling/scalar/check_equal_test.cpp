#include <stan/error_handling/scalar/check_equal.hpp>
#include <gtest/gtest.h>

using stan::error_handling::check_equal;

TEST(ErrorHandlingScalar,CheckEqual) {
  const std::string function = "check_equal";
  double x = 0.0;
  double eq = 0.0;
 
  EXPECT_TRUE(check_equal(function, "x", x, eq))
    << "check_equal should be true with x = eq";
  
  x = -1.0;
  EXPECT_THROW(check_equal(function, "x", x, eq),
               std::domain_error)
    << "check_equal should throw an exception with x < eq";

  x = eq;
  EXPECT_NO_THROW(check_equal(function, "x", x, eq))
    << "check_equal should not throw an exception with x == eq";

  x = std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_equal(function, "x", x, eq), 
               std::domain_error)
    << "check_equal should be false with x == Inf and eq = 0.0";

  x = 10.0;
  eq = std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_equal(function, "x", x, eq),
               std::domain_error)
    << "check_equal should throw an exception with x == 10.0 and eq == Inf";

  x = std::numeric_limits<double>::infinity();
  eq = std::numeric_limits<double>::infinity();
  EXPECT_NO_THROW(check_equal(function, "x", x, eq))
    << "check_equal should not throw an exception with x == Inf and eq == Inf";
}

TEST(ErrorHandlingScalar,CheckEqualMatrix) {
  const std::string function = "check_equal";
  Eigen::Matrix<double,Eigen::Dynamic,1> x_vec;
  Eigen::Matrix<double,Eigen::Dynamic,1> eq_vec;
  x_vec.resize(3);
  eq_vec.resize(3);

  // x_vec, low_vec
  x_vec   << -1, 0, 1;
  eq_vec << -1, 0, 1;
  EXPECT_TRUE(check_equal(function, "x", x_vec, eq_vec)) 
    << "check_equal: matrix<3,1>, matrix<3,1>";

  x_vec   <<   -1,    0,   1;
  eq_vec << -1.1, -0.1, 0.9;
  EXPECT_THROW(check_equal(function, "x", x_vec, eq_vec),
               std::domain_error) 
    << "check_equal: matrix<3,1>, matrix<3,1>";
  
  x_vec   << -1, 0,  1;
  eq_vec << -2, -1, std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_equal(function, "x", x_vec, eq_vec), 
               std::domain_error) 
    << "check_equal: matrix<3,1>, matrix<3,1>, should fail with infinity";
}


TEST(ErrorHandlingScalar,CheckEqual_Matrix_one_indexed_message) {
  const std::string function = "check_equal";
  double x;
  double eq;
  Eigen::Matrix<double,Eigen::Dynamic,1> x_vec(3);
  Eigen::Matrix<double,Eigen::Dynamic,1> eq_vec(3);
  std::string message;
  
  // x_vec, eq_vec
  x_vec   <<   0,    0,   1;
  eq_vec <<    0,    0,   0;

  try {
    check_equal(function, "x", x_vec, eq_vec);
    FAIL() << "should have thrown";
  } catch (std::domain_error& e) {
    message = e.what();
  } catch (...) {
    FAIL() << "threw the wrong error";
  }
  
  EXPECT_NE(std::string::npos, message.find("[3]"))
    << message;

  // x, eq_vec
  x = 1;
  try {
    check_equal(function, "x", x, eq_vec);
    FAIL() << "should have thrown";
  } catch (std::domain_error& e) {
    message = e.what();
  } catch (...) {
    FAIL() << "threw the wrong error";
  }
  
  EXPECT_EQ(std::string::npos, message.find("["))
    << "no index information" << std::endl
    << message;

  // x_vec, eq
  eq = 1;
  try {
    check_equal(function, "x", x_vec, eq);
    FAIL() << "should have thrown";
  } catch (std::domain_error& e) {
    message = e.what();
  } catch (...) {
    FAIL() << "threw the wrong error";
  }
  
  EXPECT_NE(std::string::npos, message.find("[1]"))
    << message;

}

TEST(ErrorHandlingScalar,CheckEqual_nan) {
  const std::string function = "check_equal";
  double x = 0.0;
  double eq = 0.0;
  double nan = std::numeric_limits<double>::quiet_NaN();

  EXPECT_THROW(check_equal(function, "x", x, nan),
               std::domain_error);
  EXPECT_THROW(check_equal(function, "x", nan, eq),
               std::domain_error);
  EXPECT_THROW(check_equal(function, "x", nan, nan),
               std::domain_error);

  Eigen::Matrix<double,Eigen::Dynamic,1> x_vec;
  Eigen::Matrix<double,Eigen::Dynamic,1> eq_vec;
  x_vec.resize(3);
  eq_vec.resize(3);

  // x_vec, low_vec
  x_vec   << nan, 0, 1;
  eq_vec << -1, 0, 1;
  EXPECT_THROW(check_equal(function, "x", x_vec, eq_vec),
               std::domain_error);

  eq_vec << nan, 0, 1;
  EXPECT_THROW(check_equal(function, "x", x_vec, eq_vec),
               std::domain_error);

  x_vec << -1, 0, 1;
  EXPECT_THROW(check_equal(function, "x", x_vec, eq_vec),
               std::domain_error);
}
