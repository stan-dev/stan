#include <stan/math/error_handling/matrix/check_symmetric.hpp>
#include <gtest/gtest.h>

TEST(MathErrorHandlingMatrix, checkSymmetric) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  double result;
  
  y.resize(2,2);
  y << 1, 3, 3, 1;
  EXPECT_TRUE(stan::math::check_symmetric("checkSymmetric(%1%)",
                                          y, "y", &result));

  y(0,1) = 3.5;
  EXPECT_THROW(stan::math::check_symmetric("checkSymmetric(%1%)", y, "y", &result), 
               std::domain_error);
}

TEST(MathErrorHandlingMatrix, checkSymmetric_one_indexed_message) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  double result;
  std::string message;
  
  y.resize(2,2);
  y << 1, 0, 3, 1;
  try {
    stan::math::check_symmetric("checkSymmetric(%1%)", y, "y", &result);
    FAIL() << "should have thrown";
  } catch (std::domain_error& e) {
    message = e.what();
  } catch (...) {
    FAIL() << "threw the wrong error";
  }

  EXPECT_NE(std::string::npos, message.find("[1,2]"))
    << message;
  EXPECT_NE(std::string::npos, message.find("[2,1]"))
    << message;
}
