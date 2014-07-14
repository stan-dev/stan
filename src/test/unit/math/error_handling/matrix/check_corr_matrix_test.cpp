#include <stan/math/error_handling/matrix/check_corr_matrix.hpp>
#include <gtest/gtest.h>

using stan::math::check_corr_matrix;

TEST(MathErrorHandlingMatrix, CheckCorrMatrix) {
  double result;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> y;
  y.resize(2,2);
  
  y << 1, 0, 0, 1;
  EXPECT_TRUE(check_corr_matrix("test(%1%)", y, "y", &result));

  y << 10, 0, 0, 10;
  EXPECT_THROW(check_corr_matrix("test(%1%)", y, "y", &result), 
               std::domain_error);
}

TEST(MathErrorHandlingMatrix, CheckCorrMatrix_one_indexed_message) {
  std::string message;
  double result;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> y;
  y.resize(2,2);
  
  y << 10, 0, 0, 1;
  try {
    check_corr_matrix("test(%1%)", y, "y", &result);
    FAIL() << "should have thrown";
  } catch (std::domain_error& e) {
    message = e.what();
  } catch (...) {
    FAIL() << "threw the wrong error";
  }

  EXPECT_NE(std::string::npos, message.find("(1,1)"))
    << message;

  EXPECT_EQ(std::string::npos, message.find("(0,0)"))
    << message;
}
