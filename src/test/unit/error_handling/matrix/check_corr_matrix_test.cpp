#include <stan/error_handling/matrix/check_corr_matrix.hpp>
#include <gtest/gtest.h>

using stan::error_handling::check_corr_matrix;

TEST(ErrorHandlingMatrix, CheckCorrMatrix) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> y;
  y.resize(2,2);
  
  y << 1, 0, 0, 1;
  EXPECT_TRUE(check_corr_matrix("test", "y", y));

  y << 10, 0, 0, 10;
  EXPECT_THROW(check_corr_matrix("test", "y", y), 
               std::domain_error);
}

TEST(ErrorHandlingMatrix, CheckCorrMatrix_one_indexed_message) {
  std::string message;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> y;
  y.resize(2,2);
  
  y << 10, 0, 0, 1;
  try {
    check_corr_matrix("test", "y", y);
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

TEST(ErrorHandlingMatrix, CheckCorrMatrix_nan) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> y;
  y.resize(2,2);
  double nan = std::numeric_limits<double>::quiet_NaN();

  for (int i = 0; i < y.size(); i++) {
    y << 1, 0, 0, 1;
    y(i) = nan;
    EXPECT_THROW(check_corr_matrix("test", "y", y), std::domain_error);

    y << 10, 0, 0, 10;
    y(i) = nan;
    EXPECT_THROW(check_corr_matrix("test", "y", y), 
                 std::domain_error);
  }
}
