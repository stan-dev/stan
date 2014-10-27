#include <stan/error_handling/matrix/check_cov_matrix.hpp>
#include <gtest/gtest.h>

TEST(MathErrorHandlingMatrix, checkCovMatrix) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  double result;
  
  y.resize(3,3);
  y << 2, -1, 0, -1, 2, -1, 0, -1, 2;
  EXPECT_TRUE(stan::error_handling::check_cov_matrix("checkCovMatrix(%1%)",
                                           y, "y", &result));

  y << 1, 2, 3, 2, 1, 2, 3, 2, 1;
  EXPECT_THROW(stan::error_handling::check_cov_matrix("checkCovMatrix(%1%)", y, "y", &result), 
               std::domain_error);
}

TEST(MathErrorHandlingMatrix, checkCovMatrix_nan) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  double result;
  double nan = std::numeric_limits<double>::quiet_NaN();

  y.resize(3,3);
  y << 2, -1, 0, -1, 2, -1, 0, -1, 2;
  EXPECT_TRUE(stan::error_handling::check_cov_matrix("checkCovMatrix(%1%)",
                                           y, "y", &result));
  
  for (int i = 0; i < y.size(); i++) {
    y.resize(3,3);
    y << 2, -1, 0, -1, 2, -1, 0, -1, 2;
    y(i) = nan;
    EXPECT_THROW(stan::error_handling::check_cov_matrix("checkCovMatrix(%1%)", y, "y", &result), 
                 std::domain_error);
    y << 2, -1, 0, -1, 2, -1, 0, -1, 2;
  }
}
