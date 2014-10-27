#include <stan/error_handling/matrix/check_spsd_matrix.hpp>
#include <gtest/gtest.h>

TEST(ErrorHandlingMatrix, checkSpsdMatrixPosDef) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  double result;
  
  y.resize(3,3);
  y << 2, -1, 0, -1, 2, -1, 0, -1, 2;
  EXPECT_TRUE(stan::error_handling::check_spsd_matrix("checkSpsdMatrix(%1%)",
                                           y, "y", &result));

  y << 1, 2, 3, 2, 1, 2, 3, 2, 1;
  EXPECT_THROW(stan::error_handling::check_spsd_matrix("checkSpsdMatrix(%1%)", y, "y", &result), 
               std::domain_error);

  y.setZero();
  EXPECT_TRUE(stan::error_handling::check_spsd_matrix("checkSpsdMatrix(%1%)", y, "y", &result));
}

TEST(ErrorHandlingMatrix, checkSpsdMatrixZero) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y = 
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>::Zero(3,3);
  double result;
  EXPECT_TRUE(stan::error_handling::check_spsd_matrix("checkSpsdMatrix(%1%)", y, "y", &result));
}

TEST(ErrorHandlingMatrix, checkSpsdNotSquare) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y = 
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>::Zero(3,2);
  double result;
  EXPECT_THROW(stan::error_handling::check_spsd_matrix("checkSpsdMatrix(%1%)", y, "y", &result), 
               std::domain_error);
}

TEST(ErrorHandlingMatrix, checkSpsdMatrixPosDef_nan) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  double result;
  double nan = std::numeric_limits<double>::quiet_NaN();

  y.resize(3,3);
  y << 2, -1, 0, -1, 2, -1, 0, -1, 2;
  EXPECT_TRUE(stan::error_handling::check_spsd_matrix("checkSpsdMatrix(%1%)",
                                           y, "y", &result));

  y.setZero();
  EXPECT_TRUE(stan::error_handling::check_spsd_matrix("checkSpsdMatrix(%1%)", y, "y", &result));

  for (int i = 0; i < y.size(); i++) {
    y << 2, -1, 0, -1, 2, -1, 0, -1, 2;
    y(i) = nan;
    EXPECT_THROW(stan::error_handling::check_spsd_matrix("checkSpsdMatrix(%1%)",
                                               y, "y", &result),
                 std::domain_error);

    y.setZero();
    y(i) = nan;
    EXPECT_THROW(stan::error_handling::check_spsd_matrix("checkSpsdMatrix(%1%)",
                                               y, "y", &result),
                 std::domain_error);
  }
}
