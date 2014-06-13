#include <stan/math/error_handling/matrix/check_spsd_matrix.hpp>
#include <gtest/gtest.h>

TEST(MathErrorHandlingMatrix, checkSpsdMatrixPosDef) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  double result;
  
  y.resize(3,3);
  y << 2, -1, 0, -1, 2, -1, 0, -1, 2;
  EXPECT_TRUE(stan::math::check_spsd_matrix("checkSpsdMatrix(%1%)",
                                           y, "y", &result));

  y << 1, 2, 3, 2, 1, 2, 3, 2, 1;
  EXPECT_THROW(stan::math::check_spsd_matrix("checkSpsdMatrix(%1%)", y, "y", &result), 
               std::domain_error);

  y.setZero();
  EXPECT_TRUE(stan::math::check_spsd_matrix("checkSpsdMatrix(%1%)", y, "y", &result));
}

TEST(MathErrorHandlingMatrix, checkSpsdMatrixZero) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y = 
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>::Zero(3,3);
  double result;
  EXPECT_TRUE(stan::math::check_spsd_matrix("checkSpsdMatrix(%1%)", y, "y", &result));
}

TEST(MathErrorHandlingMatrix, checkSpsdNotSquare) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y = 
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>::Zero(3,2);
  double result;
  EXPECT_THROW(stan::math::check_spsd_matrix("checkSpsdMatrix(%1%)", y, "y", &result), 
               std::domain_error);
}
