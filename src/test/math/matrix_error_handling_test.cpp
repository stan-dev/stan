#include <gtest/gtest.h>
#include <stan/math/matrix_error_handling.hpp>

TEST(stanMathMatrixErrorHandling, checkNotNanEigenRow) {
  stan::math::vector_d y;
  double result;
  y.resize(3);
  
  EXPECT_TRUE(stan::math::check_not_nan("checkNotNanEigenRow(%1)",
                                        y, "y", &result));
  EXPECT_TRUE(stan::math::check_not_nan("checkNotNanEigenRow(%1)",
                                        y, "y"));
  
  y(1) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(stan::math::check_not_nan("checkNotNanEigenRow(%1%)", y, "y", &result), 
               std::domain_error);
  EXPECT_THROW(stan::math::check_not_nan("checkNotNanEigenRow(%1%)", y, "y"), 
               std::domain_error);
  
}
TEST(stanMathMatrixErrorHandling, checkSimplex) {
  Eigen::Matrix<double,Eigen::Dynamic,1> y(2);
  double result;
  y << 0.5, 0.5;
  
  EXPECT_TRUE(stan::math::check_simplex("checkSimplex(%1%)",
                                        y, "y", &result));
  EXPECT_TRUE(stan::math::check_simplex("checkSimplex(%1%)",
                                        y, "y"));
                  
  y[1] = 0.55;
  EXPECT_THROW(stan::math::check_simplex("checkSimplex(%1%)", y, "y", &result), 
               std::domain_error);
  EXPECT_THROW(stan::math::check_simplex("checkSimplex(%1%)", y, "y"),
               std::domain_error);
}
TEST(stanMathMatrixErrorHandling, checkSymmetric) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  double result;
  
  y.resize(2,2);
  y << 1, 3, 3, 1;
  EXPECT_TRUE(stan::math::check_symmetric("checkSymmetric(%1%)",
                                          y, "y", &result));
  EXPECT_TRUE(stan::math::check_symmetric("checkSymmetric(%1%)",
                                          y, "y"));

  y(0,1) = 3.5;
  EXPECT_THROW(stan::math::check_symmetric("checkSymmetric(%1%)", y, "y", &result), 
               std::domain_error);
  EXPECT_THROW(stan::math::check_symmetric("checkSymmetric(%1%)", y, "y"),
               std::domain_error);
}
