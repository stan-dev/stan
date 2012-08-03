#include <gtest/gtest.h>
#include <stan/math/matrix_error_handling.hpp>
using stan::math::default_policy;

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
TEST(stanMathMatrixErrorHandling, checkCovMatrix) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  double result;
  
  y.resize(3,3);
  y << 2, -1, 0, -1, 2, -1, 0, -1, 2;
  EXPECT_TRUE(stan::math::check_cov_matrix("checkCovMatrix(%1%)",
                                           y, "y", &result));
  EXPECT_TRUE(stan::math::check_cov_matrix("checkCovMatrix(%1%)",
                                           y, "y"));

  y << 1, 2, 3, 2, 1, 2, 3, 2, 1;
  EXPECT_THROW(stan::math::check_cov_matrix("checkCovMatrix(%1%)", y, "y", &result), 
               std::domain_error);
  EXPECT_THROW(stan::math::check_cov_matrix("checkCovMatrix(%1%)", y, "y"),
               std::domain_error);
}

TEST(MathMatrixErrorHandling, checkConsistentSize) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::check_consistent_size;
  using stan::size_of;

  const char* function = "checkConsistentSize(%1%)";
  const char* name1 = "name1";
  
  double result;

  Matrix<double,Dynamic,1> v1(4);
  EXPECT_EQ(4U, size_of(v1));
  EXPECT_TRUE(check_consistent_size(4U, function, v1, name1, &result, default_policy()));
}

TEST(MathMatrixErrorHandling, checkConsistentSizes) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::check_consistent_sizes;
  using stan::size_of;

  const char* function = "testConsSizes(%1%)";
  const char* name1 = "name1";
  const char* name2 = "name2";
  const char* name3 = "name3";
  
  double result;

  Matrix<double,Dynamic,1> v1(4);
  Matrix<double,Dynamic,1> v2(4);
  Matrix<double,Dynamic,1> v3(4);
  EXPECT_EQ(4U, size_of(v1));
  EXPECT_TRUE(check_consistent_sizes(function,v1,v2,v3,name1,name2,name3,&result));
}

