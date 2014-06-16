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
