#include <stan/math/error_handling/matrix/check_square.hpp>
#include <gtest/gtest.h>

TEST(MathErrorHandlingMatrix, checkSquareMatrix) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  double result;
  
  y.resize(3,3);
  EXPECT_TRUE(stan::math::check_square("checkSquareMatrix(%1%)",
                                           y, "y", &result));
  EXPECT_TRUE(stan::math::check_square("checkSquareMatrix(%1%)",
                                           y, "y"));

  y.resize(3, 2);
  EXPECT_THROW(stan::math::check_square("checkSquareMatrix(%1%)", y, "y", &result), 
               std::domain_error);
  EXPECT_THROW(stan::math::check_square("checkSquareMatrix(%1%)", y, "y"),
               std::domain_error);
}
