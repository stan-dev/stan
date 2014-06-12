#include <stan/math/error_handling/matrix/check_multiplicable.hpp>
#include <gtest/gtest.h>

TEST(MathErrorHandlingMatrix, checkMultiplicableMatrix) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> x;
  double result;
  
  y.resize(3,3);
  x.resize(3,3);
  EXPECT_TRUE(stan::math::check_multiplicable("checkMultiplicable(%1%)",x,"x",
                                             y, "y", &result));
  x.resize(3,2);
  y.resize(2,4);
  EXPECT_TRUE(stan::math::check_multiplicable("checkMultiplicable(%1%)",x,"x",
                                             y, "y", &result));

  y.resize(1,2);
  EXPECT_THROW(stan::math::check_multiplicable("checkMultiplicable(%1%)",x,"x",
                                               y, "y",&result), 
               std::domain_error);

  x.resize(2,2);
  EXPECT_THROW(stan::math::check_multiplicable("checkMultiplicable(%1%)",x,"x",
                                               y, "y",&result), 
               std::domain_error);
}
