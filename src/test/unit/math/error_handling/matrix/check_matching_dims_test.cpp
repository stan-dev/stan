#include <stan/math/error_handling/matrix/check_matching_dims.hpp>
#include <gtest/gtest.h>

TEST(MathErrorHandlingMatrix, checkMatchingDimsMatrix) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> x;
  double result;
  
  y.resize(3,3);
  x.resize(3,3);
  EXPECT_TRUE(stan::math::check_matching_dims("checkMatchingDims(%1%)",x,"x",
                                             y, "y", &result));
  x.resize(0,0);
  y.resize(0,0);
  EXPECT_TRUE(stan::math::check_matching_dims("checkMatchingDims(%1%)",x,"x",
                                             y, "y", &result));

  y.resize(1,2);
  EXPECT_THROW(stan::math::check_matching_dims("checkMatchingDims(%1%)",x,"x",
                                               y, "y",&result), 
               std::domain_error);

  x.resize(2,1);
  EXPECT_THROW(stan::math::check_matching_dims("checkMatchingDims(%1%)",x,"x",
                                               y, "y",&result), 
               std::domain_error);
}
