#include <stan/math/error_handling/matrix/check_vector.hpp>
#include <gtest/gtest.h>

TEST(MathErrorHandlingMatrix, checkVectorMatrix) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> x;
  double result;
  
  x.resize(3,3);
  EXPECT_THROW(stan::math::check_vector("checkVector(%1%)",x,"x", &result),
               std::domain_error);
  x.resize(0,0);
  EXPECT_THROW(stan::math::check_vector("checkVector(%1%)",x,"x", &result),
               std::domain_error);

  x.resize(1,5);
  EXPECT_TRUE(stan::math::check_vector("checkVector(%1%)",x,"x", &result));

  x.resize(5,1);
  EXPECT_TRUE(stan::math::check_vector("checkVector(%1%)",x,"x", &result));
}
