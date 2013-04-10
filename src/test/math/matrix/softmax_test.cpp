#include <stan/math/matrix/softmax.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix,softmax) {
  using stan::math::softmax;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  
  Matrix<double,Dynamic,1> x(1);
  x << 0.0;
  
  Matrix<double,Dynamic,1> theta = softmax(x);
  EXPECT_EQ(1,theta.size());
  EXPECT_FLOAT_EQ(1.0,theta[0]);

  Matrix<double,Dynamic,1> x2(2);
  x2 << -1.0, 1.0;
  Matrix<double,Dynamic,1> theta2 = softmax(x2);
  EXPECT_EQ(2,theta2.size());
  EXPECT_FLOAT_EQ(exp(-1)/(exp(-1) + exp(1)), theta2[0]);
  EXPECT_FLOAT_EQ(exp(1)/(exp(-1) + exp(1)), theta2[1]);

  Matrix<double,Dynamic,1> x3(3);
  x3 << -1.0, 1.0, 10.0;
  Matrix<double,Dynamic,1> theta3 = softmax(x3);
  EXPECT_EQ(3,theta3.size());
  EXPECT_FLOAT_EQ(exp(-1)/(exp(-1) + exp(1) + exp(10.0)), theta3[0]);
  EXPECT_FLOAT_EQ(exp(1)/(exp(-1) + exp(1) + exp(10.0)), theta3[1]);
  EXPECT_FLOAT_EQ(exp(10)/(exp(-1) + exp(1) + exp(10.0)), theta3[2]);
}
TEST(MathMatrix,softmax_exception) {
  using stan::math::softmax;
  stan::math::vector_d v0;

  EXPECT_THROW(softmax(v0),std::domain_error);
}  
