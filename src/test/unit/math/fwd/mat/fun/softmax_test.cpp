#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/softmax.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>

TEST(AgradFwdMatrixSoftmax,fd) {
  using stan::math::softmax;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::vector_fd;
  using stan::math::fvar;

  EXPECT_THROW(softmax(vector_fd()),std::invalid_argument);
  
  Matrix<fvar<double>,Dynamic,1> x(1);
  x << 0.0;
  x(0).d_ = 1.0;
  
  Matrix<fvar<double>,Dynamic,1> theta = stan::math::softmax(x);
  EXPECT_EQ(1,theta.size());
  EXPECT_FLOAT_EQ(1.0,theta[0].val_);
  EXPECT_FLOAT_EQ(0.0,theta[0].d_);

  Matrix<fvar<double>,Dynamic,1> x2(2);
  x2 << -1.0, 1.0;
   x2(0).d_ = 2.0;
   x2(1).d_ = 1.0;

  Matrix<fvar<double>,Dynamic,1> theta2 = softmax(x2);
  EXPECT_EQ(2,theta2.size());
  EXPECT_FLOAT_EQ(exp(-1)/(exp(-1) + exp(1)), theta2[0].val_);
  EXPECT_FLOAT_EQ(exp(1)/(exp(-1) + exp(1)), theta2[1].val_);
  EXPECT_FLOAT_EQ(0.10499358, theta2[0].d_);
  EXPECT_FLOAT_EQ(-0.10499358, theta2[1].d_);

  Matrix<fvar<double>,Dynamic,1> x3(3);
  x3 << -1.0, 1.0, 10.0;
   x3(0).d_ = 2.0;
   x3(1).d_ = 1.0;
   x3(2).d_ = 1.0;

  Matrix<fvar<double>,Dynamic,1> theta3 = softmax(x3);
  EXPECT_EQ(3,theta3.size());
  EXPECT_FLOAT_EQ(exp(-1)/(exp(-1) + exp(1) + exp(10.0)), theta3[0].val_);
  EXPECT_FLOAT_EQ(exp(1)/(exp(-1) + exp(1) + exp(10.0)), theta3[1].val_);
  EXPECT_FLOAT_EQ(exp(10)/(exp(-1) + exp(1) + exp(10.0)), theta3[2].val_);
  EXPECT_FLOAT_EQ(1.6699081e-05, theta3[0].d_);
  EXPECT_FLOAT_EQ(-2.0605762e-09, theta3[1].d_);
  EXPECT_FLOAT_EQ(-1.6697022e-05, theta3[2].d_);
}
TEST(AgradFwdMatrixSoftmax,ffd) {
  using stan::math::softmax;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::vector_ffd;
  using stan::math::fvar;

  EXPECT_THROW(softmax(vector_ffd()),std::invalid_argument);
  
  Matrix<fvar<fvar<double> >,Dynamic,1> x(1);
  x << 0.0;
   x(0).d_ = 1.0;
  
  Matrix<fvar<fvar<double> >,Dynamic,1> theta = softmax(x);
  EXPECT_EQ(1,theta.size());
  EXPECT_FLOAT_EQ(1.0,theta[0].val_.val());
  EXPECT_FLOAT_EQ(0.0,theta[0].d_.val());

  Matrix<fvar<fvar<double> >,Dynamic,1> x2(2);
  x2 << -1.0, 1.0;
   x2(0).d_ = 2.0;
   x2(1).d_ = 1.0;

  Matrix<fvar<fvar<double> >,Dynamic,1> theta2 = softmax(x2);
  EXPECT_EQ(2,theta2.size());
  EXPECT_FLOAT_EQ(exp(-1)/(exp(-1) + exp(1)), theta2[0].val_.val());
  EXPECT_FLOAT_EQ(exp(1)/(exp(-1) + exp(1)), theta2[1].val_.val());
  EXPECT_FLOAT_EQ(0.10499358, theta2[0].d_.val());
  EXPECT_FLOAT_EQ(-0.10499358, theta2[1].d_.val());

  Matrix<fvar<fvar<double> >,Dynamic,1> x3(3);
  x3 << -1.0, 1.0, 10.0;
   x3(0).d_ = 2.0;
   x3(1).d_ = 1.0;
   x3(2).d_ = 1.0;

  Matrix<fvar<fvar<double> >,Dynamic,1> theta3 = softmax(x3);
  EXPECT_EQ(3,theta3.size());
  EXPECT_FLOAT_EQ(exp(-1)/(exp(-1) + exp(1) + exp(10.0)), theta3[0].val_.val());
  EXPECT_FLOAT_EQ(exp(1)/(exp(-1) + exp(1) + exp(10.0)), theta3[1].val_.val());
  EXPECT_FLOAT_EQ(exp(10)/(exp(-1) + exp(1) + exp(10.0)), theta3[2].val_.val());
  EXPECT_FLOAT_EQ(1.6699081e-05, theta3[0].d_.val());
  EXPECT_FLOAT_EQ(-2.0605762e-09, theta3[1].d_.val());
  EXPECT_FLOAT_EQ(-1.6697022e-05, theta3[2].d_.val());
}
