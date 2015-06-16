#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/log_softmax.hpp>
#include <stan/math/fwd/mat/fun/log_sum_exp.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>

TEST(AgradFwdMatrixLogSoftmax,fd) {
  using stan::math::log_softmax;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::vector_fd;
  using stan::math::fvar;

  EXPECT_THROW(softmax(vector_fd()),std::invalid_argument);
  
  Matrix<fvar<double>,Dynamic,1> x(1);
  x << 0.0;
  x(0).d_ = 1.0;
  
  Matrix<fvar<double>,Dynamic,1> theta = stan::math::log_softmax(x);
  EXPECT_EQ(1,theta.size());
  EXPECT_FLOAT_EQ(0.0,theta[0].val_);
  EXPECT_FLOAT_EQ(0.0,theta[0].d_);

  Matrix<fvar<double>,Dynamic,1> x2(2);
  x2 << -1.0, 1.0;
   x2(0).d_ = 2.0;
   x2(1).d_ = 1.0;

  Matrix<fvar<double>,Dynamic,1> theta2 = log_softmax(x2);
  EXPECT_EQ(2,theta2.size());
  EXPECT_FLOAT_EQ(-2.1269281, theta2[0].val_);
  EXPECT_FLOAT_EQ(-0.12692802, theta2[1].val_);
  EXPECT_FLOAT_EQ(0.88079709, theta2[0].d_);
  EXPECT_FLOAT_EQ(-0.11920292, theta2[1].d_);

  Matrix<fvar<double>,Dynamic,1> x3(3);
  x3 << -1.0, 1.0, 10.0;
   x3(0).d_ = 2.0;
   x3(1).d_ = 1.0;
   x3(2).d_ = 1.0;

  Matrix<fvar<double>,Dynamic,1> theta3 = log_softmax(x3);
  EXPECT_EQ(3,theta3.size());
  EXPECT_FLOAT_EQ(-11.00014, theta3[0].val_);
  EXPECT_FLOAT_EQ(-9.0001402, theta3[1].val_);
  EXPECT_FLOAT_EQ(-0.00014010169, theta3[2].val_);
  EXPECT_FLOAT_EQ(0.99998331, theta3[0].d_);
  EXPECT_FLOAT_EQ(-1.6699361e-05, theta3[1].d_);
  EXPECT_FLOAT_EQ(-1.6699361e-05, theta3[2].d_);
}
TEST(AgradFwdMatrixLogSoftmax,ffd) {
  using stan::math::log_softmax;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::vector_ffd;
  using stan::math::fvar;

  EXPECT_THROW(softmax(vector_ffd()),std::invalid_argument);
  
  Matrix<fvar<fvar<double> >,Dynamic,1> x(1);
  x << 0.0;
   x(0).d_ = 1.0;
  
  Matrix<fvar<fvar<double> >,Dynamic,1> theta = log_softmax(x);
  EXPECT_EQ(1,theta.size());
  EXPECT_FLOAT_EQ(0.0,theta[0].val_.val());
  EXPECT_FLOAT_EQ(0.0,theta[0].d_.val());

  Matrix<fvar<fvar<double> >,Dynamic,1> x2(2);
  x2 << -1.0, 1.0;
   x2(0).d_ = 2.0;
   x2(1).d_ = 1.0;

  Matrix<fvar<fvar<double> >,Dynamic,1> theta2 = log_softmax(x2);
  EXPECT_EQ(2,theta2.size());
  EXPECT_FLOAT_EQ(-2.1269281, theta2[0].val_.val());
  EXPECT_FLOAT_EQ(-0.12692802, theta2[1].val_.val());
  EXPECT_FLOAT_EQ(0.88079709, theta2[0].d_.val());
  EXPECT_FLOAT_EQ(-0.11920292, theta2[1].d_.val());

  Matrix<fvar<fvar<double> >,Dynamic,1> x3(3);
  x3 << -1.0, 1.0, 10.0;
   x3(0).d_ = 2.0;
   x3(1).d_ = 1.0;
   x3(2).d_ = 1.0;

  Matrix<fvar<fvar<double> >,Dynamic,1> theta3 = log_softmax(x3);
  EXPECT_EQ(3,theta3.size());
  EXPECT_FLOAT_EQ(-11.00014, theta3[0].val_.val());
  EXPECT_FLOAT_EQ(-9.0001402, theta3[1].val_.val());
  EXPECT_FLOAT_EQ(-0.00014010169, theta3[2].val_.val());
  EXPECT_FLOAT_EQ(0.99998331, theta3[0].d_.val());
  EXPECT_FLOAT_EQ(-1.6699361e-05, theta3[1].d_.val());
  EXPECT_FLOAT_EQ(-1.6699361e-05, theta3[2].d_.val());
}
