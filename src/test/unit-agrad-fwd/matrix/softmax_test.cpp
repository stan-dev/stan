#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/fwd/matrix/softmax.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>

TEST(AgradFwdMatrixSoftmax,fd) {
  using stan::math::softmax;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::agrad::vector_fd;
  using stan::agrad::fvar;

  EXPECT_THROW(softmax(vector_fd()),std::domain_error);
  
  Matrix<fvar<double>,Dynamic,1> x(1);
  x << 0.0;
  x(0).d_ = 1.0;
  
  Matrix<fvar<double>,Dynamic,1> theta = stan::agrad::softmax(x);
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
TEST(AgradFwdMatrixSoftmax,fv_1stDeriv) {
  using stan::math::softmax;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  EXPECT_THROW(softmax(vector_fv()),std::domain_error);
  
  Matrix<fvar<var>,Dynamic,1> x(1);
  x << 0.0;
   x(0).d_ = 1.0;
  
  Matrix<fvar<var>,Dynamic,1> theta = softmax(x);
  EXPECT_EQ(1,theta.size());
  EXPECT_FLOAT_EQ(1.0,theta[0].val_.val());
  EXPECT_FLOAT_EQ(0.0,theta[0].d_.val());

  Matrix<fvar<var>,Dynamic,1> x2(2);
  x2 << -1.0, 1.0;
   x2(0).d_ = 2.0;
   x2(1).d_ = 1.0;

  Matrix<fvar<var>,Dynamic,1> theta2 = softmax(x2);
  EXPECT_EQ(2,theta2.size());
  EXPECT_FLOAT_EQ(exp(-1)/(exp(-1) + exp(1)), theta2[0].val_.val());
  EXPECT_FLOAT_EQ(exp(1)/(exp(-1) + exp(1)), theta2[1].val_.val());
  EXPECT_FLOAT_EQ(0.10499358, theta2[0].d_.val());
  EXPECT_FLOAT_EQ(-0.10499358, theta2[1].d_.val());

  Matrix<fvar<var>,Dynamic,1> x3(3);
  x3 << -1.0, 1.0, 10.0;
   x3(0).d_ = 2.0;
   x3(1).d_ = 1.0;
   x3(2).d_ = 1.0;

  Matrix<fvar<var>,Dynamic,1> theta3 = softmax(x3);
  EXPECT_EQ(3,theta3.size());
  EXPECT_FLOAT_EQ(exp(-1)/(exp(-1) + exp(1) + exp(10.0)), theta3[0].val_.val());
  EXPECT_FLOAT_EQ(exp(1)/(exp(-1) + exp(1) + exp(10.0)), theta3[1].val_.val());
  EXPECT_FLOAT_EQ(exp(10)/(exp(-1) + exp(1) + exp(10.0)), theta3[2].val_.val());
  EXPECT_FLOAT_EQ(1.6699081e-05, theta3[0].d_.val());
  EXPECT_FLOAT_EQ(-2.0605762e-09, theta3[1].d_.val());
  EXPECT_FLOAT_EQ(-1.6697022e-05, theta3[2].d_.val());

  AVEC q = createAVEC(x3(0).val(),x3(1).val(),x3(2).val());
  VEC h;
  theta3[0].val_.grad(q,h);
  EXPECT_FLOAT_EQ(1.6699081e-05,h[0]);
  EXPECT_FLOAT_EQ(-2.0605762e-09,h[1]);
  EXPECT_FLOAT_EQ(-1.6697022e-05,h[2]);
}
TEST(AgradFwdMatrixSoftmax,fv_2ndDeriv) {
  using stan::math::softmax;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  Matrix<fvar<var>,Dynamic,1> x3(3);
  x3 << -1.0, 1.0, 10.0;
   x3(0).d_ = 2.0;
   x3(1).d_ = 1.0;
   x3(2).d_ = 1.0;

  Matrix<fvar<var>,Dynamic,1> theta3 = softmax(x3);

  AVEC q = createAVEC(x3(0).val(),x3(1).val(),x3(2).val());
  VEC h;
  theta3[0].d_.grad(q,h);
  EXPECT_FLOAT_EQ(1.6698525e-05,h[0]);
  EXPECT_FLOAT_EQ(-2.0605073e-09,h[1]);
  EXPECT_FLOAT_EQ(-1.6696464e-05,h[2]);
}
TEST(AgradFwdMatrixSoftmax,ffd) {
  using stan::math::softmax;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::agrad::vector_ffd;
  using stan::agrad::fvar;

  EXPECT_THROW(softmax(vector_ffd()),std::domain_error);
  
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
TEST(AgradFwdMatrixSoftmax,ffv_1stDeriv) {
  using stan::math::softmax;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  EXPECT_THROW(softmax(vector_ffv()),std::domain_error);
  
  Matrix<fvar<fvar<var> >,Dynamic,1> x(1);
  x << 0.0;
   x(0).d_ = 1.0;
  
  Matrix<fvar<fvar<var> >,Dynamic,1> theta = softmax(x);
  EXPECT_EQ(1,theta.size());
  EXPECT_FLOAT_EQ(1.0,theta[0].val_.val().val());
  EXPECT_FLOAT_EQ(0.0,theta[0].d_.val().val());

  Matrix<fvar<fvar<var> >,Dynamic,1> x2(2);
  x2 << -1.0, 1.0;
   x2(0).d_ = 2.0;
   x2(1).d_ = 1.0;

  Matrix<fvar<fvar<var> >,Dynamic,1> theta2 = softmax(x2);
  EXPECT_EQ(2,theta2.size());
  EXPECT_FLOAT_EQ(exp(-1)/(exp(-1) + exp(1)), theta2[0].val_.val().val());
  EXPECT_FLOAT_EQ(exp(1)/(exp(-1) + exp(1)), theta2[1].val_.val().val());
  EXPECT_FLOAT_EQ(0.10499358, theta2[0].d_.val().val());
  EXPECT_FLOAT_EQ(-0.10499358, theta2[1].d_.val().val());

  Matrix<fvar<fvar<var> >,Dynamic,1> x3(3);
  x3 << -1.0, 1.0, 10.0;
   x3(0).d_ = 2.0;
   x3(1).d_ = 1.0;
   x3(2).d_ = 1.0;

  Matrix<fvar<fvar<var> >,Dynamic,1> theta3 = softmax(x3);
  EXPECT_EQ(3,theta3.size());
  EXPECT_FLOAT_EQ(exp(-1)/(exp(-1) + exp(1) + exp(10.0)), theta3[0].val_.val().val());
  EXPECT_FLOAT_EQ(exp(1)/(exp(-1) + exp(1) + exp(10.0)), theta3[1].val_.val().val());
  EXPECT_FLOAT_EQ(exp(10)/(exp(-1) + exp(1) + exp(10.0)), theta3[2].val_.val().val());
  EXPECT_FLOAT_EQ(1.6699081e-05, theta3[0].d_.val().val());
  EXPECT_FLOAT_EQ(-2.0605762e-09, theta3[1].d_.val().val());
  EXPECT_FLOAT_EQ(-1.6697022e-05, theta3[2].d_.val().val());

  AVEC q = createAVEC(x3(0).val().val(),x3(1).val().val(),x3(2).val().val());
  VEC h;
  theta3[0].val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1.6699081e-05,h[0]);
  EXPECT_FLOAT_EQ(-2.0605762e-09,h[1]);
  EXPECT_FLOAT_EQ(-1.6697022e-05,h[2]);
}
TEST(AgradFwdMatrixSoftmax,ffv_2ndDeriv_1) {
  using stan::math::softmax;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  Matrix<fvar<fvar<var> >,Dynamic,1> x3(3);
  x3 << -1.0, 1.0, 10.0;
   x3(0).d_ = 2.0;
   x3(1).d_ = 1.0;
   x3(2).d_ = 1.0;

  Matrix<fvar<fvar<var> >,Dynamic,1> theta3 = softmax(x3);

  AVEC q = createAVEC(x3(0).val().val(),x3(1).val().val(),x3(2).val().val());
  VEC h;
  theta3[0].val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}

TEST(AgradFwdMatrixSoftmax,ffv_2ndDeriv_2) {
  using stan::math::softmax;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  Matrix<fvar<fvar<var> >,Dynamic,1> x3(3);
  x3 << -1.0, 1.0, 10.0;
   x3(0).d_ = 2.0;
   x3(1).d_ = 1.0;
   x3(2).d_ = 1.0;

  Matrix<fvar<fvar<var> >,Dynamic,1> theta3 = softmax(x3);

  AVEC q = createAVEC(x3(0).val().val(),x3(1).val().val(),x3(2).val().val());
  VEC h;
  theta3[0].d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1.6698525e-05,h[0]);
  EXPECT_FLOAT_EQ(-2.0605073e-09,h[1]);
  EXPECT_FLOAT_EQ(-1.6696464e-05,h[2]);
}

TEST(AgradFwdMatrixSoftmax,ffv_3rdDeriv) {
  using stan::math::softmax;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  Matrix<fvar<fvar<var> >,Dynamic,1> x3(3);
  x3 << -1.0, 1.0, 10.0;
   x3(0).d_ = 2.0;
   x3(1).d_ = 1.0;
   x3(2).d_ = 1.0;
   x3(0).val_.d_ = 2.0;
   x3(1).val_.d_ = 1.0;
   x3(2).val_.d_ = 1.0;

  Matrix<fvar<fvar<var> >,Dynamic,1> theta3 = softmax(x3);

  AVEC q = createAVEC(x3(0).val().val(),x3(1).val().val(),x3(2).val().val());
  VEC h;
  theta3[0].d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(1.669741e-05,h[0]);
  EXPECT_FLOAT_EQ(-2.0603697e-09,h[1]);
  EXPECT_FLOAT_EQ(-1.6695349e-05,h[2]);
}
