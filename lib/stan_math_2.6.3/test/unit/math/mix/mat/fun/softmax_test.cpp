#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/softmax.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>

TEST(AgradMixMatrixSoftmax,fv_1stDeriv) {
  using stan::math::softmax;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::vector_fv;
  using stan::math::fvar;
  using stan::math::var;

  EXPECT_THROW(softmax(vector_fv()),std::invalid_argument);
  
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
TEST(AgradMixMatrixSoftmax,fv_2ndDeriv) {
  using stan::math::softmax;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::vector_fv;
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixMatrixSoftmax,ffv_1stDeriv) {
  using stan::math::softmax;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::vector_ffv;
  using stan::math::fvar;
  using stan::math::var;

  EXPECT_THROW(softmax(vector_ffv()),std::invalid_argument);
  
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
TEST(AgradMixMatrixSoftmax,ffv_2ndDeriv_1) {
  using stan::math::softmax;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::vector_ffv;
  using stan::math::fvar;
  using stan::math::var;

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

TEST(AgradMixMatrixSoftmax,ffv_2ndDeriv_2) {
  using stan::math::softmax;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::vector_ffv;
  using stan::math::fvar;
  using stan::math::var;

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

TEST(AgradMixMatrixSoftmax,ffv_3rdDeriv) {
  using stan::math::softmax;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::vector_ffv;
  using stan::math::fvar;
  using stan::math::var;

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
