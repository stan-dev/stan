#include <stan/math/matrix/qr_Q.hpp>
#include <stan/math/matrix/qr_R.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/transpose.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>

using stan::agrad::var;

TEST(AgradFwdMatrixQrR, fd) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fd;
  matrix_fd m0(0,0);
  matrix_d m2(3,2);
  matrix_fd m1(3,2);
  m1 << 1, 2, 3, 4, 5, 6;
  m2 << 1, 2, 3, 4, 5, 6;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;
  m1(2,0).d_ = 1.0;
  m1(2,1).d_ = 1.0;

  using stan::math::qr_R;
  using stan::math::transpose;
  EXPECT_THROW(qr_R(m0),std::domain_error);
  EXPECT_NO_THROW(qr_R(m1));
  EXPECT_THROW(qr_R(transpose(m1)),std::domain_error);

  matrix_fd res = qr_R(m1);
  matrix_d res2 = qr_R(m2);

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(res2(i,j), res(i,j).val_);

  EXPECT_FLOAT_EQ(1.5212777, res(0,0).d_);
  EXPECT_FLOAT_EQ(1.6371845, res(0,1).d_);
  EXPECT_FLOAT_EQ(0, res(1,0).d_);
  EXPECT_FLOAT_EQ(-0.21293451, res(1,1).d_);
  EXPECT_FLOAT_EQ(0, res(2,0).d_);
  EXPECT_FLOAT_EQ(0, res(2,1).d_);
}

TEST(AgradFwdMatrixQrR, ffd) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffd;
  matrix_ffd m0(0,0);
  matrix_d m2(3,2);
  matrix_ffd m1(3,2);
  m1 << 1, 2, 3, 4, 5, 6;
  m2 << 1, 2, 3, 4, 5, 6;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;
  m1(2,0).d_ = 1.0;
  m1(2,1).d_ = 1.0;

  using stan::math::qr_R;
  using stan::math::transpose;
  EXPECT_THROW(qr_R(m0),std::domain_error);
  EXPECT_NO_THROW(qr_R(m1));
  EXPECT_THROW(qr_R(transpose(m1)),std::domain_error);

  matrix_ffd res = qr_R(m1);
  matrix_d res2 = qr_R(m2);
  
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(res2(i,j), res(i,j).val_.val_);

  EXPECT_FLOAT_EQ(1.5212777, res(0,0).d_.val_);
  EXPECT_FLOAT_EQ(1.6371845, res(0,1).d_.val_);
  EXPECT_FLOAT_EQ(0, res(1,0).d_.val_);
  EXPECT_FLOAT_EQ(-0.21293451, res(1,1).d_.val_);
  EXPECT_FLOAT_EQ(0, res(2,0).d_.val_);
  EXPECT_FLOAT_EQ(0, res(2,1).d_.val_);
}

TEST(AgradFwdMatrixQrR, fv1) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  matrix_fv m0(0,0);
  matrix_d m2(3,2);
  matrix_fv m1(3,2);
  m1 << 1, 2, 3, 4, 5, 6;
  m2 << 1, 2, 3, 4, 5, 6;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;
  m1(2,0).d_ = 1.0;
  m1(2,1).d_ = 1.0;

  using stan::math::qr_R;
  using stan::math::transpose;
  EXPECT_THROW(qr_R(m0),std::domain_error);
  EXPECT_NO_THROW(qr_R(m1));
  EXPECT_THROW(qr_R(transpose(m1)),std::domain_error);

  matrix_fv res = qr_R(m1);
  matrix_d res2 = qr_R(m2);
  
  std::vector<var> vars;

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(res2(i,j), res(i,j).val_.val());
      vars.push_back(m1(i,j).val_);
    }

  EXPECT_FLOAT_EQ(1.5212777, res(0,0).d_.val());
  EXPECT_FLOAT_EQ(1.6371845, res(0,1).d_.val());
  EXPECT_FLOAT_EQ(0, res(1,0).d_.val());
  EXPECT_FLOAT_EQ(-0.21293451, res(1,1).d_.val());
  EXPECT_FLOAT_EQ(0, res(2,0).d_.val());
  EXPECT_FLOAT_EQ(0, res(2,1).d_.val());

  std::vector<double> grads;
  res(0,0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.16903085, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.50709254, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(0.84515423, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);

}

TEST(AgradFwdMatrixQrR, fv2) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  matrix_fv m0(0,0);
  matrix_d m2(3,2);
  matrix_fv m1(3,2);
  m1 << 1, 2, 3, 4, 5, 6;
  m2 << 1, 2, 3, 4, 5, 6;
  m1(0,0).d_ = 1.0;
  m1(0,1).d_ = 1.0;
  m1(1,0).d_ = 1.0;
  m1(1,1).d_ = 1.0;
  m1(2,0).d_ = 1.0;
  m1(2,1).d_ = 1.0;

  using stan::math::qr_R;
  using stan::math::transpose;

  matrix_fv res = qr_R(m1);
  matrix_d res2 = qr_R(m2);
  
  std::vector<var> vars;

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(res2(i,j), res(i,j).val_.val());
      vars.push_back(m1(i,j).val_);
    }

  std::vector<double> grads;
  res(0,0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.12556578, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.038635623, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(-0.048294529, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
}

TEST(AgradFwdMatrixQrR, ffv1) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  matrix_ffv m0(0,0);
  matrix_d m2(3,2);
  matrix_ffv m1(3,2);
  m1 << 1, 2, 3, 4, 5, 6;
  m2 << 1, 2, 3, 4, 5, 6;
  m1(0,0).d_.val_ = 1.0;
  m1(0,1).d_.val_ = 1.0;
  m1(1,0).d_.val_ = 1.0;
  m1(1,1).d_.val_ = 1.0;
  m1(2,0).d_.val_ = 1.0;
  m1(2,1).d_.val_ = 1.0;

  using stan::math::qr_R;
  using stan::math::transpose;
  EXPECT_THROW(qr_R(m0),std::domain_error);
  EXPECT_NO_THROW(qr_R(m1));
  EXPECT_THROW(qr_R(transpose(m1)),std::domain_error);

  matrix_ffv res = qr_R(m1);
  matrix_d res2 = qr_R(m2);
  
  std::vector<var> vars;

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(res2(i,j), res(i,j).val_.val_.val());
      vars.push_back(m1(i,j).val_.val_);
    }

  EXPECT_FLOAT_EQ(1.5212777, res(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(1.6371845, res(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(0, res(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(-0.21293451, res(1,1).d_.val_.val());
  EXPECT_FLOAT_EQ(0, res(2,0).d_.val_.val());
  EXPECT_FLOAT_EQ(0, res(2,1).d_.val_.val());

  std::vector<double> grads;
  res(0,0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.16903085, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.50709254, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(0.84515423, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);

}

TEST(AgradFwdMatrixQrR, ffv2) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  matrix_ffv m0(0,0);
  matrix_d m2(3,2);
  matrix_ffv m1(3,2);
  m1 << 1, 2, 3, 4, 5, 6;
  m2 << 1, 2, 3, 4, 5, 6;
  m1(0,0).d_.val_ = 1.0;
  m1(0,1).d_.val_ = 1.0;
  m1(1,0).d_.val_ = 1.0;
  m1(1,1).d_.val_ = 1.0;
  m1(2,0).d_.val_ = 1.0;
  m1(2,1).d_.val_ = 1.0;

  using stan::math::qr_R;
  using stan::math::transpose;

  matrix_ffv res = qr_R(m1);
  matrix_d res2 = qr_R(m2);
  
  std::vector<var> vars;

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(res2(i,j), res(i,j).val_.val_.val());
      vars.push_back(m1(i,j).val_.val_);
    }

  std::vector<double> grads;
  res(0,0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.12556578, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.038635623, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(-0.048294529, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
}

TEST(AgradFwdMatrixQrR, ffv3) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  matrix_ffv m0(0,0);
  matrix_d m2(3,2);
  matrix_ffv m1(3,2);
  m1 << 1, 2, 3, 4, 5, 6;
  m2 << 1, 2, 3, 4, 5, 6;
  m1(0,0).d_.val_ = 1.0;
  m1(0,1).d_.val_ = 1.0;
  m1(1,0).d_.val_ = 1.0;
  m1(1,1).d_.val_ = 1.0;
  m1(2,0).d_.val_ = 1.0;
  m1(2,1).d_.val_ = 1.0;
  m1(0,0).val_.d_ = 1.0;
  m1(0,1).val_.d_ = 1.0;
  m1(1,0).val_.d_ = 1.0;
  m1(1,1).val_.d_ = 1.0;
  m1(2,0).val_.d_ = 1.0;
  m1(2,1).val_.d_ = 1.0;

  using stan::math::qr_R;
  using stan::math::transpose;

  matrix_ffv res = qr_R(m1);
  matrix_d res2 = qr_R(m2);
  
  std::vector<var> vars;

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(res2(i,j), res(i,j).val_.val_.val());
      vars.push_back(m1(i,j).val_.val_);
    }

  std::vector<double> grads;
  res(0,0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0.12556578, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0.038635623, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(-0.048294529, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
}

TEST(AgradFwdMatrixQrR, ffv4) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  matrix_ffv m0(0,0);
  matrix_d m2(3,2);
  matrix_ffv m1(3,2);
  m1 << 1, 2, 3, 4, 5, 6;
  m2 << 1, 2, 3, 4, 5, 6;
  m1(0,0).d_.val_ = 1.0;
  m1(0,1).d_.val_ = 1.0;
  m1(1,0).d_.val_ = 1.0;
  m1(1,1).d_.val_ = 1.0;
  m1(2,0).d_.val_ = 1.0;
  m1(2,1).d_.val_ = 1.0;
  m1(0,0).val_.d_ = 1.0;
  m1(0,1).val_.d_ = 1.0;
  m1(1,0).val_.d_ = 1.0;
  m1(1,1).val_.d_ = 1.0;
  m1(2,0).val_.d_ = 1.0;
  m1(2,1).val_.d_ = 1.0;

  using stan::math::qr_R;
  using stan::math::transpose;

  matrix_ffv res = qr_R(m1);
  matrix_d res2 = qr_R(m2);
  
  std::vector<var> vars;

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(res2(i,j), res(i,j).val_.val_.val());
      vars.push_back(m1(i,j).val_.val_);
    }

  std::vector<double> grads;
  res(0,0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-0.067888312, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(-0.029804623, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(0.0082790619, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
}

