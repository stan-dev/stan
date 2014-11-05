#include <gtest/gtest.h>
#include <stan/agrad/rev/matrix.hpp>
#include <stan/agrad/fwd/matrix.hpp>
#include <stan/prob/distributions/multivariate/continuous/multi_normal_prec.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;
using std::vector;

TEST(ProbDistributionsMultiNormalPrec,NotVectorized) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  Matrix<double,Dynamic,Dynamic> L = Sigma.inverse();
  EXPECT_FLOAT_EQ(-11.73908, stan::prob::multi_normal_prec_log(y,mu,L));
}

TEST(ProbDistributionsMultiNormalPrec,Vectorized) {
  vector< Matrix<double,Dynamic,1> > vec_y(2);
  vector< Matrix<double,1,Dynamic> > vec_y_t(2);
  Matrix<double,Dynamic,1> y(3);
  Matrix<double,1,Dynamic> y_t(3);
  y << 2.0, -2.0, 11.0;
  vec_y[0] = y;
  vec_y_t[0] = y;
  y << 4.0, -2.0, 1.0;
  vec_y[1] = y;
  vec_y_t[1] = y;
  y_t = y;
  
  vector< Matrix<double,Dynamic,1> > vec_mu(2);
  vector< Matrix<double,1,Dynamic> > vec_mu_t(2);
  Matrix<double,Dynamic,1> mu(3);
  Matrix<double,1,Dynamic> mu_t(3);
  mu << 1.0, -1.0, 3.0;
  vec_mu[0] = mu;
  vec_mu_t[0] = mu;
  mu << 2.0, -1.0, 4.0;
  vec_mu[1] = mu;
  vec_mu_t[1] = mu;
  mu_t = mu;
  
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 10.0, -3.0, 0.0,
    -3.0,  5.0, 0.0,
    0.0, 0.0, 5.0;
  Sigma = Sigma.inverse();
    
  //y and mu vectorized
  EXPECT_FLOAT_EQ(-11.928077-6.5378327, stan::prob::multi_normal_prec_log(vec_y,vec_mu,Sigma));
  EXPECT_FLOAT_EQ(-11.928077-6.5378327, stan::prob::multi_normal_prec_log(vec_y_t,vec_mu,Sigma));
  EXPECT_FLOAT_EQ(-11.928077-6.5378327, stan::prob::multi_normal_prec_log(vec_y,vec_mu_t,Sigma));
  EXPECT_FLOAT_EQ(-11.928077-6.5378327, stan::prob::multi_normal_prec_log(vec_y_t,vec_mu_t,Sigma));

  //y vectorized
  EXPECT_FLOAT_EQ(-10.44027-6.537833, stan::prob::multi_normal_prec_log(vec_y,mu,Sigma));
  EXPECT_FLOAT_EQ(-10.44027-6.537833, stan::prob::multi_normal_prec_log(vec_y_t,mu,Sigma));
  EXPECT_FLOAT_EQ(-10.44027-6.537833, stan::prob::multi_normal_prec_log(vec_y,mu_t,Sigma));
  EXPECT_FLOAT_EQ(-10.44027-6.537833, stan::prob::multi_normal_prec_log(vec_y_t,mu_t,Sigma));

  //mu vectorized
  EXPECT_FLOAT_EQ(-6.26954-6.537833, stan::prob::multi_normal_prec_log(y,vec_mu,Sigma));
  EXPECT_FLOAT_EQ(-6.26954-6.537833, stan::prob::multi_normal_prec_log(y_t,vec_mu,Sigma));
  EXPECT_FLOAT_EQ(-6.26954-6.537833, stan::prob::multi_normal_prec_log(y,vec_mu_t,Sigma));
  EXPECT_FLOAT_EQ(-6.26954-6.537833, stan::prob::multi_normal_prec_log(y_t,vec_mu_t,Sigma));
}
TEST(ProbDistributionsMultiNormalPrec,MultiNormalVar) {
  using stan::agrad::var;
  Matrix<var,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<var,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<var,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  Matrix<var,Dynamic,Dynamic> L = Sigma.inverse();
  EXPECT_FLOAT_EQ(-11.73908, stan::prob::multi_normal_prec_log(y,mu,L).val());
}
/*
TEST(ProbDistributionsMultiNormalPrec,MultiNormalOneRow) {
  Matrix<double,Dynamic,Dynamic> y(1,3);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  Matrix<double,Dynamic,Dynamic> L = Sigma.inverse();
  EXPECT_FLOAT_EQ(-11.73908, stan::prob::multi_normal_prec_log(y,mu,L));
}

TEST(ProbDistributionsMultiNormalPrec,MultiNormalMultiRow) {
  Matrix<double,Dynamic,Dynamic> y(2,3);
  y << 2.0, -2.0, 11.0,
       4.0, -4.0, 22.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  Matrix<double,Dynamic,Dynamic> L = Sigma.inverse();
  EXPECT_FLOAT_EQ(-54.2152, stan::prob::multi_normal_prec_log(y,mu,L));
}
*/

TEST(ProbDistributionsMultiNormalPrec,fvar_double) {
  using stan::agrad::fvar;
  Matrix<fvar<double>,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<fvar<double>,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<fvar<double>,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;

  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  for (int i = 0; i < 3; i++) {
    y(i).d_ = 1.0;
    mu(i).d_ = 1.0;
    for (int j = 0; j < 3; j++)
      Sigma(i,j).d_ = 1.0;
  }

  Matrix<fvar<double>,Dynamic,Dynamic> L = Sigma.inverse();
  EXPECT_FLOAT_EQ(-11.73908, stan::prob::multi_normal_prec_log(y,mu,L).val_);
  EXPECT_FLOAT_EQ(0.54899865, stan::prob::multi_normal_prec_log(y,mu,L).d_);
}

TEST(ProbDistributionsMultiNormalPrec,fvar_var) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  Matrix<fvar<var>,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<fvar<var>,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<fvar<var>,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;

  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  for (int i = 0; i < 3; i++) {
    y(i).d_ = 1.0;
    mu(i).d_ = 1.0;
    for (int j = 0; j < 3; j++)
      Sigma(i,j).d_ = 1.0;
  }

  Matrix<fvar<var>,Dynamic,Dynamic> L = Sigma.inverse();
  EXPECT_FLOAT_EQ(-11.73908, stan::prob::multi_normal_prec_log(y,mu,L).val_.val());
  EXPECT_FLOAT_EQ(0.54899865, stan::prob::multi_normal_prec_log(y,mu,L).d_.val());
}

TEST(ProbDistributionsMultiNormalPrec,fvar_fvar_double) {
  using stan::agrad::fvar;
  Matrix<fvar<fvar<double> >,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<fvar<fvar<double> >,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<fvar<fvar<double> >,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;

  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  for (int i = 0; i < 3; i++) {
    y(i).d_.val_ = 1.0;
    mu(i).d_.val_ = 1.0;
    for (int j = 0; j < 3; j++)
      Sigma(i,j).d_.val_ = 1.0;
  }

  Matrix<fvar<fvar<double> >,Dynamic,Dynamic> L = Sigma.inverse();
  EXPECT_FLOAT_EQ(-11.73908, stan::prob::multi_normal_prec_log(y,mu,L).val_.val_);
  EXPECT_FLOAT_EQ(0.54899865, stan::prob::multi_normal_prec_log(y,mu,L).d_.val_);
}

TEST(ProbDistributionsMultiNormalPrec,fvar_fvar_var) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  Matrix<fvar<fvar<var> >,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<fvar<fvar<var> >,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<fvar<fvar<var> >,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;

  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  for (int i = 0; i < 3; i++) {
    y(i).d_.val_ = 1.0;
    mu(i).d_.val_ = 1.0;
    for (int j = 0; j < 3; j++)
      Sigma(i,j).d_.val_ = 1.0;
  }

  Matrix<fvar<fvar<var> >,Dynamic,Dynamic> L = Sigma.inverse();
  EXPECT_FLOAT_EQ(-11.73908, stan::prob::multi_normal_prec_log(y,mu,L).val_.val_.val());
  EXPECT_FLOAT_EQ(0.54899865, stan::prob::multi_normal_prec_log(y,mu,L).d_.val_.val());
}
