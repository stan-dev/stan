#include <gtest/gtest.h>
#include <stan/agrad/rev/matrix.hpp>
#include <stan/agrad/fwd/matrix.hpp>
#include <stan/prob/distributions/multivariate/continuous/matrix_normal.hpp>


using Eigen::Dynamic;
using Eigen::Matrix;

TEST(ProbDistributionsMatrixNormal,MatrixNormalPrec) {
  Matrix<double,Dynamic,Dynamic> mu(3,5);
  mu.setZero();
  
  Matrix<double,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0,
       11.0, 2.0, -5.0, 11.0, 0.0,
       -2.0, 11.0, 2.0, -2.0, -11.0;

  Matrix<double,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
          -3.0,  4.0, 0.0,  0.0, 0.0,
           0.0,  0.0, 5.0,  1.0, 0.0,
           0.0,  0.0, 1.0, 10.0, 0.0,
           0.0,  0.0, 0.0,  0.0, 2.0;

  Matrix<double,Dynamic,Dynamic> D(3,3);
  D << 1.0, 0.5, 0.1,
       0.5, 1.0, 0.2,
       0.1, 0.2, 1.0;
  
  double lp_ref;
  lp_ref = stan::prob::matrix_normal_prec_log(y,mu,D,Sigma);
  EXPECT_FLOAT_EQ(lp_ref,-2132.0748232368409845);
}

TEST(ProbDistributionsMatrixNormal,DefaultPolicySigma) {
  Matrix<double,Dynamic,Dynamic> mu(3,5);
  mu.setZero();
  
  Matrix<double,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0,
  11.0, 2.0, -5.0, 11.0, 0.0,
  -2.0, 11.0, 2.0, -2.0, -11.0;
  
  Matrix<double,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
  -3.0,  4.0, 0.0,  0.0, 0.0,
  0.0,  0.0, 5.0,  1.0, 0.0,
  0.0,  0.0, 1.0, 10.0, 0.0,
  0.0,  0.0, 0.0,  0.0, 2.0;
  
  Matrix<double,Dynamic,Dynamic> D(3,3);
  D << 1.0, 0.5, 0.1,
  0.5, 1.0, 0.2,
  0.1, 0.2, 1.0;
  
  // non-symmetric
  Sigma(0, 1) = -2.5;
  EXPECT_THROW (stan::prob::matrix_normal_prec_log(y,mu,D,Sigma), std::domain_error);
  Sigma(0, 1) = Sigma(1, 0);

  // non-spd
  Sigma(0, 0) = -3.0;
  EXPECT_THROW (stan::prob::matrix_normal_prec_log(y,mu,D,Sigma), std::domain_error);
  Sigma(0, 0) = 9.0;
}

TEST(ProbDistributionsMatrixNormal,DefaultPolicyD) {
  Matrix<double,Dynamic,Dynamic> mu(3,5);
  mu.setZero();
  
  Matrix<double,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0,
  11.0, 2.0, -5.0, 11.0, 0.0,
  -2.0, 11.0, 2.0, -2.0, -11.0;
  
  Matrix<double,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
  -3.0,  4.0, 0.0,  0.0, 0.0,
  0.0,  0.0, 5.0,  1.0, 0.0,
  0.0,  0.0, 1.0, 10.0, 0.0,
  0.0,  0.0, 0.0,  0.0, 2.0;
  
  Matrix<double,Dynamic,Dynamic> D(3,3);
  D << 1.0, 0.5, 0.1,
  0.5, 1.0, 0.2,
  0.1, 0.2, 1.0;
  
  // non-symmetric
  D(0, 1) = -2.5;
  EXPECT_THROW (stan::prob::matrix_normal_prec_log(y,mu,D,Sigma), std::domain_error);
  D(0, 1) = Sigma(1, 0);
  
  // non-spd
  D(0, 0) = -3.0;
  EXPECT_THROW (stan::prob::matrix_normal_prec_log(y,mu,D,Sigma), std::domain_error);
  D(0, 0) = 1.0;
}

TEST(ProbDistributionsMatrixNormal,DefaultPolicyY) {
  Matrix<double,Dynamic,Dynamic> mu(3,5);
  mu.setZero();
  
  Matrix<double,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0,
  11.0, 2.0, -5.0, 11.0, 0.0,
  -2.0, 11.0, 2.0, -2.0, -11.0;
  
  Matrix<double,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
  -3.0,  4.0, 0.0,  0.0, 0.0,
  0.0,  0.0, 5.0,  1.0, 0.0,
  0.0,  0.0, 1.0, 10.0, 0.0,
  0.0,  0.0, 0.0,  0.0, 2.0;
  
  Matrix<double,Dynamic,Dynamic> D(3,3);
  D << 1.0, 0.5, 0.1,
  0.5, 1.0, 0.2,
  0.1, 0.2, 1.0;
  
  // non-finite values
  y(0, 0) = std::numeric_limits<double>::infinity();
  EXPECT_THROW (stan::prob::matrix_normal_prec_log(y,mu,D,Sigma), std::domain_error);
  y(0, 0) = -std::numeric_limits<double>::infinity();
  EXPECT_THROW (stan::prob::matrix_normal_prec_log(y,mu,D,Sigma), std::domain_error);
  y(0,0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW (stan::prob::matrix_normal_prec_log(y,mu,D,Sigma), std::domain_error);
}


TEST(ProbDistributionsMatrixNormal,fvar_double) {
  using stan::agrad::fvar;

  Matrix<fvar<double>,Dynamic,Dynamic> mu(3,5);
  mu.setZero();
  
  Matrix<fvar<double>,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0,
       11.0, 2.0, -5.0, 11.0, 0.0,
       -2.0, 11.0, 2.0, -2.0, -11.0;

  Matrix<fvar<double>,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
          -3.0,  4.0, 0.0,  0.0, 0.0,
           0.0,  0.0, 5.0,  1.0, 0.0,
           0.0,  0.0, 1.0, 10.0, 0.0,
           0.0,  0.0, 0.0,  0.0, 2.0;

  Matrix<fvar<double>,Dynamic,Dynamic> D(3,3);
  D << 1.0, 0.5, 0.1,
       0.5, 1.0, 0.2,
       0.1, 0.2, 1.0;
  
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 5; j++) {
      Sigma(i,j).d_ = 1.0;
      if (i < 3) {
        mu(i,j).d_ = 1.0;
        y(i,j).d_ = 1.0;
        if (j < 3)
          D(i,j).d_ = 1.0;
      }
    } 

  fvar<double> lp_ref = stan::prob::matrix_normal_prec_log(y,mu,D,Sigma);
  EXPECT_FLOAT_EQ(-2132.07482, lp_ref.val_);
  EXPECT_FLOAT_EQ(-2075.1274, lp_ref.d_);
}

TEST(ProbDistributionsMatrixNormal,fvar_var) {
  using stan::agrad::var;
  using stan::agrad::fvar;

  Matrix<fvar<var>,Dynamic,Dynamic> mu(3,5);
  mu.setZero();
  
  Matrix<fvar<var>,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0,
       11.0, 2.0, -5.0, 11.0, 0.0,
       -2.0, 11.0, 2.0, -2.0, -11.0;

  Matrix<fvar<var>,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
          -3.0,  4.0, 0.0,  0.0, 0.0,
           0.0,  0.0, 5.0,  1.0, 0.0,
           0.0,  0.0, 1.0, 10.0, 0.0,
           0.0,  0.0, 0.0,  0.0, 2.0;

  Matrix<fvar<var>,Dynamic,Dynamic> D(3,3);
  D << 1.0, 0.5, 0.1,
       0.5, 1.0, 0.2,
       0.1, 0.2, 1.0;
  
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 5; j++) {
      Sigma(i,j).d_ = 1.0;
      if (i < 3) {
        mu(i,j).d_ = 1.0;
        y(i,j).d_ = 1.0;
        if (j < 3)
          D(i,j).d_ = 1.0;
      }
    } 

  fvar<var> lp_ref = stan::prob::matrix_normal_prec_log(y,mu,D,Sigma);
  EXPECT_FLOAT_EQ(-2132.07482, lp_ref.val_.val());
  EXPECT_FLOAT_EQ(-2075.1274, lp_ref.d_.val());
}

TEST(ProbDistributionsMatrixNormal,fvar_fvar_double) {
  using stan::agrad::fvar;

  Matrix<fvar<fvar<double> >,Dynamic,Dynamic> mu(3,5);
  mu.setZero();
  
  Matrix<fvar<fvar<double> >,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0,
       11.0, 2.0, -5.0, 11.0, 0.0,
       -2.0, 11.0, 2.0, -2.0, -11.0;

  Matrix<fvar<fvar<double> >,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
          -3.0,  4.0, 0.0,  0.0, 0.0,
           0.0,  0.0, 5.0,  1.0, 0.0,
           0.0,  0.0, 1.0, 10.0, 0.0,
           0.0,  0.0, 0.0,  0.0, 2.0;

  Matrix<fvar<fvar<double> >,Dynamic,Dynamic> D(3,3);
  D << 1.0, 0.5, 0.1,
       0.5, 1.0, 0.2,
       0.1, 0.2, 1.0;
  
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 5; j++) {
      Sigma(i,j).d_.val_ = 1.0;
      if (i < 3) {
        mu(i,j).d_.val_ = 1.0;
        y(i,j).d_.val_ = 1.0;
        if (j < 3)
          D(i,j).d_.val_ = 1.0;
      }
    } 

  fvar<fvar<double> > lp_ref = stan::prob::matrix_normal_prec_log(y,mu,D,Sigma);
  EXPECT_FLOAT_EQ(-2132.07482, lp_ref.val_.val_);
  EXPECT_FLOAT_EQ(-2075.1274, lp_ref.d_.val_);
}

TEST(ProbDistributionsMatrixNormal,fvar_fvar_var) {
  using stan::agrad::var;
  using stan::agrad::fvar;

  Matrix<fvar<fvar<var> >,Dynamic,Dynamic> mu(3,5);
  mu.setZero();
  
  Matrix<fvar<fvar<var> >,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0,
       11.0, 2.0, -5.0, 11.0, 0.0,
       -2.0, 11.0, 2.0, -2.0, -11.0;

  Matrix<fvar<fvar<var> >,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
          -3.0,  4.0, 0.0,  0.0, 0.0,
           0.0,  0.0, 5.0,  1.0, 0.0,
           0.0,  0.0, 1.0, 10.0, 0.0,
           0.0,  0.0, 0.0,  0.0, 2.0;

  Matrix<fvar<fvar<var> >,Dynamic,Dynamic> D(3,3);
  D << 1.0, 0.5, 0.1,
       0.5, 1.0, 0.2,
       0.1, 0.2, 1.0;
  
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 5; j++) {
      Sigma(i,j).d_.val_ = 1.0;
      if (i < 3) {
        mu(i,j).d_.val_ = 1.0;
        y(i,j).d_.val_ = 1.0;
        if (j < 3)
          D(i,j).d_.val_ = 1.0;
      }
    } 

  fvar<fvar<var> > lp_ref = stan::prob::matrix_normal_prec_log(y,mu,D,Sigma);
  EXPECT_FLOAT_EQ(-2132.07482, lp_ref.val_.val_.val());
  EXPECT_FLOAT_EQ(-2075.1274, lp_ref.d_.val_.val());
}
