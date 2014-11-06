#include <gtest/gtest.h>
#include <stan/agrad/rev/matrix.hpp>
#include <stan/agrad/fwd/matrix.hpp>
#include <stan/prob/distributions/multivariate/continuous/multi_normal.hpp>
#include <stan/prob/distributions/multivariate/continuous/multi_gp.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;

using boost::math::policies::policy;
using boost::math::policies::evaluation_error;
using boost::math::policies::domain_error;
using boost::math::policies::overflow_error;
using boost::math::policies::domain_error;
using boost::math::policies::pole_error;
using boost::math::policies::errno_on_error;

typedef policy<
  domain_error<errno_on_error>, 
  pole_error<errno_on_error>,
  overflow_error<errno_on_error>,
  evaluation_error<errno_on_error> 
  > errno_policy;


TEST(ProbDistributionsMultiGP,MultiGP) {
  Matrix<double,Dynamic,1> mu(5,1);
  mu.setZero();
  
  Matrix<double,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0, 11.0, 2.0, -5.0, 11.0, 0.0, -2.0, 11.0, 2.0, -2.0, -11.0;

  Matrix<double,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
          -3.0,  4.0, 0.0,  0.0, 0.0,
           0.0,  0.0, 5.0,  1.0, 0.0,
           0.0,  0.0, 1.0, 10.0, 0.0,
           0.0,  0.0, 0.0,  0.0, 2.0;

  Matrix<double,Dynamic,1> w(3,1);
  w << 1.0, 0.5, 1.5;
  
  double lp_ref(0);
  for (size_t i = 0; i < 3; i++) {
    Matrix<double,Dynamic,1> cy(y.row(i).transpose());
    Matrix<double,Dynamic,Dynamic> cSigma((1.0/w[i])*Sigma);
    lp_ref += stan::prob::multi_normal_log(cy,mu,cSigma);
  }
  
  EXPECT_FLOAT_EQ(lp_ref, stan::prob::multi_gp_log(y,Sigma,w));
}

TEST(ProbDistributionsMultiGP,DefaultPolicySigma) {
  Matrix<double,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0, 11.0, 2.0, -5.0, 11.0, 0.0, -2.0, 11.0, 2.0, -2.0, -11.0;
  
  Matrix<double,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
  -3.0,  4.0, 0.0,  0.0, 0.0,
  0.0,  0.0, 5.0,  1.0, 0.0,
  0.0,  0.0, 1.0, 10.0, 0.0,
  0.0,  0.0, 0.0,  0.0, 2.0;
  
  Matrix<double,Dynamic,1> w(3,1);
  w << 1.0, 0.5, 1.5;
  
  // non-symmetric
  Sigma(0, 1) = -2.5;
  EXPECT_THROW (stan::prob::multi_gp_log(y, Sigma, w), std::domain_error);
  Sigma(0, 1) = Sigma(1, 0);

  // non-spd
  Sigma(0, 0) = -3.0;
  EXPECT_THROW (stan::prob::multi_gp_log(y, Sigma, w), std::domain_error);
  Sigma(0, 1) = 9.0;
}

TEST(ProbDistributionsMultiGP,DefaultPolicyW) {
  Matrix<double,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0, 11.0, 2.0, -5.0, 11.0, 0.0, -2.0, 11.0, 2.0, -2.0, -11.0;
  
  Matrix<double,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
  -3.0,  4.0, 0.0,  0.0, 0.0,
  0.0,  0.0, 5.0,  1.0, 0.0,
  0.0,  0.0, 1.0, 10.0, 0.0,
  0.0,  0.0, 0.0,  0.0, 2.0;
  
  Matrix<double,Dynamic,1> w(3,1);
  w << 1.0, 0.5, 1.5;
  
  // negative w
  w(0, 0) = -2.5;
  EXPECT_THROW (stan::prob::multi_gp_log(y, Sigma, w), std::domain_error);

  // non-finite values
  w(0, 0) = std::numeric_limits<double>::infinity();
  EXPECT_THROW (stan::prob::multi_gp_log(y, Sigma, w), std::domain_error);
  w(0, 0) = -std::numeric_limits<double>::infinity();
  EXPECT_THROW (stan::prob::multi_gp_log(y, Sigma, w), std::domain_error);
  w(0,0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW (stan::prob::multi_gp_log(y, Sigma, w), std::domain_error);
}

TEST(ProbDistributionsMultiGP,DefaultPolicyY) {
  Matrix<double,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0, 11.0, 2.0, -5.0, 11.0, 0.0, -2.0, 11.0, 2.0, -2.0, -11.0;
  
  Matrix<double,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
  -3.0,  4.0, 0.0,  0.0, 0.0,
  0.0,  0.0, 5.0,  1.0, 0.0,
  0.0,  0.0, 1.0, 10.0, 0.0,
  0.0,  0.0, 0.0,  0.0, 2.0;
  
  Matrix<double,Dynamic,1> w(3,1);
  w << 1.0, 0.5, 1.5;
  
  // non-finite values
  y(0, 0) = std::numeric_limits<double>::infinity();
  EXPECT_THROW (stan::prob::multi_gp_log(y, Sigma, w), std::domain_error);
  y(0, 0) = -std::numeric_limits<double>::infinity();
  EXPECT_THROW (stan::prob::multi_gp_log(y, Sigma, w), std::domain_error);
  y(0,0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW (stan::prob::multi_gp_log(y, Sigma, w), std::domain_error);
}

TEST(ProbDistributionsMultiGP,fvar_double) {
  using stan::agrad::fvar;
  Matrix<fvar<double>,Dynamic,1> mu(5,1);
  mu.setZero();
  
  Matrix<fvar<double>,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0, 11.0, 2.0, -5.0, 11.0, 0.0, -2.0, 11.0, 2.0, -2.0, -11.0;

  Matrix<fvar<double>,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
          -3.0,  4.0, 0.0,  0.0, 0.0,
           0.0,  0.0, 5.0,  1.0, 0.0,
           0.0,  0.0, 1.0, 10.0, 0.0,
           0.0,  0.0, 0.0,  0.0, 2.0;

  Matrix<fvar<double>,Dynamic,1> w(3,1);
  w << 1.0, 0.5, 1.5;
  
  for (int i = 0; i < 5; i++) {
    mu(i).d_ = 1.0;
    if (i < 3)
      w(i).d_ = 1.0;
    for (int j = 0; j < 5; j++) {
      Sigma(i,j).d_ = 1.0;
      if (i < 3)
        y(i,j).d_ = 1.0;
    }
  }

  fvar<double> lp_ref(0);
  for (size_t i = 0; i < 3; i++) {
    Matrix<fvar<double>,Dynamic,1> cy(y.row(i).transpose());
    Matrix<fvar<double>,Dynamic,Dynamic> cSigma((1.0/w[i])*Sigma);
    lp_ref += stan::prob::multi_normal_log(cy,mu,cSigma);
  }
  
  EXPECT_FLOAT_EQ(lp_ref.val_, stan::prob::multi_gp_log(y,Sigma,w).val_);
  EXPECT_FLOAT_EQ(-74.572952, stan::prob::multi_gp_log(y,Sigma,w).d_);
}

TEST(ProbDistributionsMultiGP,fvar_var) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  Matrix<fvar<var>,Dynamic,1> mu(5,1);
  mu.setZero();
  
  Matrix<fvar<var>,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0, 11.0, 2.0, -5.0, 11.0, 0.0, -2.0, 11.0, 2.0, -2.0, -11.0;

  Matrix<fvar<var>,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
          -3.0,  4.0, 0.0,  0.0, 0.0,
           0.0,  0.0, 5.0,  1.0, 0.0,
           0.0,  0.0, 1.0, 10.0, 0.0,
           0.0,  0.0, 0.0,  0.0, 2.0;

  Matrix<fvar<var>,Dynamic,1> w(3,1);
  w << 1.0, 0.5, 1.5;
  
  for (int i = 0; i < 5; i++) {
    mu(i).d_ = 1.0;
    if (i < 3)
      w(i).d_ = 1.0;
    for (int j = 0; j < 5; j++) {
      Sigma(i,j).d_ = 1.0;
      if (i < 3)
        y(i,j).d_ = 1.0;
    }
  }

  fvar<var> lp_ref(0);
  for (size_t i = 0; i < 3; i++) {
    Matrix<fvar<var>,Dynamic,1> cy(y.row(i).transpose());
    Matrix<fvar<var>,Dynamic,Dynamic> cSigma((1.0/w[i])*Sigma);
    lp_ref += stan::prob::multi_normal_log(cy,mu,cSigma);
  }
  
  EXPECT_FLOAT_EQ(lp_ref.val_.val(), stan::prob::multi_gp_log(y,Sigma,w).val_.val());
  EXPECT_FLOAT_EQ(-74.572952, stan::prob::multi_gp_log(y,Sigma,w).d_.val());
}

TEST(ProbDistributionsMultiGP,fvar_fvar_double) {
  using stan::agrad::fvar;
  Matrix<fvar<fvar<double> >,Dynamic,1> mu(5,1);
  mu.setZero();
  
  Matrix<fvar<fvar<double> >,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0, 11.0, 2.0, -5.0, 11.0, 0.0, -2.0, 11.0, 2.0, -2.0, -11.0;

  Matrix<fvar<fvar<double> >,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
          -3.0,  4.0, 0.0,  0.0, 0.0,
           0.0,  0.0, 5.0,  1.0, 0.0,
           0.0,  0.0, 1.0, 10.0, 0.0,
           0.0,  0.0, 0.0,  0.0, 2.0;

  Matrix<fvar<fvar<double> >,Dynamic,1> w(3,1);
  w << 1.0, 0.5, 1.5;
  
  for (int i = 0; i < 5; i++) {
    mu(i).d_.val_ = 1.0;
    if (i < 3)
      w(i).d_.val_ = 1.0;
    for (int j = 0; j < 5; j++) {
      Sigma(i,j).d_.val_ = 1.0;
      if (i < 3)
        y(i,j).d_.val_ = 1.0;
    }
  }

  fvar<fvar<double> > lp_ref(0);
  for (size_t i = 0; i < 3; i++) {
    Matrix<fvar<fvar<double> >,Dynamic,1> cy(y.row(i).transpose());
    Matrix<fvar<fvar<double> >,Dynamic,Dynamic> cSigma((1.0/w[i])*Sigma);
    lp_ref += stan::prob::multi_normal_log(cy,mu,cSigma);
  }
  
  EXPECT_FLOAT_EQ(lp_ref.val_.val_, stan::prob::multi_gp_log(y,Sigma,w).val_.val_);
  EXPECT_FLOAT_EQ(-74.572952, stan::prob::multi_gp_log(y,Sigma,w).d_.val_);
}

TEST(ProbDistributionsMultiGP,fvar_fvar_var) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  Matrix<fvar<fvar<var> >,Dynamic,1> mu(5,1);
  mu.setZero();
  
  Matrix<fvar<fvar<var> >,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0, 11.0, 2.0, -5.0, 11.0, 0.0, -2.0, 11.0, 2.0, -2.0, -11.0;

  Matrix<fvar<fvar<var> >,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
          -3.0,  4.0, 0.0,  0.0, 0.0,
           0.0,  0.0, 5.0,  1.0, 0.0,
           0.0,  0.0, 1.0, 10.0, 0.0,
           0.0,  0.0, 0.0,  0.0, 2.0;

  Matrix<fvar<fvar<var> >,Dynamic,1> w(3,1);
  w << 1.0, 0.5, 1.5;
  
  for (int i = 0; i < 5; i++) {
    mu(i).d_.val_ = 1.0;
    if (i < 3)
      w(i).d_.val_ = 1.0;
    for (int j = 0; j < 5; j++) {
      Sigma(i,j).d_.val_ = 1.0;
      if (i < 3)
        y(i,j).d_.val_ = 1.0;
    }
  }

  fvar<fvar<var> > lp_ref(0);
  for (size_t i = 0; i < 3; i++) {
    Matrix<fvar<fvar<var> >,Dynamic,1> cy(y.row(i).transpose());
    Matrix<fvar<fvar<var> >,Dynamic,Dynamic> cSigma((1.0/w[i])*Sigma);
    lp_ref += stan::prob::multi_normal_log(cy,mu,cSigma);
  }
  
  EXPECT_FLOAT_EQ(lp_ref.val_.val_.val(), stan::prob::multi_gp_log(y,Sigma,w).val_.val_.val());
  EXPECT_FLOAT_EQ(-74.572952, stan::prob::multi_gp_log(y,Sigma,w).d_.val_.val());
}
