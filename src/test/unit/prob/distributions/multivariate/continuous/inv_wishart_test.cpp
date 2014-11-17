#include <stan/agrad/rev/matrix.hpp>
#include <stan/agrad/fwd/matrix.hpp>
#include <stan/prob/distributions/multivariate/continuous/inv_wishart.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <stan/math/matrix/determinant.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;

using stan::prob::inv_wishart_log;

TEST(ProbDistributionsInvWishart,LowerTriangular) {
  //Tests if only of the lower triangular portion of
  //outcome and scale matrices are taken
  using Eigen::MatrixXd;
  using stan::prob::inv_wishart_log;
  
  MatrixXd Sigma(4,4);
  MatrixXd Sigma_sym(4,4);
  MatrixXd Y(4,4);
  MatrixXd Y_sym(4,4);

  Y << 7.988168,  -10.955605, -14.47483,   4.395895,
    -9.555605,  44.750570,  49.21577, -15.454186,
    -14.474830,  49.215769,  60.08987, -20.481079,
    4.395895, -18.454186, -21.48108, 7.885833;

  Y_sym << 7.988168,  -9.555605, -14.474830,   4.395895,
    -9.555605,  44.750570,  49.215769, -18.454186,
    -14.474830,  49.215769,  60.08987, -21.48108,
    4.395895, -18.454186, -21.48108, 7.885833;
  
  Sigma << 2.9983662,  0.2898776, -2.650523,  0.1055911,
    0.2898776, 11.4803610,  7.157993, -3.1129955,
    -2.6505229,  7.1579931, 11.676181, -3.5866852,
    0.1055911, -3.1129955, -3.586685,  1.4482736;
  
  Sigma_sym << 2.9983662,  0.2898776, -2.6505229,  0.1055911,
    0.2898776, 11.4803610,  7.1579931, -3.1129955,
    -2.6505229,  7.1579931, 11.676181, -3.586685,
    0.1055911, -3.1129955, -3.586685,  1.4482736;

  unsigned int dof = 5;
   
  EXPECT_EQ(inv_wishart_log(Y,dof,Sigma), inv_wishart_log(Y_sym,dof,Sigma));
  EXPECT_EQ(inv_wishart_log(Y,dof,Sigma), inv_wishart_log(Y,dof,Sigma_sym));
  EXPECT_EQ(inv_wishart_log(Y,dof,Sigma), inv_wishart_log(Y_sym,dof,Sigma_sym));
}
TEST(ProbDistributionsInvWishart,InvWishart) {
  Matrix<double,Dynamic,Dynamic> Y(3,3);
  Y <<  12.147233, -11.9036079, 1.0910458,
    -11.903608,  16.7585782, 0.8530256,
    1.091046,   0.8530256, 2.5786609;

  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 7.785215,  3.0597878,  1.1071663,
    3.059788, 10.3515035, -0.1232598,
    1.107166, -0.1232598,  7.7623386;
  
  double dof = 4.0;
  double log_p = log(2.008407e-08);

  EXPECT_NEAR(log_p, stan::prob::inv_wishart_log(Y,dof,Sigma), 0.01);
}
TEST(ProbDistributionsInvWishart,Propto) {
  Matrix<double,Dynamic,Dynamic> Y(3,3);
  Y <<  12.147233, -11.9036079, 1.0910458,
    -11.903608,  16.7585782, 0.8530256,
    1.091046,   0.8530256, 2.5786609;

  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 7.785215,  3.0597878,  1.1071663,
    3.059788, 10.3515035, -0.1232598,
    1.107166, -0.1232598,  7.7623386;
  
  double dof = 4.0;
  
  EXPECT_FLOAT_EQ(0.0, stan::prob::inv_wishart_log<true>(Y,dof,Sigma));
}
TEST(ProbDistributionsInvWishart,DefaultPolicy) {
  Matrix<double,Dynamic,Dynamic> Sigma;
  Matrix<double,Dynamic,Dynamic> Y;
  double nu;
  
  Sigma.resize(1,1);
  Y.resize(1,1);
  Sigma.setIdentity();
  Y.setIdentity();
  nu = 1;
  EXPECT_NO_THROW(inv_wishart_log(Y, nu, Sigma));
  
  nu = 5;
  Sigma.resize(2,1);
  EXPECT_THROW(inv_wishart_log(Y, nu, Sigma), std::domain_error);

  nu = 5;
  Sigma.resize(2,2);
  Y.resize(2,1);
  EXPECT_THROW(inv_wishart_log(Y, nu, Sigma), std::domain_error);
  
  nu = 5;
  Sigma.resize(2,2);
  Y.resize(3,3);
  EXPECT_THROW(inv_wishart_log(Y, nu, Sigma), std::domain_error);

  Sigma.resize(3,3);
  Sigma.setIdentity();
  Y.resize(3,3);
  Y.setIdentity();
  nu = 3;
  EXPECT_NO_THROW(inv_wishart_log(Y, nu, Sigma));
  nu = 2;
  EXPECT_THROW(inv_wishart_log(Y, nu, Sigma), std::domain_error);
}

TEST(ProbDistributionsInvWishart, error_checks) {
  boost::random::mt19937 rng;

  Matrix<double,Dynamic,Dynamic> sigma(3,3);
  sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    2.0, 1.0, 3.0;
  EXPECT_NO_THROW(stan::prob::inv_wishart_rng(3.0, sigma,rng));
  EXPECT_THROW(stan::prob::inv_wishart_rng(2.0, sigma,rng),std::domain_error);

  Matrix<double,Dynamic,Dynamic> sigma2(4,3);
  EXPECT_THROW(stan::prob::inv_wishart_rng(2.0, sigma2,rng),std::domain_error);
}

TEST(ProbDistributionsInvWishart, chiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  Matrix<double,Dynamic,Dynamic> sigma(3,3);
  sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 1.0,
    0.0, 1.0, 3.0;
  int N = 10000;
  boost::math::chi_squared mydist(1);

  Matrix<double,Dynamic,Dynamic> siginv(3,3);
  siginv = sigma.inverse();

  int count = 0;
  double avg = 0;
  double expect = sigma.rows() * std::log(2.0) + std::log(stan::math::determinant(siginv)) + boost::math::digamma(5.0 / 2.0) + boost::math::digamma(4.0 / 2.0) + boost::math::digamma(3.0 / 2.0);

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> a(sigma.rows(),sigma.rows());
  while (count < N) {
    a = stan::prob::inv_wishart_rng(5.0, sigma, rng);
    avg += std::log(stan::math::determinant(a)) / N;
    count++;
   }
 
  double chi = (expect - avg) * (expect - avg) / expect;

  EXPECT_TRUE(chi < quantile(complement(mydist, 1e-6)));
}


TEST(ProbDistributionsInvWishart, SpecialRNGTest) {
  //When the scale matrix is an identity matrix and df = k + 2
  //The avg of the samples should also be an identity matrix
  
  boost::random::mt19937 rng;
  using Eigen::MatrixXd;
  
  MatrixXd sigma;
  MatrixXd Z;
  int N = 1e5;
  double tol = .1;
  for (int k = 1; k < 5; k++) {
    sigma = MatrixXd::Identity(k, k);
    Z = MatrixXd::Zero(k, k);
    for (int i = 0; i < N; i++)
      Z += stan::prob::inv_wishart_rng(k + 2, sigma, rng);
    Z /= N;
    for (int j = 0; j < k; j++) {
      for (int i = 0; i < k; i++) {
        if (j == i)
          EXPECT_NEAR(Z(i, j), 1.0, tol);
        else
          EXPECT_NEAR(Z(i, j), 0.0, tol);
      }
    }
  }
}

TEST(ProbDistributionsInvWishart,fvar_double) {
  using stan::agrad::fvar;

  Matrix<fvar<double>,Dynamic,Dynamic> Y(3,3);
  Y <<  12.147233, -11.9036079, 1.0910458,
    -11.903608,  16.7585782, 0.8530256,
    1.091046,   0.8530256, 2.5786609;

  Matrix<fvar<double>,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 7.785215,  3.0597878,  1.1071663,
    3.059788, 10.3515035, -0.1232598,
    1.107166, -0.1232598,  7.7623386;
  
  double dof = 4.0;
  double log_p = log(2.008407e-08);

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      Y(i,j).d_ = 1.0;
      Sigma(i,j).d_ = 1.0;
    }

  EXPECT_NEAR(log_p, stan::prob::inv_wishart_log(Y,dof,Sigma).val_, 0.01);
  EXPECT_NEAR(-1.4893348387330674, stan::prob::inv_wishart_log(Y,dof,Sigma).d_, 0.01);
}

TEST(ProbDistributionsInvWishart,fvar_fvar_double) {
  using stan::agrad::fvar;

  Matrix<fvar<fvar<double> >,Dynamic,Dynamic> Y(3,3);
  Y <<  12.147233, -11.9036079, 1.0910458,
    -11.903608,  16.7585782, 0.8530256,
    1.091046,   0.8530256, 2.5786609;

  Matrix<fvar<fvar<double> >,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 7.785215,  3.0597878,  1.1071663,
    3.059788, 10.3515035, -0.1232598,
    1.107166, -0.1232598,  7.7623386;
  
  double dof = 4.0;
  double log_p = log(2.008407e-08);

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      Y(i,j).d_ = 1.0;
      Sigma(i,j).d_ = 1.0;
    }

  EXPECT_NEAR(log_p, stan::prob::inv_wishart_log(Y,dof,Sigma).val_.val_, 0.01);
  EXPECT_NEAR(-1.4893348387330674, stan::prob::inv_wishart_log(Y,dof,Sigma).d_.val_, 0.01);
}

TEST(ProbDistributionsInvWishart,fvar_var) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  Matrix<fvar<var>,Dynamic,Dynamic> Y(3,3);
  Y <<  12.147233, -11.9036079, 1.0910458,
    -11.903608,  16.7585782, 0.8530256,
    1.091046,   0.8530256, 2.5786609;

  Matrix<fvar<var>,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 7.785215,  3.0597878,  1.1071663,
    3.059788, 10.3515035, -0.1232598,
    1.107166, -0.1232598,  7.7623386;
  
  double dof = 4.0;
  double log_p = log(2.008407e-08);

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      Y(i,j).d_ = 1.0;
      Sigma(i,j).d_ = 1.0;
    }

  EXPECT_NEAR(log_p, stan::prob::inv_wishart_log(Y,dof,Sigma).val_.val(), 0.01);
  EXPECT_NEAR(-1.4893348387330674, stan::prob::inv_wishart_log(Y,dof,Sigma).d_.val(), 0.01);
}

TEST(ProbDistributionsInvWishart,fvar_fvar_var) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  Matrix<fvar<fvar<var> >,Dynamic,Dynamic> Y(3,3);
  Y <<  12.147233, -11.9036079, 1.0910458,
    -11.903608,  16.7585782, 0.8530256,
    1.091046,   0.8530256, 2.5786609;

  Matrix<fvar<fvar<var> >,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 7.785215,  3.0597878,  1.1071663,
    3.059788, 10.3515035, -0.1232598,
    1.107166, -0.1232598,  7.7623386;
  
  double dof = 4.0;
  double log_p = log(2.008407e-08);

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      Y(i,j).d_ = 1.0;
      Sigma(i,j).d_ = 1.0;
    }

  EXPECT_NEAR(log_p, stan::prob::inv_wishart_log(Y,dof,Sigma).val_.val_.val(), 0.01);
  EXPECT_NEAR(-1.4893348387330674, stan::prob::inv_wishart_log(Y,dof,Sigma).d_.val_.val(), 0.01);
}
