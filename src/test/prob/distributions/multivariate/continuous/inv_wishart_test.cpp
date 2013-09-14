#include <gtest/gtest.h>
#include <stan/prob/distributions/multivariate/continuous/inv_wishart.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <stan/math/matrix/determinant.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;

using stan::prob::inv_wishart_log;

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
    -3.0,  4.0, 0.0,
    2.0, 1.0, 3.0;
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
