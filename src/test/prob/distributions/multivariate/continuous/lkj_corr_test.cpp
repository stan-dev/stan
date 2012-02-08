#include <gtest/gtest.h>
#include "stan/prob/distributions/multivariate/continuous/lkj_corr.hpp"

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

TEST(ProbDistributionsLkjCorr,testIdentity) {
  unsigned int K = 4;
  Eigen::MatrixXd Sigma(K,K);
  Sigma.setZero();
  Sigma.diagonal().setOnes();
  srand(time(0));
  double eta = rand() / double(RAND_MAX) + 0.5;
  double f = stan::prob::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f, stan::prob::lkj_corr_log(Sigma, eta));
  eta = 1.0;
  f = stan::prob::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f, stan::prob::lkj_corr_log(Sigma, eta));
}


TEST(ProbDistributionsLkjCorr,testHalf) {
  unsigned int K = 4;
  Eigen::MatrixXd Sigma(K,K);
  Sigma.setConstant(0.5);
  Sigma.diagonal().setOnes();
  double eta = rand() / double(RAND_MAX) + 0.5;
  double f = stan::prob::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f + (eta - 1.0) * log(0.3125), stan::prob::lkj_corr_log(Sigma, eta));
  eta = 1.0;
  f = stan::prob::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f, stan::prob::lkj_corr_log(Sigma, eta));
}

TEST(ProbDistributionsLkjCorr,DefaultPolicySigma) {
  unsigned int K = 4;
  Eigen::MatrixXd Sigma(K,K);
  Sigma.setZero();
  Sigma.diagonal().setOnes();
  double eta = rand() / double(RAND_MAX) + 0.5;
  EXPECT_NO_THROW (stan::prob::lkj_corr_log(Sigma, eta));
  
  EXPECT_THROW (stan::prob::lkj_corr_log(Sigma, -eta), std::domain_error);
  
  Sigma = Sigma * -1.0;
  EXPECT_THROW (stan::prob::lkj_corr_log(Sigma, eta), std::domain_error);
  Sigma = Sigma * (0.0 / 0.0);
  EXPECT_THROW (stan::prob::lkj_corr_log(Sigma, eta), std::domain_error);
}

TEST(ProbDistributionsLkjCorr,ErrnoPolicySigma) {
  unsigned int K = 4;
  Eigen::MatrixXd Sigma(K,K);
  Sigma.setZero();
  Sigma.diagonal().setOnes();
  double eta = rand() / double(RAND_MAX) + 0.5;
    
  double result;
  EXPECT_NO_THROW (result=stan::prob::lkj_corr_log(Sigma, eta, errno_policy()));

  // non-symmetric
  Sigma(0, 1) = .5;
  EXPECT_NO_THROW (result=stan::prob::lkj_corr_log(Sigma, eta, errno_policy()));
  EXPECT_TRUE(std::isnan(result)) << "non-symmetric Sigma should return nan.";
}

