#include <gtest/gtest.h>
#include "stan/prob/distributions/multivariate/continuous/lkj_corr.hpp"
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>

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

TEST(ProbDistributionsLkjCorr,Sigma) {
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

TEST(ProbDistributionsLKJCorr, error_check) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::lkj_corr_cholesky_rng(5, 1.0,rng));
  EXPECT_NO_THROW(stan::prob::lkj_corr_rng(5, 1.0,rng));

  EXPECT_THROW(stan::prob::lkj_corr_cholesky_rng(5, -1.0,rng),std::domain_error);
  EXPECT_THROW(stan::prob::lkj_corr_rng(5, -1.0,rng),std::domain_error);
}

TEST(ProbDistributionsLKJCorr, chiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  int N = 10000;
  int K = boost::math::round(2 * std::pow(N, 0.4));
  boost::math::beta_distribution<>dist (2.5,2.5);
  boost::math::chi_squared mydist(K-1);

  double loc[K - 1];
  for(int i = 1; i < K; i++)
    loc[i - 1] = quantile(dist, i * std::pow(K, -1.0));

  int count = 0;
  int bin [K];
  double expect [K];
  for(int i = 0 ; i < K; i++)
  {
    bin[i] = 0;
    expect[i] = N / K;
  }

  while (count < N) {
    double a = 0.5 * (1.0 + stan::prob::lkj_corr_rng(5,1.0,rng)(3,4));
    int i = 0;
    while (i < K-1 && a > loc[i])
  ++i;
    ++bin[i];
    count++;
   }

  double chi = 0;

  for(int j = 0; j < K; j++)
    chi += ((bin[j] - expect[j]) * (bin[j] - expect[j]) / expect[j]);

  EXPECT_TRUE(chi < quantile(complement(mydist, 1e-6)));
}
