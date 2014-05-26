#include <gtest/gtest.h>
#include "stan/prob/distributions/multivariate/continuous/multi_normal_cholesky.hpp"
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <stan/agrad/rev.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;

TEST(ProbDistributionsMultiNormalCholesky,MultiNormal) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  Matrix<double,Dynamic,Dynamic> L = Sigma.llt().matrixL();
  EXPECT_FLOAT_EQ(-11.73908, stan::prob::multi_normal_cholesky_log(y,mu,L));
}

TEST(ProbDistributionsMultiNormalCholesky,MultiNormalVar) {
  using stan::agrad::var;
  Matrix<var,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<var,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<var,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  Matrix<var,Dynamic,Dynamic> L = Sigma.llt().matrixL();
  EXPECT_FLOAT_EQ(-11.73908, stan::prob::multi_normal_cholesky_log(y,mu,L).val());
}

TEST(ProbDistributionsMultiNormalCholesky,MultiNormalOneRow) {
  Matrix<double,Dynamic,Dynamic> y(1,3);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  Matrix<double,Dynamic,Dynamic> L = Sigma.llt().matrixL();
  EXPECT_FLOAT_EQ(-11.73908, stan::prob::multi_normal_cholesky_log(y,mu,L));
}

TEST(ProbDistributionsMultiNormal,MultiNormalMultiRow) {
  Matrix<double,Dynamic,Dynamic> y(2,3);
  y << 2.0, -2.0, 11.0,
       4.0, -4.0, 22.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  Matrix<double,Dynamic,Dynamic> L = Sigma.llt().matrixL();
  EXPECT_FLOAT_EQ(-54.2152, stan::prob::multi_normal_cholesky_log(y,mu,L));
}


TEST(ProbDistributionsMultiNormalCholesky, error_check) {
  boost::random::mt19937 rng;
  Matrix<double,Dynamic,Dynamic> mu(3,1);
  mu << 2.0, 
    -2.0,
    11.0;

  Matrix<double,Dynamic,Dynamic> sigma(3,3);
  sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 1.0,
    0.0, 1.0, 3.0;
  sigma = sigma.llt().matrixL();
  EXPECT_NO_THROW(stan::prob::multi_normal_cholesky_rng(mu, sigma,rng));

  mu << stan::math::positive_infinity(), 
    -2.0,
    11.0;
  EXPECT_THROW(stan::prob::multi_normal_cholesky_rng(mu,sigma,rng),std::domain_error);
}

TEST(ProbDistributionsMultiNormalCholesky, marginalOneChiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  Matrix<double,Dynamic,Dynamic> sigma(3,3);
  sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 1.0,
    0.0, 1.0, 3.0;
  sigma = sigma.llt().matrixL();
  Matrix<double,Dynamic,Dynamic> mu(3,1);
  mu << 2.0, 
    -2.0,
    11.0;
  int N = 10000;
  int K = boost::math::round(2 * std::pow(N, 0.4));
  boost::math::normal_distribution<>dist (2.0,3.0);
  boost::math::chi_squared mydist(K-1);

  double loc[K - 1];
  for(int i = 1; i < K; i++)
    loc[i - 1] = quantile(dist, i * std::pow(K, -1.0));

  int count = 0;
  int bin [K];
  double expect [K];
  for(int i = 0 ; i < K; i++) {
    bin[i] = 0;
    expect[i] = N / K;
  }
  Eigen::VectorXd a(mu.rows());
  while (count < N) {
    a = stan::prob::multi_normal_cholesky_rng(mu,sigma,rng);
    int i = 0;
    while (i < K-1 && a(0) > loc[i]) 
      ++i;
    ++bin[i];
    count++;
   }

  double chi = 0;
  for(int j = 0; j < K; j++)
    chi += ((bin[j] - expect[j]) * (bin[j] - expect[j]) / expect[j]);

  EXPECT_TRUE(chi < quantile(complement(mydist, 1e-6)));
}

TEST(ProbDistributionsMultiNormalCholesky, marginalTwoChiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  Matrix<double,Dynamic,Dynamic> sigma(3,3);
  sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 1.0,
    0.0, 1.0, 3.0;
  sigma = sigma.llt().matrixL();
  Matrix<double,Dynamic,Dynamic> mu(3,1);
  mu << 2.0, 
    -2.0,
    11.0;
  int N = 10000;
  int K = boost::math::round(2 * std::pow(N, 0.4));
  boost::math::normal_distribution<>dist (-2.0,2.0);
  boost::math::chi_squared mydist(K-1);

  double loc[K - 1];
  for(int i = 1; i < K; i++)
    loc[i - 1] = quantile(dist, i * std::pow(K, -1.0));

  int count = 0;
  int bin [K];
  double expect [K];
  for(int i = 0 ; i < K; i++) {
    bin[i] = 0;
    expect[i] = N / K;
  }
  Eigen::VectorXd a(mu.rows());
  while (count < N) {
    a = stan::prob::multi_normal_cholesky_rng(mu,sigma,rng);
    int i = 0;
    while (i < K-1 && a(1) > loc[i]) 
      ++i;
    ++bin[i];
    count++;
   }

  double chi = 0;
  for(int j = 0; j < K; j++)
    chi += ((bin[j] - expect[j]) * (bin[j] - expect[j]) / expect[j]);

  EXPECT_TRUE(chi < quantile(complement(mydist, 1e-6)));
}

TEST(ProbDistributionsMultiNormalCholesky, marginalThreeChiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  Matrix<double,Dynamic,Dynamic> sigma(3,3);
  sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 1.0,
    0.0, 1.0, 16.0;
  sigma = sigma.llt().matrixL();
  Matrix<double,Dynamic,Dynamic> mu(3,1);
  mu << 2.0, 
    -2.0,
    11.0;
  int N = 10000;
  int K = boost::math::round(2 * std::pow(N, 0.4));
  boost::math::normal_distribution<>dist (11.0,4.0);
  boost::math::chi_squared mydist(K-1);

  double loc[K - 1];
  for(int i = 1; i < K; i++)
    loc[i - 1] = quantile(dist, i * std::pow(K, -1.0));

  int count = 0;
  int bin [K];
  double expect [K];
  for(int i = 0 ; i < K; i++) {
    bin[i] = 0;
    expect[i] = N / K;
  }
  Eigen::VectorXd a(mu.rows());
  while (count < N) {
    a = stan::prob::multi_normal_cholesky_rng(mu,sigma,rng);
    int i = 0;
    while (i < K-1 && a(2) > loc[i]) 
      ++i;
    ++bin[i];
    count++;
   }

  double chi = 0;
  for(int j = 0; j < K; j++)
    chi += ((bin[j] - expect[j]) * (bin[j] - expect[j]) / expect[j]);

  EXPECT_TRUE(chi < quantile(complement(mydist, 1e-6)));
}
