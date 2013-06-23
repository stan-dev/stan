#include <gtest/gtest.h>
#include <stan/prob/distributions/multivariate/continuous/dirichlet.hpp>
#include <boost/random/mersenne_twister.hpp>
#include<boost/math/distributions.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;

TEST(ProbDistributions,Dirichlet) {
  Matrix<double,Dynamic,1> theta(3,1);
  theta << 0.2, 0.3, 0.5;
  Matrix<double,Dynamic,1> alpha(3,1);
  alpha << 1.0, 1.0, 1.0;
  EXPECT_FLOAT_EQ(0.6931472, stan::prob::dirichlet_log(theta,alpha));
  
  Matrix<double,Dynamic,1> theta2(4,1);
  theta2 << 0.01, 0.01, 0.8, 0.18;
  Matrix<double,Dynamic,1> alpha2(4,1);
  alpha2 << 10.5, 11.5, 19.3, 5.1;
  EXPECT_FLOAT_EQ(-43.40045, stan::prob::dirichlet_log(theta2,alpha2));
}

TEST(ProbDistributions,DirichletPropto) {
  Matrix<double,Dynamic,1> theta(3,1);
  theta << 0.2, 0.3, 0.5;
  Matrix<double,Dynamic,1> alpha(3,1);
  alpha << 1.0, 1.0, 1.0;
  EXPECT_FLOAT_EQ(0.0, stan::prob::dirichlet_log<true>(theta,alpha));
  
  Matrix<double,Dynamic,1> theta2(4,1);
  theta2 << 0.01, 0.01, 0.8, 0.18;
  Matrix<double,Dynamic,1> alpha2(4,1);
  alpha2 << 10.5, 11.5, 19.3, 5.1;
  EXPECT_FLOAT_EQ(0.0, stan::prob::dirichlet_log<true>(theta2,alpha2));
}

TEST(ProbDistributionsDirichlet, random) {
  boost::random::mt19937 rng;
  Matrix<double,Dynamic,Dynamic> alpha(3,1);
  alpha << 2.0, 
    3.0,
    11.0;

  EXPECT_NO_THROW(stan::prob::dirichlet_rng(alpha,rng));
}

TEST(ProbDistributionsDirichlet, marginalOneChiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  Matrix<double,Dynamic,Dynamic> alpha(3,1);
  alpha << 2.0, 
    3.0,
    11.0;
  int N = 10000;
  int K = boost::math::round(2 * std::pow(N, 0.4));
  boost::math::beta_distribution<>dist (2.0,3.0 + 11.0);
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

  Eigen::VectorXd a(alpha.rows());

  while (count < N) {
    a = stan::prob::dirichlet_rng(alpha,rng);
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


TEST(ProbDistributionsDirichlet, marginalTwoChiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  Matrix<double,Dynamic,Dynamic> alpha(3,1);
  alpha << 2.0, 
    3.0,
    11.0;
  int N = 10000;
  int K = boost::math::round(2 * std::pow(N, 0.4));
  boost::math::beta_distribution<>dist (3.0,13.0);
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

  Eigen::VectorXd a(alpha.rows());

  while (count < N) {
    a = stan::prob::dirichlet_rng(alpha,rng);
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
