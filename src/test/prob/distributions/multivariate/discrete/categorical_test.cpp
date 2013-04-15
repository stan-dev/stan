#include <gtest/gtest.h>
#include <stan/prob/distributions/multivariate/discrete/categorical.hpp>
#include <boost/random/mersenne_twister.hpp>
#include<boost/math/distributions.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;

TEST(ProbDistributionsCategorical,Categorical) {
  Matrix<double,Dynamic,1> theta(3,1);
  theta << 0.3, 0.5, 0.2;
  EXPECT_FLOAT_EQ(-1.203973, stan::prob::categorical_log(1,theta));
  EXPECT_FLOAT_EQ(-0.6931472, stan::prob::categorical_log(2,theta));
}
TEST(ProbDistributionsCategorical,Propto) {
  Matrix<double,Dynamic,1> theta(3,1);
  theta << 0.3, 0.5, 0.2;
  EXPECT_FLOAT_EQ(0.0, stan::prob::categorical_log<true>(1,theta));
  EXPECT_FLOAT_EQ(0.0, stan::prob::categorical_log<true>(2,theta));
}

using stan::prob::categorical_log;

TEST(ProbDistributionsCategorical,DefaultPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();
  
  unsigned int n = 1;
  unsigned int N = 3;
  Matrix<double,Dynamic,1> theta(N,1);
  theta << 0.3, 0.5, 0.2;

  EXPECT_NO_THROW(categorical_log(N, theta));
  EXPECT_NO_THROW(categorical_log(n, theta));
  EXPECT_NO_THROW(categorical_log(2, theta));
  EXPECT_THROW(categorical_log(N+1, theta), std::domain_error);
  EXPECT_THROW(categorical_log(0, theta), std::domain_error);

  
  theta(0) = nan;
  EXPECT_THROW(categorical_log(n, theta), std::domain_error);
  theta(0) = inf;
  EXPECT_THROW(categorical_log(n, theta), std::domain_error);
  theta(0) = -inf;
  EXPECT_THROW(categorical_log(n, theta), std::domain_error);
  theta(0) = -1;
  theta(1) = 1;
  theta(2) = 0;
  EXPECT_THROW(categorical_log(n, theta), std::domain_error);
}

TEST(ProbDistributionCategorical, chiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  int N = 10000;
  Matrix<double,Dynamic,Dynamic> theta(3,1);
  theta << 0.15, 
    0.45,
    0.40;
  int K = theta.rows();
  boost::math::chi_squared mydist(K-1);

  Eigen::Matrix<double,Eigen::Dynamic,1> loc(theta.rows(),1);
  for(int i = 0; i < theta.rows(); i++)
    loc(i) = 0;

  for(int i = 0; i < theta.rows(); i++) {
    for(int j = i; j < theta.rows(); j++)
      loc(j) += theta(i);
  }

  int count = 0;
  int bin [K];
  double expect [K];
  for(int i = 0 ; i < K; i++) {
    bin[i] = 0;
    expect[i] = N * theta(i);
  }

  while (count < N) {
    int a = stan::prob::categorical_rng(theta,rng);
    bin[a - 1]++;
    count++;
   }

  double chi = 0;

  for(int j = 0; j < K; j++)
    chi += ((bin[j] - expect[j]) * (bin[j] - expect[j]) / expect[j]);

  EXPECT_TRUE(chi < quantile(complement(mydist, 1e-6)));
}
