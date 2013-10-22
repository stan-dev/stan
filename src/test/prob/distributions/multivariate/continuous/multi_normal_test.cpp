#include <gtest/gtest.h>
#include "stan/prob/distributions/multivariate/continuous/multi_normal.hpp"
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <stan/agrad/agrad.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;

TEST(ProbDistributionsMultiNormal,MultiNormal) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  EXPECT_FLOAT_EQ(-11.73908, stan::prob::multi_normal_log(y,mu,Sigma));
  Matrix<double,Dynamic,Dynamic> L = Sigma.llt().matrixL();
  EXPECT_FLOAT_EQ(-11.73908, stan::prob::multi_normal_cholesky_log(y,mu,L));
}
TEST(ProbDistributionsMultiNormal,MultiNormalVar) {
  using stan::agrad::var;
  Matrix<var,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<var,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<var,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  EXPECT_FLOAT_EQ(-11.73908, stan::prob::multi_normal_log(y,mu,Sigma).val());
  Matrix<var,Dynamic,Dynamic> L = Sigma.llt().matrixL();
  EXPECT_FLOAT_EQ(-11.73908, stan::prob::multi_normal_cholesky_log(y,mu,L).val());
}
TEST(ProbDistributionsMultiNormal,MultiNormalGradientUnivariate) {
  using stan::agrad::var;
  using std::vector;
  using Eigen::VectorXd;
  using stan::prob::multi_normal_log;
  
  Matrix<var,Dynamic,1> y_var(1,1);
  y_var << 2.0;

  Matrix<var,Dynamic,1> mu_var(1,1);
  mu_var << 1.0;

  Matrix<var,Dynamic,Dynamic> Sigma_var(1,1);
  Sigma_var(0,0) = 9.0;

  std::vector<var> x;
  x.push_back(y_var(0));
  x.push_back(mu_var(0));
  x.push_back(Sigma_var(0,0));

  var lp = stan::prob::multi_normal_log(y_var,mu_var,Sigma_var);
  vector<double> grad;
  lp.grad(x,grad);

  // ===================================


  Matrix<double,Dynamic,1> y(1,1);
  y << 2.0;

  Matrix<double,Dynamic,1> mu(1,1);
  mu << 1.0;

  Matrix<double,Dynamic,Dynamic> Sigma(1,1);
  Sigma << 9.0;

  double epsilon = 1e-6;


  Matrix<double,Dynamic,1> y_m(1,1);
  Matrix<double,Dynamic,1> y_p(1,1);
  y_p[0] = y[0] + epsilon;
  y_m[0] = y[0] - epsilon;
  double grad_diff 
    =  (multi_normal_log(y_p,mu,Sigma) - multi_normal_log(y_m,mu,Sigma)) 
    / (2 * epsilon);
  EXPECT_FLOAT_EQ(grad_diff, grad[0]);

  Matrix<double,Dynamic,1> mu_m(1,1);
  Matrix<double,Dynamic,1> mu_p(1,1);
  mu_p[0] = mu[0] + epsilon;
  mu_m[0] = mu[0] - epsilon;
  grad_diff 
    =  (multi_normal_log(y,mu_p,Sigma) - multi_normal_log(y,mu_m,Sigma)) 
    / (2 * epsilon);
  EXPECT_FLOAT_EQ(grad_diff, grad[1]);

  Matrix<double,Dynamic,Dynamic> Sigma_m(1,1);
  Matrix<double,Dynamic,Dynamic> Sigma_p(1,1);
  Sigma_p(0) = Sigma(0) + epsilon;
  Sigma_m(0) = Sigma(0) - epsilon;
  grad_diff 
    =  (multi_normal_log(y,mu,Sigma_p) - multi_normal_log(y,mu,Sigma_m)) 
    / (2 * epsilon);
  EXPECT_FLOAT_EQ(grad_diff, grad[2]);
}

TEST(ProbDistributionsMultiNormal,MultiNormalGradient) {
  using stan::agrad::var;
  using std::vector;
  using Eigen::VectorXd;
  using stan::prob::multi_normal_log;

  Matrix<var,Dynamic,1> y_var(3,1);
  y_var << 2.0, -2.0, 11.0;

  Matrix<var,Dynamic,1> mu_var(3,1);
  mu_var << 1.0, -1.0, 3.0;

  Matrix<var,Dynamic,Dynamic> Sigma_var(3,3);
  Sigma_var(0,0) = 9.0;
  Sigma_var(1,1) = 4.0;
  Sigma_var(2,2) = 5.0;
  Sigma_var(0,1) = -3;
  Sigma_var(0,2) = 0;
  Sigma_var(1,2) = 0;

  Sigma_var(1,0) =  Sigma_var(0,1);
  Sigma_var(2,0) =  Sigma_var(0,2);
  Sigma_var(2,1) =  Sigma_var(1,2);

  std::vector<var> x;
  for (size_t i = 0; i < y_var.size(); ++i)
    x.push_back(y_var(i));
  for (size_t i = 0; i < mu_var.size(); ++i)
    x.push_back(mu_var(i));
  for (size_t i = 0; i < Sigma_var.rows(); ++i)
    for (size_t j = 0; j < Sigma_var.cols(); ++j)
      x.push_back(Sigma_var(i,j));

  var lp = stan::prob::multi_normal_log(y_var,mu_var,Sigma_var);
  vector<double> grad;
  lp.grad(x,grad);

  // --------------------------------------------

  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;

  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;

  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;

  double epsilon = 1e-6;

  int pos = 0;

  Matrix<double,Dynamic,1> y_m(3,1);
  Matrix<double,Dynamic,1> y_p(3,1);
  for (int i = 0; i < 3; ++i) {
    y_p[i] = y[i];
    y_m[i] = y[i];
  }
  for (int i = 0; i < 3; ++i) {
    y_p[i] = y[i] + epsilon;
    y_m[i] = y[i] - epsilon;
    double grad_diff 
      =  (multi_normal_log(y_p,mu,Sigma) - multi_normal_log(y_m,mu,Sigma)) 
         / (2 * epsilon);
    EXPECT_FLOAT_EQ(grad_diff, grad[pos++]);
    y_p[i] = y[i];
    y_m[i] = y[i];
  }

  Matrix<double,Dynamic,1> mu_m(3,1);
  Matrix<double,Dynamic,1> mu_p(3,1);
  for (int i = 0; i < 3; ++i) {
    mu_p[i] = mu[i];
    mu_m[i] = mu[i];
  }
  for (int i = 0; i < 3; ++i) {
    mu_p[i] = mu[i] + epsilon;
    mu_m[i] = mu[i] - epsilon;
    double grad_diff 
      =  (multi_normal_log(y,mu_p,Sigma) - multi_normal_log(y,mu_m,Sigma)) 
         / (2 * epsilon);
    EXPECT_FLOAT_EQ(grad_diff, grad[pos++]);
    mu_p[i] = mu[i];
    mu_m[i] = mu[i];
  }
  
  Matrix<double,Dynamic,Dynamic> Sigma_m(3,3);
  Matrix<double,Dynamic,Dynamic> Sigma_p(3,3);
  for (int i = 0; i < 9; ++i) {
    Sigma_p(i) = Sigma(i);
    Sigma_m(i) = Sigma(i);
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      Sigma_p(j,i) = Sigma_p(i,j) = Sigma(i,j) + epsilon;
      Sigma_m(j,i) = Sigma_m(i,j) = Sigma(i,j) - epsilon;
      double grad_diff 
        =  (multi_normal_log(y,mu,Sigma_p) - multi_normal_log(y,mu,Sigma_m)) 
        / (2 * epsilon);
      EXPECT_FLOAT_EQ(grad_diff, grad[pos++]);
      Sigma_p(j,i) = Sigma_p(i,j) = Sigma(i,j);
      Sigma_m(j,i) = Sigma_m(i,j) = Sigma(i,j);
    }
  }
      
}

TEST(ProbDistributionsMultiNormal,Sigma) {
  Matrix<double,Dynamic,1> y(2,1);
  y << 2.0, -2.0;
  Matrix<double,Dynamic,1> mu(2,1);
  mu << 1.0, -1.0;
  Matrix<double,Dynamic,Dynamic> Sigma(2,2);
  Sigma << 9.0, -3.0, -3.0, 4.0;
  EXPECT_NO_THROW (stan::prob::multi_normal_log(y, mu, Sigma));

  // non-symmetric
  Sigma(0, 1) = -2.5;
  EXPECT_THROW (stan::prob::multi_normal_log(y, mu, Sigma), std::domain_error);
}
TEST(ProbDistributionsMultiNormal,Mu) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  EXPECT_NO_THROW (stan::prob::multi_normal_log(y, mu, Sigma));

  mu(0) = std::numeric_limits<double>::infinity();
  EXPECT_THROW (stan::prob::multi_normal_log(y, mu, Sigma), std::domain_error);
  mu(0) = -std::numeric_limits<double>::infinity();
  EXPECT_THROW (stan::prob::multi_normal_log(y, mu, Sigma), std::domain_error);
  mu(0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW (stan::prob::multi_normal_log(y, mu, Sigma), std::domain_error);
}
TEST(ProbDistributionsMultiNormal,MultiNormalOneRow) {
  Matrix<double,Dynamic,Dynamic> y(1,3);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  EXPECT_FLOAT_EQ(-11.73908, stan::prob::multi_normal_log(y,mu,Sigma));
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
  EXPECT_FLOAT_EQ(-54.2152, stan::prob::multi_normal_log(y,mu,Sigma));
  Matrix<double,Dynamic,Dynamic> L = Sigma.llt().matrixL();
  EXPECT_FLOAT_EQ(-54.2152, stan::prob::multi_normal_cholesky_log(y,mu,L));
}
TEST(ProbDistributionsMultiNormal,SigmaMultiRow) {
  Matrix<double,Dynamic,Dynamic> y(1,2);
  y << 2.0, -2.0;
  Matrix<double,Dynamic,1> mu(2,1);
  mu << 1.0, -1.0;
  Matrix<double,Dynamic,Dynamic> Sigma(2,2);
  Sigma << 9.0, -3.0, -3.0, 4.0;
  EXPECT_NO_THROW (stan::prob::multi_normal_log(y, mu, Sigma));

  // non-symmetric
  Sigma(0, 1) = -2.5;
  EXPECT_THROW (stan::prob::multi_normal_log(y, mu, Sigma), std::domain_error);
  Matrix<double,Dynamic,Dynamic> z(2,1);
  
  // wrong dimensions
  z << 2.0, -2.0;
  EXPECT_THROW (stan::prob::multi_normal_log(z, mu, Sigma), std::domain_error);
}
TEST(ProbDistributionsMultiNormal,MuMultiRow) {
  Matrix<double,Dynamic,Dynamic> y(1,3);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  EXPECT_NO_THROW (stan::prob::multi_normal_log(y, mu, Sigma));

  mu(0) = std::numeric_limits<double>::infinity();
  EXPECT_THROW (stan::prob::multi_normal_log(y, mu, Sigma), std::domain_error);
  mu(0) = -std::numeric_limits<double>::infinity();
  EXPECT_THROW (stan::prob::multi_normal_log(y, mu, Sigma), std::domain_error);
  mu(0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW (stan::prob::multi_normal_log(y, mu, Sigma), std::domain_error);
}
TEST(ProbDistributionsMultiNormal,SizeMismatch) {
  Matrix<double,Dynamic,Dynamic> y(1,3);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(2,1);
  mu << 1.0, -1.0;
  Matrix<double,Dynamic,Dynamic> Sigma(2,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0;  
  EXPECT_THROW(stan::prob::multi_normal_log(y, mu, Sigma), std::domain_error);
}

TEST(ProbDistributionsMultiNormal, error_check) {
  boost::random::mt19937 rng;
  Matrix<double,Dynamic,Dynamic> mu(3,1);
  mu << 2.0, 
    -2.0,
    11.0;

  Matrix<double,Dynamic,Dynamic> sigma(3,3);
  sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 1.0,
    0.0, 1.0, 3.0;
  EXPECT_NO_THROW(stan::prob::multi_normal_rng(mu, sigma,rng));

  mu << stan::math::positive_infinity(), 
    -2.0,
    11.0;
  EXPECT_THROW(stan::prob::multi_normal_rng(mu,sigma,rng),std::domain_error);

  mu << 2.0, 
    -2.0,
    11.0;
  sigma << 9.0, -3.0, 0.0,
    3.0,  4.0, 0.0,
    -2.0, 1.0, 3.0;
  EXPECT_THROW(stan::prob::multi_normal_rng(mu,sigma,rng),std::domain_error);

}

TEST(ProbDistributionsMultiNormal, marginalOneChiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  Matrix<double,Dynamic,Dynamic> sigma(3,3);
  sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 1.0,
    0.0, 1.0, 3.0;
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
    a = stan::prob::multi_normal_rng(mu,sigma,rng);
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

TEST(ProbDistributionsMultiNormal, marginalTwoChiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  Matrix<double,Dynamic,Dynamic> sigma(3,3);
  sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 1.0,
    0.0, 1.0, 3.0;
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
    a = stan::prob::multi_normal_rng(mu,sigma,rng);
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

TEST(ProbDistributionsMultiNormal, marginalThreeChiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  Matrix<double,Dynamic,Dynamic> sigma(3,3);
  sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 1.0,
    0.0, 1.0, 16.0;
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
    a = stan::prob::multi_normal_rng(mu,sigma,rng);
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
