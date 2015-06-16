#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/prob/multi_normal_log.hpp>
#include <stan/math/prim/mat/prob/multi_normal_rng.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;
using std::vector;

TEST(ProbDistributionsMultiNormal,NotVectorized) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  EXPECT_FLOAT_EQ(-11.73908, stan::math::multi_normal_log(y,mu,Sigma));
}

TEST(ProbDistributionsMultiNormal,Vectorized) {
  vector< Matrix<double,Dynamic,1> > vec_y(2);
  vector< Matrix<double,1,Dynamic> > vec_y_t(2);
  Matrix<double,Dynamic,1> y(3);
  Matrix<double,1,Dynamic> y_t(3);
  y << 2.0, -2.0, 11.0;
  vec_y[0] = y;
  vec_y_t[0] = y;
  y << 4.0, -2.0, 1.0;
  vec_y[1] = y;
  vec_y_t[1] = y;
  y_t = y;
  
  vector< Matrix<double,Dynamic,1> > vec_mu(2);
  vector< Matrix<double,1,Dynamic> > vec_mu_t(2);
  Matrix<double,Dynamic,1> mu(3);
  Matrix<double,1,Dynamic> mu_t(3);
  mu << 1.0, -1.0, 3.0;
  vec_mu[0] = mu;
  vec_mu_t[0] = mu;
  mu << 2.0, -1.0, 4.0;
  vec_mu[1] = mu;
  vec_mu_t[1] = mu;
  mu_t = mu;
  
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 10.0, -3.0, 0.0,
    -3.0,  5.0, 0.0,
    0.0, 0.0, 5.0;
    
  //y and mu vectorized
  EXPECT_FLOAT_EQ(-11.928077-6.5378327, stan::math::multi_normal_log(vec_y,vec_mu,Sigma));
  EXPECT_FLOAT_EQ(-11.928077-6.5378327, stan::math::multi_normal_log(vec_y_t,vec_mu,Sigma));
  EXPECT_FLOAT_EQ(-11.928077-6.5378327, stan::math::multi_normal_log(vec_y,vec_mu_t,Sigma));
  EXPECT_FLOAT_EQ(-11.928077-6.5378327, stan::math::multi_normal_log(vec_y_t,vec_mu_t,Sigma));

  //y vectorized
  EXPECT_FLOAT_EQ(-10.44027-6.537833, stan::math::multi_normal_log(vec_y,mu,Sigma));
  EXPECT_FLOAT_EQ(-10.44027-6.537833, stan::math::multi_normal_log(vec_y_t,mu,Sigma));
  EXPECT_FLOAT_EQ(-10.44027-6.537833, stan::math::multi_normal_log(vec_y,mu_t,Sigma));
  EXPECT_FLOAT_EQ(-10.44027-6.537833, stan::math::multi_normal_log(vec_y_t,mu_t,Sigma));

  //mu vectorized
  EXPECT_FLOAT_EQ(-6.26954-6.537833, stan::math::multi_normal_log(y,vec_mu,Sigma));
  EXPECT_FLOAT_EQ(-6.26954-6.537833, stan::math::multi_normal_log(y_t,vec_mu,Sigma));
  EXPECT_FLOAT_EQ(-6.26954-6.537833, stan::math::multi_normal_log(y,vec_mu_t,Sigma));
  EXPECT_FLOAT_EQ(-6.26954-6.537833, stan::math::multi_normal_log(y_t,vec_mu_t,Sigma));
}
TEST(ProbDistributionsMultiNormal,Sigma) {
  Matrix<double,Dynamic,1> y(2,1);
  y << 2.0, -2.0;
  Matrix<double,Dynamic,1> mu(2,1);
  mu << 1.0, -1.0;
  Matrix<double,Dynamic,Dynamic> Sigma(2,2);
  Sigma << 9.0, -3.0, -3.0, 4.0;
  EXPECT_NO_THROW (stan::math::multi_normal_log(y, mu, Sigma));

  // non-symmetric
  Sigma(0, 1) = -2.5;
  EXPECT_THROW (stan::math::multi_normal_log(y, mu, Sigma), std::domain_error);
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
  EXPECT_NO_THROW (stan::math::multi_normal_log(y, mu, Sigma));

  mu(0) = std::numeric_limits<double>::infinity();
  EXPECT_THROW (stan::math::multi_normal_log(y, mu, Sigma), std::domain_error);
  mu(0) = -std::numeric_limits<double>::infinity();
  EXPECT_THROW (stan::math::multi_normal_log(y, mu, Sigma), std::domain_error);
  mu(0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW (stan::math::multi_normal_log(y, mu, Sigma), std::domain_error);
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
  EXPECT_FLOAT_EQ(-11.73908, stan::math::multi_normal_log(y,mu,Sigma));
}

TEST(ProbDistributionsMultiNormal,SigmaMultiRow) {
  Matrix<double,Dynamic,Dynamic> y(1,2);
  y << 2.0, -2.0;
  Matrix<double,Dynamic,1> mu(2,1);
  mu << 1.0, -1.0;
  Matrix<double,Dynamic,Dynamic> Sigma(2,2);
  Sigma << 9.0, -3.0, -3.0, 4.0;
  EXPECT_NO_THROW (stan::math::multi_normal_log(y, mu, Sigma));

  // non-symmetric
  Sigma(0, 1) = -2.5;
  EXPECT_THROW (stan::math::multi_normal_log(y, mu, Sigma), std::domain_error);
  Matrix<double,Dynamic,Dynamic> z(2,1);
  
  // wrong dimensions
  z << 2.0, -2.0;
  EXPECT_THROW (stan::math::multi_normal_log(z, mu, Sigma), std::domain_error);
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
  EXPECT_NO_THROW (stan::math::multi_normal_log(y, mu, Sigma));

  mu(0) = std::numeric_limits<double>::infinity();
  EXPECT_THROW (stan::math::multi_normal_log(y, mu, Sigma), std::domain_error);
  mu(0) = -std::numeric_limits<double>::infinity();
  EXPECT_THROW (stan::math::multi_normal_log(y, mu, Sigma), std::domain_error);
  mu(0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW (stan::math::multi_normal_log(y, mu, Sigma), std::domain_error);
}
TEST(ProbDistributionsMultiNormal,SizeMismatch) {
  Matrix<double,1,Dynamic> y(1,3);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(2,1);
  mu << 1.0, -1.0;
  Matrix<double,Dynamic,Dynamic> Sigma(2,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0;  
  EXPECT_THROW(stan::math::multi_normal_log(y, mu, Sigma), std::invalid_argument);
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
  EXPECT_NO_THROW(stan::math::multi_normal_rng(mu, sigma,rng));

  mu << stan::math::positive_infinity(), 
    -2.0,
    11.0;
  EXPECT_THROW(stan::math::multi_normal_rng(mu,sigma,rng),std::domain_error);

  mu << 2.0, 
    -2.0,
    11.0;
  sigma << 9.0, -3.0, 0.0,
    3.0,  4.0, 0.0,
    -2.0, 1.0, 3.0;
  EXPECT_THROW(stan::math::multi_normal_rng(mu,sigma,rng),std::domain_error);

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
    a = stan::math::multi_normal_rng(mu,sigma,rng);
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
    a = stan::math::multi_normal_rng(mu,sigma,rng);
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
    a = stan::math::multi_normal_rng(mu,sigma,rng);
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
