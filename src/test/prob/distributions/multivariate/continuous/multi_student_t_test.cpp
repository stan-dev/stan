#include <gtest/gtest.h>
#include "stan/prob/distributions/multivariate/continuous/multi_student_t.hpp"
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;

using stan::prob::multi_student_t_log;

TEST(ProbDistributionsMultiStudentT,MultiT) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  double nu = 4.0;
  double lp = multi_student_t_log(y,nu,mu,Sigma);
  // calc using R's mnormt package's dmt function
  EXPECT_NEAR(-10.1246,lp,0.0001);
}

TEST(ProbDistributionsMultiStudentT,Sigma) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  double nu = 4.0;
  EXPECT_NO_THROW(multi_student_t_log(y,nu,mu,Sigma));
  
  Sigma(0,1) = 10; // non-symmetric
  EXPECT_THROW(multi_student_t_log(y,nu,mu,Sigma), std::domain_error);
}

TEST(ProbDistributionsMultiStudentT,Mu) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  double nu = 4.0;
  EXPECT_NO_THROW(multi_student_t_log(y,nu,mu,Sigma));
  
  mu(0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(multi_student_t_log(y,nu,mu,Sigma), std::domain_error);

  mu(0) = std::numeric_limits<double>::infinity();
  EXPECT_THROW(multi_student_t_log(y,nu,mu,Sigma), std::domain_error);

  mu(0) = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(multi_student_t_log(y,nu,mu,Sigma), std::domain_error);
}

TEST(ProbDistributionsMultiStudentT,Y) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  double nu = 4.0;
  EXPECT_NO_THROW(multi_student_t_log(y,nu,mu,Sigma));
  
  y(0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(multi_student_t_log(y,nu,mu,Sigma), std::domain_error);

  y(0) = std::numeric_limits<double>::infinity();
  EXPECT_NO_THROW(multi_student_t_log(y,nu,mu,Sigma));

  y(0) = -std::numeric_limits<double>::infinity();
  EXPECT_NO_THROW(multi_student_t_log(y,nu,mu,Sigma));
}

TEST(ProbDistributionsMultiStudentT,Nu) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  double nu = 4.0;
  EXPECT_NO_THROW(multi_student_t_log(y,nu,mu,Sigma));
  
  nu = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(multi_student_t_log(y,nu,mu,Sigma), std::domain_error);

  nu = 0.0;
  EXPECT_THROW(multi_student_t_log(y,nu,mu,Sigma), std::domain_error);

  nu = -1.0;
  EXPECT_THROW(multi_student_t_log(y,nu,mu,Sigma), std::domain_error);

  nu = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(multi_student_t_log(y,nu,mu,Sigma), std::domain_error);

  // nu = infinity OK
  nu = std::numeric_limits<double>::infinity();
  EXPECT_NO_THROW(multi_student_t_log(y,nu,mu,Sigma));
}


TEST(ProbDistributionsMultiStudentT,PolicySize1) {
  Matrix<double,Dynamic,1> y(2,1);
  y << 2.0, -2.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  double nu = 4.0;

  EXPECT_THROW(multi_student_t_log(y,nu,mu,Sigma),std::domain_error);
}

TEST(ProbDistributionsMultiStudentT,PolicySize2) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,2);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0;
  double nu = 4.0;

  EXPECT_THROW(multi_student_t_log(y,nu,mu,Sigma),std::domain_error);
}

TEST(ProbDistributionsMultiStudentT,PolicySize3) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(2,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0;
  double nu = 4.0;

  EXPECT_THROW(multi_student_t_log(y,nu,mu,Sigma),std::domain_error);
}

TEST(ProbDistributionsMultiStudentT,ProptoAllDoublesZero) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  double nu = 4.0;

  EXPECT_FLOAT_EQ(0.0,multi_student_t_log<true>(y,nu,mu,Sigma));
}


TEST(ProbDistributionsMultiStudentT, error_check) {
  boost::random::mt19937 rng;
  Matrix<double,Dynamic,Dynamic> mu(3,1);
  mu << 2.0, 
    3.0,
    11.0;

Matrix<double,Dynamic,Dynamic> s(3,3);
 s << 2.0, 3.0, 11.0,
   3.0, 9.0, 1.2,
   11.0, 1.2, 16.0;

  EXPECT_NO_THROW(stan::prob::multi_student_t_rng(2.0,mu,s,rng));
  EXPECT_THROW(stan::prob::multi_student_t_rng(-2.0,mu,s,rng),
               std::domain_error);

 s << 2.0, 3.0, 11.0,
   3.0, 9.0, 1.2,
   11.0, -1.2, 16.0;
  EXPECT_THROW(stan::prob::multi_student_t_rng(2.0,mu,s,rng),
               std::domain_error);

  mu << stan::math::positive_infinity(), 
    3.0,
    11.0;
 s << 2.0, 3.0, 11.0,
   3.0, 9.0, 1.2,
   11.0, 1.2, 16.0;
  EXPECT_THROW(stan::prob::multi_student_t_rng(2.0,mu,s,rng),
               std::domain_error);
}

TEST(ProbDistributionsMultiStudentT, marginalOneChiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  Matrix<double,Dynamic,Dynamic> mu(3,1);
  mu << 2.0, 
    3.0,
    11.0;

Matrix<double,Dynamic,Dynamic> s(3,3);
 s << 2.0, 3.0, 11.0,
   3.0, 9.0, 1.2,
   11.0, 1.2, 16.0;
  int N = 10000;
  int K = boost::math::round(2 * std::pow(N, 0.4));
  boost::math::students_t_distribution<>dist (3.0);
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
    a = stan::prob::multi_student_t_rng(3.0,mu,s,rng);
    a(0) = (a(0) - mu(0,0)) / std::sqrt(s(0,0));
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

TEST(ProbDistributionsMultiStudentT, marginalTwoChiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  Matrix<double,Dynamic,Dynamic> mu(3,1);
  mu << 2.0, 
    3.0,
    11.0;

Matrix<double,Dynamic,Dynamic> s(3,3);
 s << 2.0, 3.0, 11.0,
   3.0, 9.0, 1.2,
   11.0, 1.2, 16.0;
  int N = 10000;
  int K = boost::math::round(2 * std::pow(N, 0.4));
  boost::math::students_t_distribution<>dist (3.0);
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
    a = stan::prob::multi_student_t_rng(3.0,mu,s,rng);
    a(1) = (a(1) - mu(1,0)) / std::sqrt(s(1,1));
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

