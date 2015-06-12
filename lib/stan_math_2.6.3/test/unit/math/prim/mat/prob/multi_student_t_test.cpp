#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/prob/multi_student_t_log.hpp>
#include <stan/math/prim/mat/prob/multi_student_t_rng.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;
using std::vector;
using stan::math::multi_student_t_log;

TEST(ProbDistributionsMultiStudentT,NotVectorized) {
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
TEST(ProbDistributionsMultiStudentT,Vectorized) {
  vector< Matrix<double,Dynamic,1> > vec_y(2);
  vector< Matrix<double,1,Dynamic> > vec_y_t(2);
  Matrix<double,Dynamic,1> y(3);
  Matrix<double,1,Dynamic> y_t(3);
  y << 3.0, -2.0, 10.0;
  vec_y[0] = y;
  vec_y_t[0] = y;
  y << 3.0, -1.0, 5.0;
  vec_y[1] = y;
  vec_y_t[1] = y;
  y_t = y;
  
  vector< Matrix<double,Dynamic,1> > vec_mu(2);
  vector< Matrix<double,1,Dynamic> > vec_mu_t(2);
  Matrix<double,Dynamic,1> mu(3);
  Matrix<double,1,Dynamic> mu_t(3);
  mu << 2.0, -1.0, 4.0;
  vec_mu[0] = mu;
  vec_mu_t[0] = mu;
  mu << 1.0, -3.0, 4.0;
  vec_mu[1] = mu;
  vec_mu_t[1] = mu;
  mu_t = mu;
  
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 10.0, -3.0, 0.0,
    -3.0,  5.0, 0.0,
    0.0, 0.0, 5.0;

  double nu = 4.0;
    
  //y and mu vectorized
  EXPECT_FLOAT_EQ(-8.92867-6.81839, stan::math::multi_student_t_log(vec_y,nu,vec_mu,Sigma));
  EXPECT_FLOAT_EQ(-8.92867-6.81839, stan::math::multi_student_t_log(vec_y_t,nu,vec_mu,Sigma));
  EXPECT_FLOAT_EQ(-8.92867-6.81839, stan::math::multi_student_t_log(vec_y,nu,vec_mu_t,Sigma));
  EXPECT_FLOAT_EQ(-8.92867-6.81839, stan::math::multi_student_t_log(vec_y_t,nu,vec_mu_t,Sigma));

  //y vectorized
  EXPECT_FLOAT_EQ(-9.167054-6.81839, stan::math::multi_student_t_log(vec_y,nu,mu,Sigma));
  EXPECT_FLOAT_EQ(-9.167054-6.81839, stan::math::multi_student_t_log(vec_y_t,nu,mu,Sigma));
  EXPECT_FLOAT_EQ(-9.167054-6.81839, stan::math::multi_student_t_log(vec_y,nu,mu_t,Sigma));
  EXPECT_FLOAT_EQ(-9.167054-6.81839, stan::math::multi_student_t_log(vec_y_t,nu,mu_t,Sigma));

  //mu vectorized
  EXPECT_FLOAT_EQ(-5.528012-6.81839, stan::math::multi_student_t_log(y,nu,vec_mu,Sigma));
  EXPECT_FLOAT_EQ(-5.528012-6.81839, stan::math::multi_student_t_log(y_t,nu,vec_mu,Sigma));
  EXPECT_FLOAT_EQ(-5.528012-6.81839, stan::math::multi_student_t_log(y,nu,vec_mu_t,Sigma));
  EXPECT_FLOAT_EQ(-5.528012-6.81839, stan::math::multi_student_t_log(y_t,nu,vec_mu_t,Sigma));  
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


TEST(ProbDistributionsMultiStudentT,ErrorSize1) {
  Matrix<double,Dynamic,1> y(2,1);
  y << 2.0, -2.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  double nu = 4.0;

  EXPECT_THROW(multi_student_t_log(y,nu,mu,Sigma),std::invalid_argument);
}

TEST(ProbDistributionsMultiStudentT,ErrorSize2) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,2);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0;
  double nu = 4.0;

  EXPECT_THROW(multi_student_t_log(y,nu,mu,Sigma),std::invalid_argument);
}

TEST(ProbDistributionsMultiStudentT,ErrorSize3) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(2,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0;
  double nu = 4.0;

  EXPECT_THROW(multi_student_t_log(y,nu,mu,Sigma),std::invalid_argument);
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

  EXPECT_NO_THROW(stan::math::multi_student_t_rng(2.0,mu,s,rng));
  EXPECT_THROW(stan::math::multi_student_t_rng(-2.0,mu,s,rng),
               std::domain_error);

 s << 2.0, 3.0, 11.0,
   3.0, 9.0, 1.2,
   11.0, -1.2, 16.0;
  EXPECT_THROW(stan::math::multi_student_t_rng(2.0,mu,s,rng),
               std::domain_error);

  mu << stan::math::positive_infinity(), 
    3.0,
    11.0;
 s << 2.0, 3.0, 11.0,
   3.0, 9.0, 1.2,
   11.0, 1.2, 16.0;
  EXPECT_THROW(stan::math::multi_student_t_rng(2.0,mu,s,rng),
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
    a = stan::math::multi_student_t_rng(3.0,mu,s,rng);
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
    a = stan::math::multi_student_t_rng(3.0,mu,s,rng);
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
