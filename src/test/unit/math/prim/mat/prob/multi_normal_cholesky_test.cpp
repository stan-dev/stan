#include <gtest/gtest.h>
#include <stan/math/prim/mat/prob/multi_normal_cholesky_log.hpp>
#include <stan/math/prim/mat/prob/multi_normal_cholesky_rng.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <stan/math/rev/core/operator_addition.hpp>
#include <stan/math/rev/core/operator_division.hpp>
#include <stan/math/rev/core/operator_divide_equal.hpp>
#include <stan/math/rev/core/operator_greater_than.hpp>
#include <stan/math/rev/core/operator_greater_than_or_equal.hpp>
#include <stan/math/rev/core/operator_less_than.hpp>
#include <stan/math/rev/core/operator_less_than_or_equal.hpp>
#include <stan/math/rev/core/operator_multiplication.hpp>
#include <stan/math/rev/core/operator_multiply_equal.hpp>
#include <stan/math/rev/core/operator_plus_equal.hpp>
#include <stan/math/rev/core/operator_minus_equal.hpp>
#include <stan/math/rev/core/operator_equal.hpp>
#include <stan/math/rev/core/operator_not_equal.hpp>
#include <stan/math/rev/core/operator_subtraction.hpp>
#include <stan/math/rev/core/operator_unary_negative.hpp>
#include <stan/math/fwd/core/operator_addition.hpp>
#include <stan/math/fwd/core/operator_division.hpp>
#include <stan/math/fwd/core/operator_greater_than.hpp>
#include <stan/math/fwd/core/operator_greater_than_or_equal.hpp>
#include <stan/math/fwd/core/operator_less_than.hpp>
#include <stan/math/fwd/core/operator_less_than_or_equal.hpp>
#include <stan/math/fwd/core/operator_multiplication.hpp>
#include <stan/math/fwd/core/operator_subtraction.hpp>
#include <stan/math/fwd/core/operator_unary_minus.hpp>
#include <stan/math/fwd/core/operator_equal.hpp>
#include <stan/math/fwd/core/operator_not_equal.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/value_of_rec.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/fwd/mat/fun/sum.hpp>
#include <stan/math/fwd/mat/fun/multiply.hpp>
#include <stan/math/fwd/mat/fun/dot_self.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <stan/math/rev/mat/fun/dot_self.hpp>
#include <stan/math/rev/mat/fun/columns_dot_self.hpp>
#include <stan/math/rev/scal/fun/sqrt.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;
using std::vector;

TEST(ProbDistributionsMultiNormalCholesky,NotVectorized) {
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
TEST(ProbDistributionsMultiNormalCholesky,Vectorized) {
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
  Matrix<double,Dynamic,Dynamic> L = Sigma.llt().matrixL();
    
  //y and mu vectorized
  EXPECT_FLOAT_EQ(-11.928077-6.5378327, stan::prob::multi_normal_cholesky_log(vec_y,vec_mu,L));
  EXPECT_FLOAT_EQ(-11.928077-6.5378327, stan::prob::multi_normal_cholesky_log(vec_y_t,vec_mu,L));
  EXPECT_FLOAT_EQ(-11.928077-6.5378327, stan::prob::multi_normal_cholesky_log(vec_y,vec_mu_t,L));
  EXPECT_FLOAT_EQ(-11.928077-6.5378327, stan::prob::multi_normal_cholesky_log(vec_y_t,vec_mu_t,L));

  //y vectorized
  EXPECT_FLOAT_EQ(-10.44027-6.537833, stan::prob::multi_normal_cholesky_log(vec_y,mu,L));
  EXPECT_FLOAT_EQ(-10.44027-6.537833, stan::prob::multi_normal_cholesky_log(vec_y_t,mu,L));
  EXPECT_FLOAT_EQ(-10.44027-6.537833, stan::prob::multi_normal_cholesky_log(vec_y,mu_t,L));
  EXPECT_FLOAT_EQ(-10.44027-6.537833, stan::prob::multi_normal_cholesky_log(vec_y_t,mu_t,L));

  //mu vectorized
  EXPECT_FLOAT_EQ(-6.26954-6.537833, stan::prob::multi_normal_cholesky_log(y,vec_mu,L));
  EXPECT_FLOAT_EQ(-6.26954-6.537833, stan::prob::multi_normal_cholesky_log(y_t,vec_mu,L));
  EXPECT_FLOAT_EQ(-6.26954-6.537833, stan::prob::multi_normal_cholesky_log(y,vec_mu_t,L));
  EXPECT_FLOAT_EQ(-6.26954-6.537833, stan::prob::multi_normal_cholesky_log(y_t,vec_mu_t,L));
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
  Matrix<double,Dynamic,Dynamic> L = sigma.llt().matrixL();
  EXPECT_NO_THROW(stan::prob::multi_normal_cholesky_rng(mu, L, rng));

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
  Matrix<double,Dynamic,Dynamic> L = sigma.llt().matrixL();
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
    a = stan::prob::multi_normal_cholesky_rng(mu,L,rng);
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
  Matrix<double,Dynamic,Dynamic> L = sigma.llt().matrixL();
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
    a = stan::prob::multi_normal_cholesky_rng(mu,L,rng);
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
  Matrix<double,Dynamic,Dynamic> L = sigma.llt().matrixL();
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
    a = stan::prob::multi_normal_cholesky_rng(mu,L,rng);
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

TEST(ProbDistributionsMultiNormalCholesky,fvar_double) {
  using stan::agrad::fvar;
  Matrix<fvar<double>,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<fvar<double>,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<fvar<double>,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;

  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  for (int i = 0; i < 3; i++) {
    y(i).d_ = 1.0;
    mu(i).d_ = 1.0;
    for (int j = 0; j < 3; j++)
      Sigma(i,j).d_ = 1.0;
  }

  Matrix<fvar<double>,Dynamic,Dynamic> L = Sigma.llt().matrixL();
  EXPECT_FLOAT_EQ(-11.73908, stan::prob::multi_normal_cholesky_log(y,mu,L).val_);
  EXPECT_FLOAT_EQ(0.54899865, stan::prob::multi_normal_cholesky_log(y,mu,L).d_);
}

TEST(ProbDistributionsMultiNormalCholesky,fvar_var) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  Matrix<fvar<var>,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<fvar<var>,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<fvar<var>,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;

  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  for (int i = 0; i < 3; i++) {
    y(i).d_ = 1.0;
    mu(i).d_ = 1.0;
    for (int j = 0; j < 3; j++)
      Sigma(i,j).d_ = 1.0;
  }

  Matrix<fvar<var>,Dynamic,Dynamic> L = Sigma.llt().matrixL();
  EXPECT_FLOAT_EQ(-11.73908, stan::prob::multi_normal_cholesky_log(y,mu,L).val_.val());
  EXPECT_FLOAT_EQ(0.54899865, stan::prob::multi_normal_cholesky_log(y,mu,L).d_.val());
}

TEST(ProbDistributionsMultiNormalCholesky,fvar_fvar_double) {
  using stan::agrad::fvar;
  Matrix<fvar<fvar<double> >,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<fvar<fvar<double> >,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<fvar<fvar<double> >,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;

  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  for (int i = 0; i < 3; i++) {
    y(i).d_.val_ = 1.0;
    mu(i).d_.val_ = 1.0;
    for (int j = 0; j < 3; j++)
      Sigma(i,j).d_.val_ = 1.0;
  }

  Matrix<fvar<fvar<double> >,Dynamic,Dynamic> L = Sigma.llt().matrixL();
  EXPECT_FLOAT_EQ(-11.73908, stan::prob::multi_normal_cholesky_log(y,mu,L).val_.val_);
  EXPECT_FLOAT_EQ(0.54899865, stan::prob::multi_normal_cholesky_log(y,mu,L).d_.val_);
}

TEST(ProbDistributionsMultiNormalCholesky,fvar_fvar_var) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  Matrix<fvar<fvar<var> >,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<fvar<fvar<var> >,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<fvar<fvar<var> >,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;

  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  for (int i = 0; i < 3; i++) {
    y(i).d_.val_ = 1.0;
    mu(i).d_.val_ = 1.0;
    for (int j = 0; j < 3; j++)
      Sigma(i,j).d_.val_ = 1.0;
  }

  Matrix<fvar<fvar<var> >,Dynamic,Dynamic> L = Sigma.llt().matrixL();
  EXPECT_FLOAT_EQ(-11.73908, stan::prob::multi_normal_cholesky_log(y,mu,L).val_.val_.val());
  EXPECT_FLOAT_EQ(0.54899865, stan::prob::multi_normal_cholesky_log(y,mu,L).d_.val_.val());
}
