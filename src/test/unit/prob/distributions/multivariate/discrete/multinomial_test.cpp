#include <gtest/gtest.h>
#include <stan/prob/distributions/multivariate/discrete/multinomial.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/fwd/matrix.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/agrad/rev/matrix.hpp>

using Eigen::Matrix;
using Eigen::Dynamic;

TEST(ProbDistributionsMultinomial,RNGSize) {
  boost::random::mt19937 rng;
  Matrix<double,Dynamic,1> theta(5);
  // error in 2.1.0 due to overflow in binomial call due to division
  theta << 0.3, 0.1, 0.2, 0.2, 0.2;  
  std::vector<int> sample = stan::prob::multinomial_rng(theta,10,rng);
  // bug in 2.1.0 returned 10 rather than 5 for returned size
  EXPECT_EQ(5U, sample.size());  
}

TEST(ProbDistributionsMultinomial,Multinomial) {
  std::vector<int> ns;
  ns.push_back(1);
  ns.push_back(2);
  ns.push_back(3);
  Matrix<double,Dynamic,1> theta(3,1);
  theta << 0.2, 0.3, 0.5;
  EXPECT_FLOAT_EQ(-2.002481, stan::prob::multinomial_log(ns,theta));
}
TEST(ProbDistributionsMultinomial,Propto) {
  std::vector<int> ns;
  ns.push_back(1);
  ns.push_back(2);
  ns.push_back(3);
  Matrix<double,Dynamic,1> theta(3,1);
  theta << 0.2, 0.3, 0.5;
  EXPECT_FLOAT_EQ(0.0, stan::prob::multinomial_log<true>(ns,theta));
}

using stan::prob::multinomial_log;

TEST(ProbDistributionsMultinomial,DefaultPolicy) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();

  std::vector<int> ns;
  ns.push_back(1);
  ns.push_back(2);
  ns.push_back(3);
  Matrix<double,Dynamic,1> theta(3,1);
  theta << 0.2, 0.3, 0.5;
  
  EXPECT_NO_THROW(multinomial_log(ns, theta));
  
  ns[1] = 0;
  EXPECT_NO_THROW(multinomial_log(ns, theta));
  ns[1] = -1;
  EXPECT_THROW(multinomial_log(ns, theta), std::domain_error);
  ns[1] = 1;

  theta(0) = 0.0;
  EXPECT_THROW(multinomial_log(ns, theta), std::domain_error);
  theta(0) = nan;
  EXPECT_THROW(multinomial_log(ns, theta), std::domain_error);
  theta(0) = inf;
  EXPECT_THROW(multinomial_log(ns, theta), std::domain_error);
  theta(0) = -inf;
  EXPECT_THROW(multinomial_log(ns, theta), std::domain_error);
  theta(0) = -1;
  theta(1) = 1.5;
  theta(2) = 0.5;
  EXPECT_THROW(multinomial_log(ns, theta), std::domain_error);
  theta(0) = 0.2;
  theta(1) = 0.3;
  theta(2) = 0.5;
  
  ns.resize(2);
  EXPECT_THROW(multinomial_log(ns, theta), std::domain_error);
}

TEST(ProbDistributionsMultinomial, zeros) {
  double result;
  std::vector<int> ns;
  ns.push_back(0);
  ns.push_back(1);
  ns.push_back(2);
  Matrix<double,Dynamic,1> theta(3,1);
  theta << 0.2, 0.3, 0.5;

  result = multinomial_log(ns, theta);
  EXPECT_FALSE(std::isnan(result));

  std::vector<int> ns2;
  ns2.push_back(0);
  ns2.push_back(0);
  ns2.push_back(0);
  
  double result2 = multinomial_log(ns2, theta);
  EXPECT_FLOAT_EQ(0.0, result2);
}

TEST(ProbDistributionsMultinomial, error_check) {
  boost::random::mt19937 rng;

  Matrix<double,Dynamic,1> theta(3);
  theta << 0.15, 0.45, 0.40;

  EXPECT_THROW(stan::prob::multinomial_rng(theta,-3,rng), std::domain_error);

  theta << 0.15, 0.45, 0.50;
  EXPECT_THROW(stan::prob::multinomial_rng(theta,3,rng), std::domain_error);
}

TEST(ProbDistributionsMultinomial, chiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  int M = 10;
  int trials = 1000;
  int N = M * trials;

  int K = 3;
  Matrix<double,Dynamic,1> theta(K);
  theta << 0.2, 0.35, 0.45;
  boost::math::chi_squared mydist(K-1);

  double expect[K];
  for (int i = 0 ; i < K; ++i)
    expect[i] = N * theta(i);

  int bin[K];
  for (int i = 0; i < K; ++i)
    bin[i] = 0;

  for (int count = 0; count < M; ++count) {
    std::vector<int> a = stan::prob::multinomial_rng(theta,trials,rng);
    for (int i = 0; i < K; ++i)
      bin[i] += a[i];
  }

  double chi = 0;
  for (int j = 0; j < K; j++)
    chi += ((bin[j] - expect[j]) * (bin[j] - expect[j])) / expect[j];
  
  EXPECT_TRUE(chi < quantile(complement(mydist, 1e-6)));
}

TEST(ProbDistributionsMultinomial,fvar_double) {
  using stan::agrad::fvar;
  std::vector<int> ns;
  ns.push_back(1);
  ns.push_back(2);
  ns.push_back(3);
  Matrix<fvar<double>,Dynamic,1> theta(3,1);
  theta << 0.2, 0.3, 0.5;
  for (int i = 0; i < 3; i++)
    theta(i).d_ = 1.0;

  EXPECT_FLOAT_EQ(-2.002481, stan::prob::multinomial_log(ns,theta).val_);
  EXPECT_FLOAT_EQ(17.666666, stan::prob::multinomial_log(ns,theta).d_);
}
TEST(ProbDistributionsMultinomial,fvar_var) {
  using stan::agrad::var;
  using stan::agrad::fvar;
  std::vector<int> ns;
  ns.push_back(1);
  ns.push_back(2);
  ns.push_back(3);
  Matrix<fvar<var>,Dynamic,1> theta(3,1);
  theta << 0.2, 0.3, 0.5;
  for (int i = 0; i < 3; i++)
    theta(i).d_ = 1.0;

  EXPECT_FLOAT_EQ(-2.002481, stan::prob::multinomial_log(ns,theta).val_.val());
  EXPECT_FLOAT_EQ(17.666666, stan::prob::multinomial_log(ns,theta).d_.val());
}

TEST(ProbDistributionsMultinomial,fvar_fvar_double) {
  using stan::agrad::fvar;
  std::vector<int> ns;
  ns.push_back(1);
  ns.push_back(2);
  ns.push_back(3);
  Matrix<fvar<fvar<double> >,Dynamic,1> theta(3,1);
  theta << 0.2, 0.3, 0.5;
  for (int i = 0; i < 3; i++)
    theta(i).d_.val_ = 1.0;

  EXPECT_FLOAT_EQ(-2.002481, stan::prob::multinomial_log(ns,theta).val_.val_);
  EXPECT_FLOAT_EQ(17.666666, stan::prob::multinomial_log(ns,theta).d_.val_);
}
TEST(ProbDistributionsMultinomial,fvar_fvar_var) {
  using stan::agrad::var;
  using stan::agrad::fvar;
  std::vector<int> ns;
  ns.push_back(1);
  ns.push_back(2);
  ns.push_back(3);
  Matrix<fvar<fvar<var> >,Dynamic,1> theta(3,1);
  theta << 0.2, 0.3, 0.5;
  for (int i = 0; i < 3; i++)
    theta(i).d_.val_ = 1.0;

  EXPECT_FLOAT_EQ(-2.002481, stan::prob::multinomial_log(ns,theta).val_.val_.val());
  EXPECT_FLOAT_EQ(17.666666, stan::prob::multinomial_log(ns,theta).d_.val_.val());
}
