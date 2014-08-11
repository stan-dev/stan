#include <gtest/gtest.h>
#include <stan/prob/distributions/multivariate/discrete/categorical.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/fwd/matrix.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/agrad/rev/matrix.hpp>

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

TEST(ProbDistributionsCategorical,VectorInt) {
  Matrix<double,Dynamic,1> theta(3,1);
  theta << 0.3, 0.5, 0.2;
  std::vector<int> xs0;
  EXPECT_FLOAT_EQ(0.0, stan::prob::categorical_log(xs0,theta));

  std::vector<int> xs(3);
  xs[0] = 1;
  xs[1] = 3;
  xs[2] = 1;
  
  EXPECT_FLOAT_EQ(log(0.3) + log(0.2) + log(0.3),
                  stan::prob::categorical_log(xs,theta));
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

  std::vector<int> ns(3);
  ns[0] = 3;
  ns[1] = 2;
  ns[2] = 3;
  EXPECT_THROW(categorical_log(ns,theta), std::domain_error);
  
  theta << 0.3, 0.5, 0.2;
  EXPECT_NO_THROW(categorical_log(ns,theta));
  ns[1] = -1;
  EXPECT_THROW(categorical_log(ns,theta), std::domain_error);

  ns[1] = 1;
  ns[2] = 12;
  EXPECT_THROW(categorical_log(ns,theta), std::domain_error);
  
  
}

TEST(ProbDistributionsCategorical, error_check) {
  boost::random::mt19937 rng;
  
  Matrix<double,Dynamic,Dynamic> theta(3,1);
  theta << 0.15, 
    0.45,
    0.50;

  EXPECT_THROW(stan::prob::categorical_rng(theta,rng),std::domain_error);
}

TEST(ProbDistributionsCategorical, chiSquareGoodnessFitTest) {
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

TEST(ProbDistributionsCategorical,fvar_double) {
  using stan::agrad::fvar;
  Matrix<fvar<double>,Dynamic,1> theta(3,1);
  theta << 0.3, 0.5, 0.2;
  for (int i = 0; i < 3; i++)
    theta(i).d_ = 1.0;

  EXPECT_FLOAT_EQ(std::log(0.3), stan::prob::categorical_log(1,theta).val_);
  EXPECT_FLOAT_EQ(std::log(0.5), stan::prob::categorical_log(2,theta).val_);
  EXPECT_FLOAT_EQ(std::log(0.2), stan::prob::categorical_log(3,theta).val_);
  EXPECT_FLOAT_EQ(1.0/0.3, stan::prob::categorical_log(1,theta).d_);
  EXPECT_FLOAT_EQ(1.0/0.5, stan::prob::categorical_log(2,theta).d_);
  EXPECT_FLOAT_EQ(1.0/0.2, stan::prob::categorical_log(3,theta).d_);
}
TEST(ProbDistributionsCategorical,fvar_double_vector) {
  using stan::agrad::fvar;
  Matrix<fvar<double>,Dynamic,1> theta(3,1);
  theta << 0.3, 0.5, 0.2;
  for (int i = 0; i < 3; i++)
    theta(i).d_ = 1.0;

  std::vector<int> xs(3);
  xs[0] = 1;
  xs[1] = 3;
  xs[2] = 1;
  
  EXPECT_FLOAT_EQ(log(0.3) + log(0.2) + log(0.3),
                  stan::prob::categorical_log(xs,theta).val_);
  EXPECT_FLOAT_EQ(1.0 / 0.3 + 1.0 / 0.2 + 1.0 / 0.3,
                  stan::prob::categorical_log(xs,theta).d_);
}

TEST(ProbDistributionsCategorical,fvar_var) {
  using stan::agrad::var;
  using stan::agrad::fvar;
  Matrix<fvar<var>,Dynamic,1> theta(3,1);
  theta << 0.3, 0.5, 0.2;
  for (int i = 0; i < 3; i++)
    theta(i).d_ = 1.0;

  EXPECT_FLOAT_EQ(std::log(0.3), stan::prob::categorical_log(1,theta).val_.val());
  EXPECT_FLOAT_EQ(std::log(0.5), stan::prob::categorical_log(2,theta).val_.val());
  EXPECT_FLOAT_EQ(std::log(0.2), stan::prob::categorical_log(3,theta).val_.val());
  EXPECT_FLOAT_EQ(1.0/0.3, stan::prob::categorical_log(1,theta).d_.val());
  EXPECT_FLOAT_EQ(1.0/0.5, stan::prob::categorical_log(2,theta).d_.val());
  EXPECT_FLOAT_EQ(1.0/0.2, stan::prob::categorical_log(3,theta).d_.val());
}
TEST(ProbDistributionsCategorical,fvar_var_vector) {
  using stan::agrad::var;
  using stan::agrad::fvar;
  Matrix<fvar<var>,Dynamic,1> theta(3,1);
  theta << 0.3, 0.5, 0.2;
  for (int i = 0; i < 3; i++)
    theta(i).d_ = 1.0;

  std::vector<int> xs(3);
  xs[0] = 1;
  xs[1] = 3;
  xs[2] = 1;
  
  EXPECT_FLOAT_EQ(log(0.3) + log(0.2) + log(0.3),
                  stan::prob::categorical_log(xs,theta).val_.val());
  EXPECT_FLOAT_EQ(1.0 / 0.3 + 1.0 / 0.2 + 1.0 / 0.3,
                  stan::prob::categorical_log(xs,theta).d_.val());
}

TEST(ProbDistributionsCategorical,fvar_fvar_double) {
  using stan::agrad::fvar;
  Matrix<fvar<fvar<double> >,Dynamic,1> theta(3,1);
  theta << 0.3, 0.5, 0.2;
  for (int i = 0; i < 3; i++)
    theta(i).d_.val_ = 1.0;

  EXPECT_FLOAT_EQ(std::log(0.3), stan::prob::categorical_log(1,theta).val_.val_);
  EXPECT_FLOAT_EQ(std::log(0.5), stan::prob::categorical_log(2,theta).val_.val_);
  EXPECT_FLOAT_EQ(std::log(0.2), stan::prob::categorical_log(3,theta).val_.val_);
  EXPECT_FLOAT_EQ(1.0/0.3, stan::prob::categorical_log(1,theta).d_.val_);
  EXPECT_FLOAT_EQ(1.0/0.5, stan::prob::categorical_log(2,theta).d_.val_);
  EXPECT_FLOAT_EQ(1.0/0.2, stan::prob::categorical_log(3,theta).d_.val_);
}
TEST(ProbDistributionsCategorical,fvar_fvar_double_vector) {
  using stan::agrad::fvar;
  Matrix<fvar<fvar<double> >,Dynamic,1> theta(3,1);
  theta << 0.3, 0.5, 0.2;
  for (int i = 0; i < 3; i++)
    theta(i).d_.val_ = 1.0;

  std::vector<int> xs(3);
  xs[0] = 1;
  xs[1] = 3;
  xs[2] = 1;
  
  EXPECT_FLOAT_EQ(log(0.3) + log(0.2) + log(0.3),
                  stan::prob::categorical_log(xs,theta).val_.val_);
  EXPECT_FLOAT_EQ(1.0 / 0.3 + 1.0 / 0.2 + 1.0 / 0.3,
                  stan::prob::categorical_log(xs,theta).d_.val_);
}

TEST(ProbDistributionsCategorical,fvar_fvar_var) {
  using stan::agrad::var;
  using stan::agrad::fvar;
  Matrix<fvar<fvar<var> >,Dynamic,1> theta(3,1);
  theta << 0.3, 0.5, 0.2;
  for (int i = 0; i < 3; i++)
    theta(i).d_.val_ = 1.0;

  EXPECT_FLOAT_EQ(std::log(0.3), stan::prob::categorical_log(1,theta).val_.val_.val());
  EXPECT_FLOAT_EQ(std::log(0.5), stan::prob::categorical_log(2,theta).val_.val_.val());
  EXPECT_FLOAT_EQ(std::log(0.2), stan::prob::categorical_log(3,theta).val_.val_.val());
  EXPECT_FLOAT_EQ(1.0/0.3, stan::prob::categorical_log(1,theta).d_.val_.val());
  EXPECT_FLOAT_EQ(1.0/0.5, stan::prob::categorical_log(2,theta).d_.val_.val());
  EXPECT_FLOAT_EQ(1.0/0.2, stan::prob::categorical_log(3,theta).d_.val_.val());
}
TEST(ProbDistributionsCategorical,fvar_fvar_var_vector) {
  using stan::agrad::var;
  using stan::agrad::fvar;
  Matrix<fvar<fvar<var> >,Dynamic,1> theta(3,1);
  theta << 0.3, 0.5, 0.2;
  for (int i = 0; i < 3; i++)
    theta(i).d_.val_ = 1.0;

  std::vector<int> xs(3);
  xs[0] = 1;
  xs[1] = 3;
  xs[2] = 1;
  
  EXPECT_FLOAT_EQ(log(0.3) + log(0.2) + log(0.3),
                  stan::prob::categorical_log(xs,theta).val_.val_.val());
  EXPECT_FLOAT_EQ(1.0 / 0.3 + 1.0 / 0.2 + 1.0 / 0.3,
                  stan::prob::categorical_log(xs,theta).d_.val_.val());
}
