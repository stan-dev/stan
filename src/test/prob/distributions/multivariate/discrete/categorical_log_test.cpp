#include <gtest/gtest.h>
#include <limits>
#include <stan/prob/distributions/multivariate/discrete/categorical_log.hpp>
#include <boost/random/mersenne_twister.hpp>
#include<boost/math/distributions.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;

TEST(ProbDistributionsCategorical,Categorical) {
  Matrix<double,Dynamic,1> theta(3,1);
  theta << -1, 2, -10;
  EXPECT_FLOAT_EQ(-1, stan::prob::categorical_log_log(1,theta));
  EXPECT_FLOAT_EQ(2, stan::prob::categorical_log_log(2,theta));
  EXPECT_FLOAT_EQ(-10, stan::prob::categorical_log_log(3,theta));
}

TEST(ProbDistributionsCategorical,VectorInts) {
  using stan::prob::categorical_log_log;

  Matrix<double,Dynamic,1> theta(3,1);
  theta << -1, 2, -10;

  std::vector<int> xs0;
  EXPECT_FLOAT_EQ(0.0, categorical_log_log(xs0,theta));

  std::vector<int> xs(3);
  xs[0] = 1;
  xs[1] = 3;
  xs[2] = 3;
  EXPECT_FLOAT_EQ(-1 + -10 + -10, categorical_log_log(xs,theta));
}

TEST(ProbDistributionsCategorical,Propto) {
  Matrix<double,Dynamic,1> theta(3,1);
  theta << -1, 2, 10;
  EXPECT_FLOAT_EQ(0, stan::prob::categorical_log_log<true>(1,theta));
  EXPECT_FLOAT_EQ(0, stan::prob::categorical_log_log<true>(3,theta));
}


TEST(ProbDistributionsCategorical,DefaultPolicy) {
  using stan::prob::categorical_log_log;

  unsigned int n = 1;
  unsigned int N = 3;
  Matrix<double,Dynamic,1> theta(N,1);
  theta << 0.3, 0.5, 0.2;

  EXPECT_NO_THROW(categorical_log_log(N, theta));
  EXPECT_NO_THROW(categorical_log_log(n, theta));
  EXPECT_NO_THROW(categorical_log_log(2, theta));
  EXPECT_THROW(categorical_log_log(N+1, theta), std::domain_error);
  EXPECT_THROW(categorical_log_log(0, theta), std::domain_error);

  std::vector<int> xs(3);
  xs[0] = 1;
  xs[1] = 2;
  xs[2] = 1;

  theta(1) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(categorical_log_log(1, theta), std::domain_error);
  EXPECT_THROW(categorical_log_log(xs, theta), std::domain_error);

  theta(1) = std::numeric_limits<double>::infinity();
  EXPECT_THROW(categorical_log_log(1, theta), std::domain_error);
  EXPECT_THROW(categorical_log_log(xs, theta), std::domain_error);

  xs[1] = -1;
  EXPECT_THROW(categorical_log_log(xs, theta), std::domain_error);

  xs[1] = 2;
  xs[2] = 12;
  EXPECT_THROW(categorical_log_log(xs, theta), std::domain_error);
}

