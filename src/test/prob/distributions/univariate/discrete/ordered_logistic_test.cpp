#include <gtest/gtest.h>
#include <stan/math/special_functions.hpp>
#include <stan/prob/distributions/univariate/discrete/ordered_logistic.hpp>

using Eigen::Matrix;
using Eigen::Dynamic;

typedef Eigen::Matrix<double,Eigen::Dynamic,1> vector_d;

vector_d
get_simplex(double lambda, 
           const vector_d& c) {
  using stan::math::inv_logit;
  int K = c.size() + 2;
  vector_d theta(K);
  // see p. 119, Gelman and Hill
  theta(0) = 1.0 - inv_logit(lambda);
  theta(1) = inv_logit(lambda) - inv_logit(lambda - c(0));
  for (int k = 2; k < (K - 1); ++k)
    theta(k) = inv_logit(lambda - c(k - 2)) - inv_logit(lambda - c(k - 1));
  theta(K-1) = inv_logit(lambda - c(K-3)); // - 0.0
  return theta;
}



TEST(ProbDistributions,ordered_logistic_vals) {
  using Eigen::Matrix;
  using Eigen::Dynamic;

  using stan::prob::ordered_logistic_log;
  using stan::math::inv_logit;

  int K = 5;
  Matrix<double,Dynamic,1> c(K-2);
  c << 0.9, 1.2, 2.6;
  double lambda = 1.1;
  
  vector_d theta = get_simplex(lambda,c);

  double sum = 0.0;
  for (int k = 0; k < theta.size(); ++k)
    sum += theta(k);
  EXPECT_FLOAT_EQ(1.0,sum);

  for (int k = 0; k < K; ++k)
    EXPECT_FLOAT_EQ(log(theta(k)),
                     ordered_logistic_log(k,lambda,c));
}


TEST(ProbDistributions,ordered_logistic_vals_2) {
  using Eigen::Matrix;
  using Eigen::Dynamic;

  using stan::prob::ordered_logistic_log;
  using stan::math::inv_logit;

  int K = 3;
  Matrix<double,Dynamic,1> c(K-2);
  c << 0.4;
  double lambda = -0.9;
  
  vector_d theta = get_simplex(lambda,c);

  double sum = 0.0;
  for (int k = 0; k < theta.size(); ++k)
    sum += theta(k);
  EXPECT_FLOAT_EQ(1.0,sum);

  for (int k = 0; k < K; ++k)
    EXPECT_FLOAT_EQ(log(theta(k)),
                     ordered_logistic_log(k,lambda,c));
}

TEST(ProbDistributions,ordered_logistic_default_policy) {
  Eigen::Matrix<double,Eigen::Dynamic,1> c(2);
  c << 0.1, 1.2;
  double lambda = 0.5;
  EXPECT_THROW(stan::prob::ordered_logistic_log(-1,lambda,c),
               std::domain_error);
}
