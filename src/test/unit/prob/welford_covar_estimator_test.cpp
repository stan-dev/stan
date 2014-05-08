#include <stan/prob/welford_covar_estimator.hpp>
#include <gtest/gtest.h>

TEST(ProbWelfordCovarEstimator, restart) {
  
  const int n = 10;
  Eigen::VectorXd q = Eigen::VectorXd::Ones(n);
  
  const int n_learn = 10;
  
  stan::prob::welford_covar_estimator estimator(n);
  
  for (int i = 0; i < n_learn; ++i)
    estimator.add_sample(q);
  
  estimator.restart();
  
  EXPECT_EQ(0, estimator.num_samples());
  
  Eigen::VectorXd mean(n);
  estimator.sample_mean(mean);
  
  for (int i = 0; i < n; ++i)
    EXPECT_EQ(0, mean(i));
  
}

TEST(ProbWelfordCovarEstimator, num_samples) {
  
  const int n = 10;
  Eigen::VectorXd q = Eigen::VectorXd::Ones(n);
  
  const int n_learn = 10;
  
  stan::prob::welford_covar_estimator estimator(n);
  
  for (int i = 0; i < n_learn; ++i)
    estimator.add_sample(q);
  
  EXPECT_EQ(n_learn, estimator.num_samples());
  
}

TEST(ProbWelfordCovarEstimator, sample_mean) {
  
  const int n = 10;
  const int n_learn = 10;
  
  stan::prob::welford_covar_estimator estimator(n);
  
  for (int i = 0; i < n_learn; ++i) {
    Eigen::VectorXd q = Eigen::VectorXd::Constant(n, i);
    estimator.add_sample(q);
  }
  
  Eigen::VectorXd mean(n);
  estimator.sample_mean(mean);
  
  for (int i = 0; i < n; ++i)
    EXPECT_EQ(9.0 / 2.0, mean(i));
  
}

TEST(ProbWelfordCovarEstimator, sample_covariance) {
  
  const int n = 10;
  const int n_learn = 10;
  
  stan::prob::welford_covar_estimator estimator(n);
  
  for (int i = 0; i < n_learn; ++i) {
    Eigen::VectorXd q = Eigen::VectorXd::Constant(n, i);
    estimator.add_sample(q);
  }
  
  Eigen::MatrixXd covar(n, n);
  estimator.sample_covariance(covar);
  
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      EXPECT_EQ(55.0 / 6.0, covar(i, j));
  
}