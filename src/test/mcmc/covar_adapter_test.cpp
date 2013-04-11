#include <stan/mcmc/covar_adapter.hpp>
#include <gtest/gtest.h>

TEST(McmcCovarAdapter, learn_covariance) {

  const int n = 10;
  std::vector<double> q(10, 0.0);
  Eigen::MatrixXd covar(Eigen::MatrixXd::Zero(n, n));
  
  const int n_learn = 10;
  
  Eigen::MatrixXd target_covar(Eigen::MatrixXd::Identity(n, n));
  target_covar *= 5.0 / (n_learn + 5.0);
  
  stan::mcmc::covar_adapter adapter(n);
  
  for (int i = 0; i < n_learn; ++i)
    adapter.learn_covariance(covar, q);
  
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      EXPECT_EQ(target_covar(i, j), covar(i, j));
    }
  }

}