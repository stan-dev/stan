#include <stan/mcmc/var_adapter.hpp>
#include <gtest/gtest.h>

TEST(McmcVarAdapter, learn_variance) {
  
  const int n = 10;
  std::vector<double> q(10, 0.0);
  Eigen::VectorXd var(Eigen::VectorXd::Zero(n));
  
  const int n_learn = 10;
  
  Eigen::VectorXd target_var(Eigen::VectorXd::Ones(n));
  target_var *= 5.0 / (n_learn + 5.0);
  
  stan::mcmc::var_adapter adapter(n);
  
  for (int i = 0; i < n_learn; ++i)
    adapter.learn_variance(var, q);
  
  for (int i = 0; i < n; ++i)
    EXPECT_EQ(target_var(i), var(i));
}