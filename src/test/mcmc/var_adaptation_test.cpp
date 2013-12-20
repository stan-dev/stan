#include <stan/mcmc/var_adaptation.hpp>
#include <gtest/gtest.h>

TEST(McmcVarAdaptation, learn_variance) {
  
  const int n = 10;
  Eigen::VectorXd q = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd var(Eigen::VectorXd::Zero(n));
  
  const int n_learn = 10;
  
  Eigen::VectorXd target_var(Eigen::VectorXd::Ones(n));
  target_var *= 1e-3 * 5.0 / (n_learn + 5.0);
  
  stan::mcmc::var_adaptation adapter(n);
  adapter.set_window_params(50, 0, 0, n_learn);
  
  for (int i = 0; i < n_learn; ++i)
    adapter.learn_variance(var, q);
  
  for (int i = 0; i < n; ++i)
    EXPECT_EQ(target_var(i), var(i));
}
