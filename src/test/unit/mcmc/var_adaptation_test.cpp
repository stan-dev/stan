#include <stan/mcmc/var_adaptation.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <gtest/gtest.h>

TEST(McmcVarAdaptation, learn_variance) {
  stan::test::unit::instrumented_logger logger;

  const int n = 10;
  Eigen::VectorXd q = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd var(Eigen::VectorXd::Zero(n));

  const int n_learn = 10;

  Eigen::VectorXd target_var(Eigen::VectorXd::Ones(n));
  target_var *= 1e-3 * 5.0 / (n_learn + 5.0);

  stan::mcmc::var_adaptation adapter(n);
  adapter.set_window_params(30, 0, 0, n_learn, logger);

  for (int i = 0; i < n_learn - 1; ++i) {
    EXPECT_FALSE(adapter.learn_variance(var, q));
  }
  // Learn variance should return true at end of first window
  EXPECT_TRUE(adapter.learn_variance(var, q));

  for (int i = 0; i < n; ++i)
    EXPECT_EQ(target_var(i), var(i));

  // Make sure learn_variance doesn't return true after second window
  // (adaptation finished)
  for (int i = 0; i < 2 * n_learn; ++i) {
    EXPECT_FALSE(adapter.learn_variance(var, q));
  }
  EXPECT_TRUE(adapter.finished());

  EXPECT_EQ(0, logger.call_count());
}
