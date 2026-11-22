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
  adapter.set_window_params(50, 0, 0, n_learn, logger);

  for (int i = 0; i < n_learn; ++i)
    adapter.learn_variance(var, q);

  for (int i = 0; i < n; ++i)
    EXPECT_EQ(target_var(i), var(i));

  EXPECT_EQ(0, logger.call_count());
}

TEST(McmcVarAdaptation, learn_variance_reduced_windows) {
  stan::test::unit::instrumented_logger logger;

  const int n = 10;
  Eigen::VectorXd q = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd var(Eigen::VectorXd::Zero(n));

  // init_buffer 75, term_buffer 50 and window 25 do not fit into 100 warmup
  // iterations and are reduced to 15 / 10 / 75
  stan::mcmc::var_adaptation adapter(n);
  adapter.set_window_params(100, 75, 50, 25, logger);

  for (int i = 0; i < 100; ++i) {
    adapter.learn_variance(var, q);
  }

  // Estimated once, from the 75 draws of the reduced window
  for (int i = 0; i < n; ++i) {
    EXPECT_DOUBLE_EQ(1e-3 * 5.0 / (75 + 5.0), var(i));
  }
}
