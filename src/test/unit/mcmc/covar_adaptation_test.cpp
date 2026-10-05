#include <stan/mcmc/covar_adaptation.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <gtest/gtest.h>

TEST(McmcCovarAdaptation, learn_covariance) {
  stan::test::unit::instrumented_logger logger;

  const int n = 10;
  Eigen::VectorXd q = Eigen::VectorXd::Zero(n);
  Eigen::MatrixXd covar(Eigen::MatrixXd::Zero(n, n));

  const int n_learn = 10;

  Eigen::MatrixXd target_covar(Eigen::MatrixXd::Identity(n, n));
  target_covar *= 1e-3 * 5.0 / (n_learn + 5.0);

  stan::mcmc::covar_adaptation adapter(n);
  adapter.set_window_params(30, 0, 0, n_learn, logger);

  for (int i = 0; i < n_learn - 1; ++i) {
    EXPECT_FALSE(adapter.learn_covariance(covar, q));
  }
  // Learn covariance should return true at end of first window
  EXPECT_TRUE(adapter.learn_covariance(covar, q));

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      EXPECT_EQ(target_covar(i, j), covar(i, j));
    }
  }

  // Make sure learn_covariance doesn't return true after second window
  // (adaptation finished)
  for (int i = 0; i < 2 * n_learn; ++i) {
    EXPECT_FALSE(adapter.learn_covariance(covar, q));
  }
  EXPECT_TRUE(adapter.finished());

  EXPECT_EQ(0, logger.call_count());
}
