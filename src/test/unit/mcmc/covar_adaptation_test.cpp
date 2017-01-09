#include <stan/mcmc/covar_adaptation.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <gtest/gtest.h>

TEST(McmcCovarAdaptation, learn_covariance) {
  std::stringstream ss;
  stan::callbacks::stream_writer writer(ss);
  
  const int n = 10;
  Eigen::VectorXd q = Eigen::VectorXd::Zero(n);
  Eigen::MatrixXd covar(Eigen::MatrixXd::Zero(n, n));
  
  const int n_learn = 10;
  
  Eigen::MatrixXd target_covar(Eigen::MatrixXd::Identity(n, n));
  target_covar *= 1e-3 * 5.0 / (n_learn + 5.0);
  
  stan::mcmc::covar_adaptation adapter(n);
  adapter.set_window_params(50, 0, 0, n_learn, writer);
  
  for (int i = 0; i < n_learn; ++i)
    adapter.learn_covariance(covar, q);
  
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      EXPECT_EQ(target_covar(i, j), covar(i, j));
    }
  }
  EXPECT_EQ("", ss.str());
}
